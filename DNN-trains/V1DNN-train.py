import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

topology_map = {
    'LIMIT CYCLE':        0,
    'DAMPED OSCILLATION': 1,
    'STEADY STATE':       2,
    'UNDETERMINED':       3
}

best_val_acc = 0.0

class LTInetDataset(Dataset):
    def __init__(self, data_dirs, max_per_dir=None):
        self.files = []
        for d in data_dirs:
            folder_files = sorted([
                os.path.join(d, f)
                for f in os.listdir(d)
                if f.endswith('.mat')
            ])
            if max_per_dir is not None:
                folder_files = folder_files[:max_per_dir]
            self.files += folder_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = scipy.io.loadmat(self.files[idx])

        rx = data['resid_dx'].squeeze()
        ry = data['resid_dy'].squeeze()
        rx = (rx - rx.mean()) / (rx.std() + 1e-8)
        ry = (ry - ry.mean()) / (ry.std() + 1e-8)

        target_length = 501
        if len(rx) < target_length:
            pad = target_length - len(rx)
            rx  = np.concatenate([rx, np.zeros(pad)])
            ry  = np.concatenate([ry, np.zeros(pad)])

        residual = np.stack([rx, ry], axis=0).astype(np.float32)

        xi_raw = data['Xi_ternary'].astype(np.float32)
        if xi_raw.shape[0] < 9:
            pad    = np.zeros((9 - xi_raw.shape[0], 2), dtype=np.float32)
            xi_raw = np.vstack([xi_raw, pad])

        topo_str   = str(data['topology'][0]).strip()
        topo_label = topology_map.get(topo_str, 3)

        struct_label = int(data['structure_label'].squeeze())
        label_remap = {0:0, 1:1, 2:2, 3:3, 6:4, 7:5}
        struct_label = label_remap.get(struct_label, struct_label)
        return (
            torch.tensor(residual),
            torch.tensor(xi_raw),
            torch.tensor(topo_label,   dtype=torch.long),
            torch.tensor(struct_label, dtype=torch.long)
        )


class Branch1_ResidualCNN(nn.Module):
    def __init__(self, n_channels=2, n_filters=32, output_dim=64):
        super().__init__()

        self.stream_small = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel_size=5,  padding=2),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters,  n_filters, kernel_size=5,  padding=2),
            nn.BatchNorm1d(n_filters), nn.ReLU()
        )
        self.stream_medium = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel_size=21, padding=10),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters,  n_filters, kernel_size=21, padding=10),
            nn.BatchNorm1d(n_filters), nn.ReLU()
        )
        self.stream_large = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel_size=51, padding=25),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters,  n_filters, kernel_size=51, padding=25),
            nn.BatchNorm1d(n_filters), nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.projection = nn.Sequential(
            nn.Linear(3 * n_filters, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        s = self.gap(self.stream_small(x)).squeeze(-1)
        m = self.gap(self.stream_medium(x)).squeeze(-1)
        l = self.gap(self.stream_large(x)).squeeze(-1)
        return self.projection(torch.cat([s, m, l], dim=1))


class Branch2_TermAttention(nn.Module):
    def __init__(self, n_terms=9, n_equations=2, embed_dim=16, output_dim=32):
        super().__init__()

        self.embed = nn.Linear(n_terms, embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=2,
            batch_first=True
        )

        self.projection = nn.Sequential(
            nn.Linear(n_equations * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        tokens = self.embed(x)
        attn_out, _ = self.attention(tokens, tokens, tokens)
        flat = attn_out.reshape(attn_out.shape[0], -1)
        return self.projection(flat)


class Branch3_TopologyEmbed(nn.Module):
    def __init__(self, n_classes=4, embed_dim=16, output_dim=16):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.proj  = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.proj(self.embed(x))


class LTInet(nn.Module):
    def __init__(self, n_structure_classes=3):
        super().__init__()

        self.branch1 = Branch1_ResidualCNN(n_channels=2,   output_dim=64)
        self.branch2 = Branch2_TermAttention(n_terms=9,    output_dim=32)
        self.branch3 = Branch3_TopologyEmbed(n_classes=4,  output_dim=16)

        merge_dim = 64 + 32 + 16

        self.head = nn.Sequential(
            nn.Linear(merge_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_structure_classes)
        )

    def forward(self, residual, xi_ternary, topo_label):
        b1 = self.branch1(residual)
        b2 = self.branch2(xi_ternary)
        b3 = self.branch3(topo_label)
        return self.head(torch.cat([b1, b2, b3], dim=1))


def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for residual, xi, topo, label in loader:
        residual = residual.to(device)
        xi       = xi.to(device)
        topo     = topo.to(device)
        label    = label.to(device)

        optimizer.zero_grad()
        logits = model(residual, xi, topo)
        loss   = loss_fn(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (logits.argmax(dim=1) == label).sum().item()
        total      += label.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for residual, xi, topo, label in loader:
            residual = residual.to(device)
            xi       = xi.to(device)
            topo     = topo.to(device)
            label    = label.to(device)

            logits = model(residual, xi, topo)
            loss   = loss_fn(logits, label)

            total_loss += loss.item()
            correct    += (logits.argmax(dim=1) == label).sum().item()
            total      += label.size(0)

    return total_loss / len(loader), correct / total


if __name__ == '__main__':
  
    
  
    DATA_ROOT = r'D:\LTInetV1 NNdata'
    data_dirs = [
    os.path.join(DATA_ROOT, 'Lat1_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'Lat1_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'Lat1_Sys4_NNdata'),

    os.path.join(DATA_ROOT, 'Lat2_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'Lat2_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'Lat2_Sys4_NNdata'),

    os.path.join(DATA_ROOT, 'Lat3_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'Lat3_Sys3_NNdata'),

    os.path.join(DATA_ROOT, 'Lat4_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'Lat4_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'Lat4_Sys3_NNdata'),
    os.path.join(DATA_ROOT, 'Lat4_Sys4_NNdata'),

    os.path.join(DATA_ROOT, 'Lat6_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'Lat6_Sys2_NNdata'),

    os.path.join(DATA_ROOT, 'Lat7_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'Lat7_Sys2_NNdata'),
]

    device   = torch.device('cpu')
    torch.set_num_threads(8)

    dataset = LTInetDataset(data_dirs, max_per_dir=200)
    n_train    = int(0.8 * len(dataset))
    n_val      = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)

    model     = LTInet(n_structure_classes=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn   = nn.CrossEntropyLoss()



n_epochs = 100

for epoch in range(1, n_epochs + 1):
    train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, device)
    val_loss,   val_acc   = evaluate(model, val_loader, loss_fn, device)
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'LTInet_best.pth')

    if epoch % 1 == 0:
        print(f'Epoch {epoch:3d} | '
              f'Train Loss {train_loss:.4f} | Train Acc {train_acc:.3f} | '
              f'Val Loss {val_loss:.4f} | Val Acc {val_acc:.3f}')

torch.save(model.state_dict(), 'LTInet_weights.pth')
print('Model saved to LTInet_weights.pth')