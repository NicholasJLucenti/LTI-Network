import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
epoch_list               = []

def update_plots(epoch, train_loss, val_loss, train_acc, val_acc):
    epoch_list.append(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epoch_list, train_losses, 'b-',  label='Train Loss')
    ax1.plot(epoch_list, val_losses,   'r--', label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss');   ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epoch_list, train_accs, 'b-',  label='Train Acc')
    ax2.plot(epoch_list, val_accs,   'r--', label='Val Acc')
    ax2.axhline(0.125, color='gray', linestyle=':', linewidth=0.8,
                label='Chance (12.5%)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    fig.suptitle(f'LTInet V2 — Epoch {epoch} | '
                 f'Val Acc {val_acc:.3f} | '
                 f'Best {best_val_acc:.3f}', fontsize=11)

    plt.tight_layout()
    display(fig)
    plt.close(fig)


topology_map = {
    'LIMIT CYCLE':        0,
    'DAMPED OSCILLATION': 1,
    'STEADY STATE':       2,
    'UNDETERMINED':       3
}


class LTInetV2Dataset(Dataset):
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
        while True:
            try:
                data = scipy.io.loadmat(self.files[idx])
                break
            except Exception:
                idx = (idx + 1) % len(self.files)

        fft_raw = data['fft_features'].astype(np.float32)

        rx = data['resid_dx'].squeeze()
        ry = data['resid_dy'].squeeze()

        rx = np.clip(rx, -1e6, 1e6).astype(np.float32)
        ry = np.clip(ry, -1e6, 1e6).astype(np.float32)

        rx = np.nan_to_num(rx, nan=0.0, posinf=0.0, neginf=0.0)
        ry = np.nan_to_num(ry, nan=0.0, posinf=0.0, neginf=0.0)

        rx = (rx - rx.mean()) / (rx.std() + 1e-8)
        ry = (ry - ry.mean()) / (ry.std() + 1e-8)

        target_length = 501
        if len(rx) < target_length:
            pad = target_length - len(rx)
            rx  = np.concatenate([rx, np.zeros(pad)])
            ry  = np.concatenate([ry, np.zeros(pad)])

        residual_raw = np.stack([rx, ry], axis=0)

        xi_raw = data['Xi_ternary'].astype(np.float32)
        if xi_raw.shape[0] < 9:
            pad    = np.zeros((9 - xi_raw.shape[0], 2), dtype=np.float32)
            xi_raw = np.vstack([xi_raw, pad])

        topo_str   = str(data['topology'][0]).strip()
        topo_label = topology_map.get(topo_str, 3)

        struct_label = int(data['structure_label'].squeeze())

        return (
            torch.tensor(fft_raw),
            torch.tensor(residual_raw),
            torch.tensor(xi_raw),
            torch.tensor(topo_label,   dtype=torch.long),
            torch.tensor(struct_label, dtype=torch.long)
        )


class Branch1_Hybrid(nn.Module):
    def __init__(self, n_filters=32, output_dim=64):
        super().__init__()

        self.fft_stream_small = nn.Sequential(
            nn.Conv1d(3, n_filters, kernel_size=3,  padding=1),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_filters), nn.ReLU()
        )
        self.fft_stream_medium = nn.Sequential(
            nn.Conv1d(3, n_filters, kernel_size=7,  padding=3),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(n_filters), nn.ReLU()
        )
        self.fft_stream_large = nn.Sequential(
            nn.Conv1d(3, n_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(n_filters), nn.ReLU()
        )

        self.raw_stream_small = nn.Sequential(
            nn.Conv1d(2, n_filters, kernel_size=5,  padding=2),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_filters), nn.ReLU()
        )
        self.raw_stream_medium = nn.Sequential(
            nn.Conv1d(2, n_filters, kernel_size=21, padding=10),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=21, padding=10),
            nn.BatchNorm1d(n_filters), nn.ReLU()
        )
        self.raw_stream_large = nn.Sequential(
            nn.Conv1d(2, n_filters, kernel_size=51, padding=25),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=51, padding=25),
            nn.BatchNorm1d(n_filters), nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.projection = nn.Sequential(
            nn.Linear(6 * n_filters, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )

    def forward(self, fft, raw):
        fs = self.gap(self.fft_stream_small(fft)).squeeze(-1)
        fm = self.gap(self.fft_stream_medium(fft)).squeeze(-1)
        fl = self.gap(self.fft_stream_large(fft)).squeeze(-1)

        rs = self.gap(self.raw_stream_small(raw)).squeeze(-1)
        rm = self.gap(self.raw_stream_medium(raw)).squeeze(-1)
        rl = self.gap(self.raw_stream_large(raw)).squeeze(-1)

        combined = torch.cat([fs, fm, fl, rs, rm, rl], dim=1)
        return self.projection(combined)


class Branch2_TermAttention(nn.Module):
    def __init__(self, n_terms=9, n_equations=2, embed_dim=16, output_dim=32):
        super().__init__()
        self.embed     = nn.Linear(n_terms, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=2, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(n_equations * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x        = x.permute(0, 2, 1)
        tokens   = self.embed(x)
        attn_out, _ = self.attention(tokens, tokens, tokens)
        flat     = attn_out.reshape(attn_out.shape[0], -1)
        return self.projection(flat)


class Branch3_TopologyEmbed(nn.Module):
    def __init__(self, n_classes=4, embed_dim=16, output_dim=16):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.proj  = nn.Sequential(
            nn.Linear(embed_dim, output_dim), nn.ReLU())

    def forward(self, x):
        return self.proj(self.embed(x))


class LTInetV2(nn.Module):
    def __init__(self, n_structure_classes=4):
        super().__init__()
        self.branch1 = Branch1_Hybrid(n_filters=32, output_dim=64)
        self.branch2 = Branch2_TermAttention(n_terms=9, output_dim=32)
        self.branch3 = Branch3_TopologyEmbed(n_classes=4, output_dim=16)

        merge_dim = 64 + 32 + 16

        self.head = nn.Sequential(
            nn.Linear(merge_dim, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_structure_classes)
        )

    def forward(self, fft_features, residual, xi_ternary, topo_label):
        b1 = self.branch1(fft_features, residual)
        b2 = self.branch2(xi_ternary)
        b3 = self.branch3(topo_label)
        return self.head(torch.cat([b1, b2, b3], dim=1))


def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0
    for fft, raw, xi, topo, label in loader:
        fft   = fft.to(device)
        raw   = raw.to(device)
        xi    = xi.to(device)
        topo  = topo.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        logits = model(fft, raw, xi, topo)
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
        for fft, raw, xi, topo, label in loader:
            fft   = fft.to(device)
            raw   = raw.to(device)
            xi    = xi.to(device)
            topo  = topo.to(device)
            label = label.to(device)
            logits = model(fft, raw, xi, topo)
            loss   = loss_fn(logits, label)
            total_loss += loss.item()
            correct    += (logits.argmax(dim=1) == label).sum().item()
            total      += label.size(0)
    return total_loss / len(loader), correct / total


if __name__ == '__main__':

    
    # ── DATA DIRECTORIES ─────────────────────────────────────────────────
    # Sys1–Sys4: original systems (Goodwin, Brusselator, Repressilator, VdP)
    # Sys5–Sys6: new systems (FitzHugh-Nagumo, Lotka-Volterra)
    # Folders use 1-indexed Lat labels; structure_label inside .mat is 0-indexed
    DATA_ROOT = r'D:\LTInetV2 NNdata'
    data_dirs = [
    # ── Lat0-3: original latent classes ──────────────────────────────────
    # ── Lat0-3 × Sys1-6 (Goodwin, Brusselator, Repressilator,
    #                      Van der Pol, FitzHugh-Nagumo, Lotka-Volterra)
    os.path.join(DATA_ROOT, 'V2Lat0_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat0_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat0_Sys3_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat0_Sys4_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat0_Sys5_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat0_Sys6_NNdata'),

    os.path.join(DATA_ROOT, 'V2Lat1_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat1_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat1_Sys3_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat1_Sys4_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat1_Sys5_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat1_Sys6_NNdata'),

    os.path.join(DATA_ROOT, 'V2Lat2_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat2_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat2_Sys3_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat2_Sys4_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat2_Sys5_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat2_Sys6_NNdata'),

    os.path.join(DATA_ROOT, 'V2Lat3_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat3_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat3_Sys3_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat3_Sys4_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat3_Sys5_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat3_Sys6_NNdata'),

    # V2Lat4_Sys1 skipped — Goodwin generation too slow
    os.path.join(DATA_ROOT, 'V2Lat4_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat4_Sys3_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat4_Sys4_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat4_Sys5_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat4_Sys6_NNdata'),

    os.path.join(DATA_ROOT, 'V2Lat5_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat5_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat5_Sys3_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat5_Sys4_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat5_Sys5_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat5_Sys6_NNdata'),

    os.path.join(DATA_ROOT, 'V2Lat6_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat6_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat6_Sys3_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat6_Sys4_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat6_Sys5_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat6_Sys6_NNdata'),

    os.path.join(DATA_ROOT, 'V2Lat7_Sys1_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat7_Sys2_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat7_Sys3_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat7_Sys4_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat7_Sys5_NNdata'),
    os.path.join(DATA_ROOT, 'V2Lat7_Sys6_NNdata'),
]

    device = torch.device('cpu')
    torch.set_num_threads(8)

    dataset = LTInetV2Dataset(data_dirs)

    # Class balance check — run before training to verify all 4 classes
    # are populated and roughly equal across the 6 systems
    labels = []
    for f in dataset.files:
        try:
            d = scipy.io.loadmat(f, variable_names=['structure_label'])
            labels.append(int(d['structure_label'].squeeze()))
        except Exception:
            labels.append(-1)

    print(f'Total examples: {len(dataset)}')
    for i in range(8):
        print(f'  Class {i}: {labels.count(i)}')
    print(f'  Unreadable: {labels.count(-1)}')

    n_train  = int(0.8 * len(dataset))
    n_val    = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)

    model     = LTInetV2(n_structure_classes=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=30, gamma=0.5)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    n_epochs     = 150

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train(model, train_loader,
                                       optimizer, loss_fn, device)
        val_loss,   val_acc   = evaluate(model, val_loader,
                                          loss_fn, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'LTInetV2_best.pth')

        if epoch % 5 == 0:
            update_plots(epoch, train_loss, val_loss, train_acc, val_acc)

        if epoch % 1 == 0:
            print(f'Epoch {epoch:3d} | '
                  f'Train Loss {train_loss:.4f} | '
                  f'Train Acc {train_acc:.3f} | '
                  f'Val Loss {val_loss:.4f} | '
                  f'Val Acc {val_acc:.3f}')

    torch.save(model.state_dict(), 'LTInetV2_weights.pth')
    print(f'Training complete. Best Val Acc: {best_val_acc:.3f}')
    print('Model saved to LTInetV2_weights.pth')