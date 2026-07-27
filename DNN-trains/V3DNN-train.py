import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────
ACTIVE_CLASSES = [0, 1, 8, 10, 11]
LABEL_REMAP    = {orig: new for new, orig in enumerate(ACTIVE_CLASSES)}

N_CLASSES     = 5
NULLCLINE_DIM = 49
XCORR_DIM     = 10   # 5 features for resid vs x, 5 for resid vs y

topology_map = {
    'LIMIT CYCLE':        0,
    'DAMPED OSCILLATION': 1,
    'STEADY STATE':       2,
    'UNDETERMINED':       3
}

DATA_ROOT   = r'D:\LTInetV3 NNdata'

data_dirs = [
    f'{DATA_ROOT}/V3Lat0_Sys1_NNdata',  f'{DATA_ROOT}/V3Lat0_Sys2_NNdata',
    f'{DATA_ROOT}/V3Lat0_Sys3_NNdata',  f'{DATA_ROOT}/V3Lat0_Sys4_NNdata',
    f'{DATA_ROOT}/V3Lat1_Sys1_NNdata',  f'{DATA_ROOT}/V3Lat1_Sys2_NNdata',
    f'{DATA_ROOT}/V3Lat1_Sys3_NNdata',  f'{DATA_ROOT}/V3Lat1_Sys4_NNdata',
    f'{DATA_ROOT}/V3Lat8_Sys1_NNdata',  f'{DATA_ROOT}/V3Lat8_Sys2_NNdata',
    f'{DATA_ROOT}/V3Lat8_Sys3_NNdata',  f'{DATA_ROOT}/V3Lat8_Sys4_NNdata',
    f'{DATA_ROOT}/V3Lat10_Sys1_NNdata', f'{DATA_ROOT}/V3Lat10_Sys2_NNdata',
    f'{DATA_ROOT}/V3Lat10_Sys3_NNdata', f'{DATA_ROOT}/V3Lat10_Sys4_NNdata',
    f'{DATA_ROOT}/V3Lat11_Sys1_NNdata', f'{DATA_ROOT}/V3Lat11_Sys2_NNdata',
    f'{DATA_ROOT}/V3Lat11_Sys3_NNdata', f'{DATA_ROOT}/V3Lat11_Sys4_NNdata',
    
    f'{DATA_ROOT}/V3Lat0_Sys5_NNdata',  f'{DATA_ROOT}/V3Lat1_Sys5_NNdata',
    f'{DATA_ROOT}/V3Lat8_Sys5_NNdata',  f'{DATA_ROOT}/V3Lat10_Sys5_NNdata',
    f'{DATA_ROOT}/V3Lat11_Sys5_NNdata',  f'{DATA_ROOT}/V3Lat12_Sys5_NNdata',


]


# ── Dataset ───────────────────────────────────────────────────────────────
class LTInetV3Dataset(Dataset):
    def __init__(self, data_dirs, label_remap, nullcline_dim):
        self.label_remap   = label_remap
        self.nullcline_dim = nullcline_dim
        self.samples       = []
        skipped_no_xcorr   = 0

        for d in data_dirs:
            if not os.path.exists(d):
                print(f'  [MISSING] {d}')
                continue
            files = sorted([os.path.join(d, f)
                             for f in os.listdir(d) if f.endswith('.mat')])
            for fp in files:
                try:
                    mat = scipy.io.loadmat(fp)
                    lbl = int(mat['structure_label'].squeeze())

                    if lbl not in label_remap:
                        continue

                    # Skip files that haven't had xcorr appended yet
                    if 'xcorr_features' not in mat:
                        skipped_no_xcorr += 1
                        continue

                    nc_raw = mat['nullcline_features'].squeeze()
                    if nc_raw.shape[0] != nullcline_dim:
                        continue

                    nc_raw = np.clip(nc_raw, -50.0, 50.0)
                    nc     = nc_raw.astype(np.float32)
                    if not np.all(np.isfinite(nc)):
                        continue

                    xc_raw = mat['xcorr_features'].squeeze()
                    xc_raw = np.clip(xc_raw, -50.0, 50.0)
                    xc     = xc_raw.astype(np.float32)
                    if xc.shape[0] != XCORR_DIM or not np.all(np.isfinite(xc)):
                        continue

                    xi = mat['Xi_ternary'].astype(np.float32)
                    if xi.shape[0] < 9:
                        xi = np.vstack([xi,
                            np.zeros((9-xi.shape[0], 2), dtype=np.float32)])

                    topo         = topology_map.get(
                        str(mat['topology'][0]).strip(), 3)
                    remapped_lbl = label_remap[lbl]

                    self.samples.append((nc, xc, xi, topo, remapped_lbl))

                except Exception:
                    continue

        if skipped_no_xcorr > 0:
            print(f'  [INFO] {skipped_no_xcorr} files skipped — '
                  f'xcorr_features not yet appended. '
                  f'Run V3_append_xcorr_features.m first.')
        print(f'Loaded {len(self.samples)} valid examples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        nc, xc, xi, topo, lbl = self.samples[idx]
        return (
            torch.tensor(nc),
            torch.tensor(xc),
            torch.tensor(xi),
            torch.tensor(topo, dtype=torch.long),
            torch.tensor(lbl,  dtype=torch.long)
        )


# ── Model ─────────────────────────────────────────────────────────────────

class Branch1_NullclineMLP(nn.Module):
    """
    MLP on the 49-dim nullcline feature vector.
    """
    def __init__(self, input_dim=49, output_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 96),
            nn.BatchNorm1d(96), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Branch1b_XcorrMLP(nn.Module):
    """
    Small MLP on the 10-dim cross-correlation feature vector.
    Kept deliberately small — 10 features do not justify a deep network,
    and keeping it shallow prevents it from overfitting the xcorr signal.
    Output concatenated with Branch1 nullcline output before the head.
    """
    def __init__(self, input_dim=10, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Branch2_TermAttention(nn.Module):
    def __init__(self, n_terms=9, n_equations=2, embed_dim=32, output_dim=24):
        super().__init__()
        self.asym_proj = nn.Sequential(
            nn.Linear(n_terms, 16), nn.ReLU(),
            nn.Linear(16, 8))
        self.embed     = nn.Linear(n_terms, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(n_equations * embed_dim + 8, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim), nn.ReLU())

    def forward(self, x):
        if x.shape[-1] != 2:
            x = x.transpose(-1, -2)
        x = x.reshape(x.shape[0], 9, 2)
        asym     = self.asym_proj(x[:, :, 0] - x[:, :, 1])
        tokens   = self.embed(x.permute(0, 2, 1))
        attn_out, _ = self.attention(tokens, tokens, tokens)
        flat     = attn_out.reshape(attn_out.shape[0], -1)
        return self.projection(torch.cat([flat, asym], dim=1))


class Branch3_TopologyEmbed(nn.Module):
    def __init__(self, n_classes=4, embed_dim=8, output_dim=8):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.proj  = nn.Sequential(nn.Linear(embed_dim, output_dim), nn.ReLU())

    def forward(self, x):
        return self.proj(self.embed(x))


class LTInetV3(nn.Module):
    """
    V3 7-class x-driven model with nullcline + xcorr inputs.
    merge_dim = 48 (B1 nullcline) + 16 (B1b xcorr) + 24 (B2) + 8 (B3) = 96
    """
    def __init__(self, n_structure_classes=7):
        super().__init__()
        self.branch1    = Branch1_NullclineMLP(input_dim=49,  output_dim=48)
        self.branch1b   = Branch1b_XcorrMLP(input_dim=10,    output_dim=16)
        self.branch2    = Branch2_TermAttention(n_terms=9,    output_dim=24)
        self.branch3    = Branch3_TopologyEmbed(n_classes=4,  output_dim=8)
        self.head = nn.Sequential(
            nn.Linear(96, 80), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(80, 48), nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(48, n_structure_classes)
        )

    def forward(self, nullcline, xcorr, xi_ternary, topo_label):
        b1  = self.branch1(nullcline)
        b1b = self.branch1b(xcorr)
        b2  = self.branch2(xi_ternary)
        b3  = self.branch3(topo_label)
        return self.head(torch.cat([b1, b1b, b2, b3], dim=1))


# ── Train / Evaluate ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, loss_fn, device,
                nc_mean, nc_std, xc_mean, xc_std):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for nc, xc, xi, topo, label in loader:
        nc, xc, xi, topo, label = (nc.to(device), xc.to(device),
                                    xi.to(device), topo.to(device),
                                    label.to(device))
        nc = (nc - nc_mean) / nc_std
        xc = (xc - xc_mean) / xc_std
        optimizer.zero_grad()
        logits = model(nc, xc, xi, topo)
        loss   = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(dim=1) == label).sum().item()
        total      += label.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, loss_fn, device,
             nc_mean, nc_std, xc_mean, xc_std):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for nc, xc, xi, topo, label in loader:
            nc, xc, xi, topo, label = (nc.to(device), xc.to(device),
                                        xi.to(device), topo.to(device),
                                        label.to(device))
            nc = (nc - nc_mean) / nc_std
            xc = (xc - xc_mean) / xc_std
            logits = model(nc, xc, xi, topo)
            loss   = loss_fn(logits, label)
            total_loss += loss.item()
            correct    += (logits.argmax(dim=1) == label).sum().item()
            total      += label.size(0)
    return total_loss / len(loader), correct / total


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    dataset = LTInetV3Dataset(data_dirs, LABEL_REMAP, NULLCLINE_DIM)

    if len(dataset) == 0:
        raise RuntimeError('No valid examples found. Check data_dirs and '
                           'that xcorr_features has been appended via '
                           'V3_append_xcorr_features.m')

    labels      = [s[4] for s in dataset.samples]
    counts      = Counter(labels)
    class_names = ['C0 Lin-x', 'C1 Hact-x', 'C2 MM-deg',
                   'C3 Quad', 'C4 IncoFF']

    print(f'\nTotal examples: {len(dataset)}')
    for i in range(N_CLASSES):
        orig = ACTIVE_CLASSES[i]
        print(f'  Class {i} (Lat{orig} {class_names[i]}): {counts.get(i, 0)}')

    # Per-feature normalisation for nullcline and xcorr independently
    all_nc  = np.stack([s[0] for s in dataset.samples], axis=0)
    all_xc  = np.stack([s[1] for s in dataset.samples], axis=0)

    nc_mean = torch.tensor(all_nc.mean(axis=0), dtype=torch.float32)
    nc_std  = torch.tensor(all_nc.std(axis=0),  dtype=torch.float32)
    nc_std[nc_std < 1e-6] = 1.0

    xc_mean = torch.tensor(all_xc.mean(axis=0), dtype=torch.float32)
    xc_std  = torch.tensor(all_xc.std(axis=0),  dtype=torch.float32)
    xc_std[xc_std < 1e-6] = 1.0

    print(f'\nNullcline norm:  mean [{nc_mean.min():.3f}, {nc_mean.max():.3f}] '
          f'std [{nc_std.min():.3f}, {nc_std.max():.3f}]')
    print(f'Xcorr norm:      mean [{xc_mean.min():.3f}, {xc_mean.max():.3f}] '
          f'std [{xc_std.min():.3f}, {xc_std.max():.3f}]')

    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=16,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=16,
                              shuffle=False, drop_last=False)

    print(f'Train: {n_train}  Val: {n_val}')

    device  = torch.device('cpu')
    nc_mean = nc_mean.to(device)
    nc_std  = nc_std.to(device)
    xc_mean = xc_mean.to(device)
    xc_std  = xc_std.to(device)

    model     = LTInetV3(n_structure_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.05)

    SAVE_PATH    = r'C:/Users/nickj/MATLAB Drive/LTInetV3_7class_xcorr_best.pth'
    best_val_acc = 0.0
    n_epochs     = 100
    es_patience  = 40
    no_improve   = 0

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            nc_mean, nc_std, xc_mean, xc_std)
        val_loss, val_acc = evaluate(
            model, val_loader, loss_fn, device,
            nc_mean, nc_std, xc_mean, xc_std)
        scheduler.step(val_loss)

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc);    val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), SAVE_PATH)
            # Save normalisation stats alongside weights for inference
            import numpy as _np
            _np.save(r'C:/Users/nickj/MATLAB Drive/nc_mean.npy',
                     nc_mean.cpu().numpy())
            _np.save(r'C:/Users/nickj/MATLAB Drive/nc_std.npy',
                     nc_std.cpu().numpy())
            _np.save(r'C:/Users/nickj/MATLAB Drive/xc_mean.npy',
                     xc_mean.cpu().numpy())
            _np.save(r'C:/Users/nickj/MATLAB Drive/xc_std.npy',
                     xc_std.cpu().numpy())
        else:
            no_improve += 1

        print(f'Epoch {epoch:3d} | '
              f'Train Loss {train_loss:.4f} | Train Acc {train_acc:.3f} | '
              f'Val Loss {val_loss:.4f} | Val Acc {val_acc:.3f} | '
              f'LR {optimizer.param_groups[0]["lr"]:.1e}')

        if epoch % 20 == 0:
            ep = list(range(1, epoch+1))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(ep, train_losses, 'b-', label='Train')
            ax1.plot(ep, val_losses,   'r--', label='Val')
            ax1.set_title('Loss'); ax1.legend(); ax1.grid(alpha=0.3)
            ax2.plot(ep, train_accs, 'b-', label='Train')
            ax2.plot(ep, val_accs,   'r--', label='Val')
            ax2.axhline(1/N_CLASSES, color='gray', linestyle=':',
                        label=f'Chance ({100/N_CLASSES:.1f}%)')
            ax2.set_ylim([0, 1]); ax2.set_title('Accuracy')
            ax2.legend(); ax2.grid(alpha=0.3)
            fig.suptitle(f'LTInetV3 7-class xcorr | '
                         f'Epoch {epoch} | Val {val_acc:.3f} | '
                         f'Best {best_val_acc:.3f}')
            plt.tight_layout(); plt.show()

        if no_improve >= es_patience:
            print(f'\nEarly stopping at epoch {epoch} — '
                  f'no improvement for {es_patience} epochs')
            break

    print(f'\nTraining complete. Best Val Acc: {best_val_acc:.3f}')
    print(f'Weights saved to: {SAVE_PATH}')