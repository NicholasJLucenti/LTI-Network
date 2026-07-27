import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────
N_LATENT_CLASSES   = 5
N_COUPLING_CLASSES = 3
NULLCLINE_DIM      = 49
XCORR_DIM          = 10
ALL_NEW_DIM        = 88   # 16 phase-x + 16 phase-y + 8 amp + 14 rqa
                          # + 7 crqa + 12 wavelet + 6 poincare + 9 xi_mag
COUPLING_LOSS_WEIGHT = 1.0

topology_map = {
    'LIMIT CYCLE':        0,
    'DAMPED OSCILLATION': 1,
    'STEADY STATE':       2,
    'UNDETERMINED':       3
}

DATA_ROOT   = r'D:\LTInetV5 NNdata'
WEIGHTS_DIR = r'C:/Users/nickj/MATLAB Drive'

data_dirs = [
    # Lat0 — 10 folders
    f'{DATA_ROOT}/V5Lat0_Ch0_Sys1_NNdata', f'{DATA_ROOT}/V5Lat0_Ch0_Sys2_NNdata',
    f'{DATA_ROOT}/V5Lat0_Ch0_Sys3_NNdata', f'{DATA_ROOT}/V5Lat0_Ch0_Sys4_NNdata',
    f'{DATA_ROOT}/V5Lat0_Ch1_Sys1_NNdata', f'{DATA_ROOT}/V5Lat0_Ch1_Sys2_NNdata',
    f'{DATA_ROOT}/V5Lat0_Ch1_Sys4_NNdata',
    f'{DATA_ROOT}/V5Lat0_Ch2_Sys1_NNdata', f'{DATA_ROOT}/V5Lat0_Ch2_Sys2_NNdata',
    f'{DATA_ROOT}/V5Lat0_Ch2_Sys4_NNdata',
    # Lat1 — 9 folders
    f'{DATA_ROOT}/V5Lat1_Ch0_Sys1_NNdata', f'{DATA_ROOT}/V5Lat1_Ch0_Sys2_NNdata',
    f'{DATA_ROOT}/V5Lat1_Ch0_Sys3_NNdata', f'{DATA_ROOT}/V5Lat1_Ch0_Sys4_NNdata',
    f'{DATA_ROOT}/V5Lat1_Ch1_Sys2_NNdata', f'{DATA_ROOT}/V5Lat1_Ch1_Sys4_NNdata',
    f'{DATA_ROOT}/V5Lat1_Ch2_Sys2_NNdata', f'{DATA_ROOT}/V5Lat1_Ch2_Sys3_NNdata',
    f'{DATA_ROOT}/V5Lat1_Ch2_Sys4_NNdata',
    # Lat2 — 11 folders
    f'{DATA_ROOT}/V5Lat2_Ch0_Sys1_NNdata', f'{DATA_ROOT}/V5Lat2_Ch0_Sys2_NNdata',
    f'{DATA_ROOT}/V5Lat2_Ch0_Sys3_NNdata', f'{DATA_ROOT}/V5Lat2_Ch0_Sys4_NNdata',
    f'{DATA_ROOT}/V5Lat2_Ch1_Sys1_NNdata', f'{DATA_ROOT}/V5Lat2_Ch1_Sys2_NNdata',
    f'{DATA_ROOT}/V5Lat2_Ch1_Sys3_NNdata', f'{DATA_ROOT}/V5Lat2_Ch1_Sys4_NNdata',
    f'{DATA_ROOT}/V5Lat2_Ch2_Sys1_NNdata', f'{DATA_ROOT}/V5Lat2_Ch2_Sys2_NNdata',
    f'{DATA_ROOT}/V5Lat2_Ch2_Sys4_NNdata',
    # Lat3 — 9 folders
    f'{DATA_ROOT}/V5Lat3_Ch0_Sys2_NNdata', f'{DATA_ROOT}/V5Lat3_Ch0_Sys3_NNdata',
    f'{DATA_ROOT}/V5Lat3_Ch0_Sys4_NNdata',
    f'{DATA_ROOT}/V5Lat3_Ch1_Sys1_NNdata', f'{DATA_ROOT}/V5Lat3_Ch1_Sys2_NNdata',
    f'{DATA_ROOT}/V5Lat3_Ch1_Sys4_NNdata',
    f'{DATA_ROOT}/V5Lat3_Ch2_Sys1_NNdata', f'{DATA_ROOT}/V5Lat3_Ch2_Sys2_NNdata',
    f'{DATA_ROOT}/V5Lat3_Ch2_Sys4_NNdata',
    # Lat4 — 11 folders
    f'{DATA_ROOT}/V5Lat4_Ch0_Sys2_NNdata', f'{DATA_ROOT}/V5Lat4_Ch0_Sys3_NNdata',
    f'{DATA_ROOT}/V5Lat4_Ch0_Sys4_NNdata',
    f'{DATA_ROOT}/V5Lat4_Ch1_Sys1_NNdata', f'{DATA_ROOT}/V5Lat4_Ch1_Sys2_NNdata',
    f'{DATA_ROOT}/V5Lat4_Ch1_Sys3_NNdata', f'{DATA_ROOT}/V5Lat4_Ch1_Sys4_NNdata',
    f'{DATA_ROOT}/V5Lat4_Ch2_Sys1_NNdata', f'{DATA_ROOT}/V5Lat4_Ch2_Sys2_NNdata',
    f'{DATA_ROOT}/V5Lat4_Ch2_Sys3_NNdata', f'{DATA_ROOT}/V5Lat4_Ch2_Sys4_NNdata',
]

latent_names   = ['C0 Lin-x', 'C1 miRNA', 'C2 IncoFF',
                  'C3 GK-ph', 'C4 HRep-y']
coupling_names = ['Ch0 HillRep', 'Ch1 Additive', 'Ch2 Multiplicative']


# ── Dataset ───────────────────────────────────────────────────────────────
class LTInetV5Dataset(Dataset):
    def __init__(self, data_dirs, nullcline_dim):
        self.samples = []
        skipped_no_anf   = 0
        skipped_no_xcorr = 0

        for d in data_dirs:
            if not os.path.exists(d):
                print(f'  [MISSING] {d}')
                continue
            files = sorted([os.path.join(d, f)
                             for f in os.listdir(d) if f.endswith('.mat')])
            for fp in files:
                try:
                    mat = scipy.io.loadmat(fp)

                    if 'structure_label' not in mat or \
                       'coupling_label'  not in mat:
                        continue
                    struct_lbl   = int(mat['structure_label'].squeeze())
                    coupling_lbl = int(mat['coupling_label'].squeeze())
                    if not (0 <= struct_lbl   < N_LATENT_CLASSES):   continue
                    if not (0 <= coupling_lbl < N_COUPLING_CLASSES): continue

                    if 'xcorr_features' not in mat:
                        skipped_no_xcorr += 1; continue

                    if 'all_new_features' not in mat:
                        skipped_no_anf += 1; continue

                    nc = np.clip(mat['nullcline_features'].squeeze(),
                                 -50.0, 50.0).astype(np.float32)
                    if nc.shape[0] != nullcline_dim: continue
                    if not np.all(np.isfinite(nc)): continue

                    xc = np.clip(mat['xcorr_features'].squeeze(),
                                 -50.0, 50.0).astype(np.float32)
                    if xc.shape[0] != XCORR_DIM: continue
                    if not np.all(np.isfinite(xc)): continue

                    anf = np.clip(mat['all_new_features'].squeeze(),
                                  -50.0, 50.0).astype(np.float32)
                    if anf.shape[0] != ALL_NEW_DIM: continue
                    if not np.all(np.isfinite(anf)): continue

                    xi = mat['Xi_ternary'].astype(np.float32)
                    if xi.shape[0] < 9:
                        xi = np.vstack([xi,
                            np.zeros((9-xi.shape[0], 2), dtype=np.float32)])

                    topo = topology_map.get(
                        str(mat['topology'][0]).strip(), 3)

                    self.samples.append(
                        (nc, xc, anf, xi, topo, struct_lbl, coupling_lbl))

                except Exception:
                    continue

        if skipped_no_xcorr > 0:
            print(f'  [INFO] {skipped_no_xcorr} files missing xcorr_features.')
        if skipped_no_anf > 0:
            print(f'  [INFO] {skipped_no_anf} files missing all_new_features '
                  f'— run V5_annex_full.m first.')
        print(f'Loaded {len(self.samples)} valid examples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        nc, xc, anf, xi, topo, slbl, clbl = self.samples[idx]
        return (torch.tensor(nc),
                torch.tensor(xc),
                torch.tensor(anf),
                torch.tensor(xi),
                torch.tensor(topo,  dtype=torch.long),
                torch.tensor(slbl,  dtype=torch.long),
                torch.tensor(clbl,  dtype=torch.long))


# ── Model ─────────────────────────────────────────────────────────────────
class Branch1_NullclineMLP(nn.Module):
    def __init__(self, input_dim=49, output_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 96),        nn.BatchNorm1d(96),  nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, output_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)


class Branch1b_XcorrMLP(nn.Module):
    def __init__(self, input_dim=10, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)


class Branch1c_AllNewMLP(nn.Module):
    """
    Processes the 88-dim all_new_features vector:
      [0:15]   phase-resolved x-residual (16)
      [16:31]  phase-resolved y-residual (16)
      [32:39]  instantaneous amplitude envelope (8)
      [40:53]  RQA — x(7) + y(7) (14)
      [54:60]  cross-RQA x vs y (7)
      [61:72]  wavelet energy — x(6) + y(6) (12)
      [73:78]  Poincare section statistics (6)
      [79:87]  SINDy coefficient magnitudes (9)
    """
    def __init__(self, input_dim=88, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)


class Branch2_TermAttention(nn.Module):
    def __init__(self, n_terms=9, n_equations=2, embed_dim=32, output_dim=24):
        super().__init__()
        self.asym_proj  = nn.Sequential(
            nn.Linear(n_terms, 16), nn.ReLU(), nn.Linear(16, 8))
        self.embed      = nn.Linear(n_terms, embed_dim)
        self.attention  = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(n_equations*embed_dim+8, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim), nn.ReLU())

    def forward(self, x):
        if x.shape[-1] != 2: x = x.transpose(-1, -2)
        x = x.reshape(x.shape[0], 9, 2)
        asym     = self.asym_proj(x[:,:,0] - x[:,:,1])
        tokens   = self.embed(x.permute(0, 2, 1))
        attn_out, _ = self.attention(tokens, tokens, tokens)
        flat     = attn_out.reshape(attn_out.shape[0], -1)
        return self.projection(torch.cat([flat, asym], dim=1))


class Branch3_TopologyEmbed(nn.Module):
    def __init__(self, n_classes=4, embed_dim=8, output_dim=8):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.proj  = nn.Sequential(nn.Linear(embed_dim, output_dim), nn.ReLU())
    def forward(self, x): return self.proj(self.embed(x))


class LTInetV5(nn.Module):
    """
    Dual-output V5 network with full feature set.
    merge_dim = 48 (nullcline) + 16 (xcorr) + 32 (all_new)
              + 24 (term attn) + 8 (topology) = 128
    Head A — latent:   128 → 96 → 48 → 5
    Head B — coupling: 128 → 64 → 32 → 3
    """
    def __init__(self):
        super().__init__()
        self.branch1  = Branch1_NullclineMLP(input_dim=49,  output_dim=48)
        self.branch1b = Branch1b_XcorrMLP(input_dim=10,     output_dim=16)
        self.branch1c = Branch1c_AllNewMLP(input_dim=88,    output_dim=32)
        self.branch2  = Branch2_TermAttention(n_terms=9,    output_dim=24)
        self.branch3  = Branch3_TopologyEmbed(n_classes=4,  output_dim=8)

        self.head_latent = nn.Sequential(
            nn.Linear(128, 96), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(96,  48), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, N_LATENT_CLASSES)
        )
        self.head_coupling = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,  32), nn.ReLU(),
            nn.Linear(32, N_COUPLING_CLASSES)
        )

    def forward(self, nullcline, xcorr, all_new, xi_ternary, topo_label):
        merged = torch.cat([
            self.branch1(nullcline),
            self.branch1b(xcorr),
            self.branch1c(all_new),
            self.branch2(xi_ternary),
            self.branch3(topo_label)
        ], dim=1)
        return self.head_latent(merged), self.head_coupling(merged)


# ── Train / Evaluate ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, loss_fn, device,
                nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std):
    model.train()
    total_loss = 0
    lat_correct = coup_correct = total = 0

    for nc, xc, anf, xi, topo, slbl, clbl in loader:
        nc   = (nc.to(device)  - nc_mean)  / nc_std
        xc   = (xc.to(device)  - xc_mean)  / xc_std
        anf  = (anf.to(device) - anf_mean) / anf_std
        xi   = xi.to(device)
        topo = topo.to(device)
        slbl = slbl.to(device)
        clbl = clbl.to(device)

        optimizer.zero_grad()
        logits_lat, logits_coup = model(nc, xc, anf, xi, topo)
        loss = loss_fn(logits_lat,  slbl) + \
               COUPLING_LOSS_WEIGHT * loss_fn(logits_coup, clbl)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        lat_correct += (logits_lat.argmax(dim=1)  == slbl).sum().item()
        coup_correct += (logits_coup.argmax(dim=1) == clbl).sum().item()
        total       += slbl.size(0)

    return total_loss/len(loader), lat_correct/total, coup_correct/total


def evaluate(model, loader, loss_fn, device,
             nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std):
    model.eval()
    total_loss = 0
    lat_correct = coup_correct = total = 0

    with torch.no_grad():
        for nc, xc, anf, xi, topo, slbl, clbl in loader:
            nc   = (nc.to(device)  - nc_mean)  / nc_std
            xc   = (xc.to(device)  - xc_mean)  / xc_std
            anf  = (anf.to(device) - anf_mean) / anf_std
            xi   = xi.to(device)
            topo = topo.to(device)
            slbl = slbl.to(device)
            clbl = clbl.to(device)

            logits_lat, logits_coup = model(nc, xc, anf, xi, topo)
            loss = loss_fn(logits_lat,  slbl) + \
                   COUPLING_LOSS_WEIGHT * loss_fn(logits_coup, clbl)

            total_loss  += loss.item()
            lat_correct += (logits_lat.argmax(dim=1)  == slbl).sum().item()
            coup_correct += (logits_coup.argmax(dim=1) == clbl).sum().item()
            total       += slbl.size(0)

    return total_loss/len(loader), lat_correct/total, coup_correct/total


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    dataset = LTInetV5Dataset(data_dirs, NULLCLINE_DIM)

    if len(dataset) == 0:
        raise RuntimeError('No valid examples found. '
                           'Check data_dirs and that V5_annex_full.m '
                           'has been run on all folders.')

    struct_counts   = Counter(s[5] for s in dataset.samples)
    coupling_counts = Counter(s[6] for s in dataset.samples)

    print(f'\nTotal examples: {len(dataset)}')
    print('Latent distribution:')
    for i in range(N_LATENT_CLASSES):
        print(f'  {latent_names[i]}: {struct_counts.get(i, 0)}')
    print('Coupling distribution:')
    for i in range(N_COUPLING_CLASSES):
        print(f'  {coupling_names[i]}: {coupling_counts.get(i, 0)}')

    # Per-feature normalisation for all three continuous feature blocks
    all_nc  = np.stack([s[0] for s in dataset.samples], axis=0)
    all_xc  = np.stack([s[1] for s in dataset.samples], axis=0)
    all_anf = np.stack([s[2] for s in dataset.samples], axis=0)

    def make_norm(arr):
        mean = torch.tensor(arr.mean(axis=0), dtype=torch.float32)
        std  = torch.tensor(arr.std(axis=0),  dtype=torch.float32)
        std[std < 1e-6] = 1.0
        return mean, std

    nc_mean,  nc_std  = make_norm(all_nc)
    xc_mean,  xc_std  = make_norm(all_xc)
    anf_mean, anf_std = make_norm(all_anf)

    print(f'\nNullcline norm:   mean [{nc_mean.min():.3f}, {nc_mean.max():.3f}]')
    print(f'Xcorr norm:       mean [{xc_mean.min():.3f}, {xc_mean.max():.3f}]')
    print(f'All-new norm:     mean [{anf_mean.min():.3f}, {anf_mean.max():.3f}]')

    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  drop_last=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    print(f'Train: {n_train}  Val: {n_val}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    nc_mean  = nc_mean.to(device);  nc_std  = nc_std.to(device)
    xc_mean  = xc_mean.to(device);  xc_std  = xc_std.to(device)
    anf_mean = anf_mean.to(device); anf_std = anf_std.to(device)

    model     = LTInetV5().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.05)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params:,}')

    SAVE_PATH    = f'{WEIGHTS_DIR}/LTInetV5_full_best.pth'
    best_avg_acc = 0.0
    n_epochs     = 100
    es_patience  = 60
    no_improve   = 0

    train_losses,    val_losses    = [], []
    train_lat_accs,  val_lat_accs  = [], []
    train_coup_accs, val_coup_accs = [], []

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_lat, tr_coup = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std)
        va_loss, va_lat, va_coup = evaluate(
            model, val_loader, loss_fn, device,
            nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std)
        scheduler.step(va_loss)

        va_avg = (va_lat + va_coup) / 2.0

        train_losses.append(tr_loss);     val_losses.append(va_loss)
        train_lat_accs.append(tr_lat);    val_lat_accs.append(va_lat)
        train_coup_accs.append(tr_coup);  val_coup_accs.append(va_coup)

        if va_avg > best_avg_acc:
            best_avg_acc = va_avg
            no_improve   = 0
            torch.save(model.state_dict(), SAVE_PATH)
            np.save(f'{WEIGHTS_DIR}/V5_nc_mean.npy',  nc_mean.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V5_nc_std.npy',   nc_std.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V5_xc_mean.npy',  xc_mean.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V5_xc_std.npy',   xc_std.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V5_anf_mean.npy', anf_mean.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V5_anf_std.npy',  anf_std.cpu().numpy())
        else:
            no_improve += 1

        print(f'Epoch {epoch:3d} | Loss {tr_loss:.4f}/{va_loss:.4f} | '
              f'Lat {tr_lat:.3f}/{va_lat:.3f} | '
              f'Coup {tr_coup:.3f}/{va_coup:.3f} | '
              f'Avg {va_avg:.3f} | LR {optimizer.param_groups[0]["lr"]:.1e}')

        if epoch % 20 == 0:
            ep = list(range(1, epoch+1))
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(ep, train_losses,    'b-',  label='Train')
            axes[0].plot(ep, val_losses,      'r--', label='Val')
            axes[0].set_title('Combined loss')
            axes[0].legend(); axes[0].grid(alpha=0.3)
            axes[1].plot(ep, train_lat_accs,  'b-',  label='Train')
            axes[1].plot(ep, val_lat_accs,    'r--', label='Val')
            axes[1].axhline(1/N_LATENT_CLASSES, color='gray', linestyle=':',
                            label=f'Chance ({100/N_LATENT_CLASSES:.0f}%)')
            axes[1].set_ylim([0, 1]); axes[1].set_title('Latent accuracy')
            axes[1].legend(); axes[1].grid(alpha=0.3)
            axes[2].plot(ep, train_coup_accs, 'b-',  label='Train')
            axes[2].plot(ep, val_coup_accs,   'r--', label='Val')
            axes[2].axhline(1/N_COUPLING_CLASSES, color='gray', linestyle=':',
                            label=f'Chance ({100/N_COUPLING_CLASSES:.0f}%)')
            axes[2].set_ylim([0, 1]); axes[2].set_title('Coupling accuracy')
            axes[2].legend(); axes[2].grid(alpha=0.3)
            fig.suptitle(f'LTInetV5 full | Epoch {epoch} | '
                         f'Lat {va_lat:.3f} | Coup {va_coup:.3f} | '
                         f'Best avg {best_avg_acc:.3f}')
            plt.tight_layout(); plt.show()

        if no_improve >= es_patience:
            print(f'\nEarly stopping at epoch {epoch}')
            break

    print(f'\nTraining complete. Best avg val acc: {best_avg_acc:.3f}')
    print(f'Weights: {SAVE_PATH}')