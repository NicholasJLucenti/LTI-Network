import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────
N_LATENT_CLASSES   = 6   # Lat0-4 + Lat5 null
N_COUPLING_CLASSES = 4   # Ch0-2  + Ch3 null
NULLCLINE_DIM      = 49
XCORR_DIM          = 10
ALL_NEW_DIM        = 88   # legacy V5 block (phase/RQA/wavelet/Poincare/Xi-mag)
V6_NEW_DIM         = 12   # FNN(4) + TransferEntropy(4) + ResidualSpectrum(4)
FP_MULTI_DIM       = 10   # multi-fixed-point geometry (nonzero only for toggle switch)
BRANCH1C_DIM       = ALL_NEW_DIM + V6_NEW_DIM + FP_MULTI_DIM  # 110
COUPLING_LOSS_WEIGHT = 1.0

topology_map = {
    'LIMIT CYCLE': 0, 'DAMPED OSCILLATION': 1,
    'STEADY STATE': 2, 'UNDETERMINED': 3
}

DATA_ROOT   = r'D:\LTInetV6 NNdata'
WEIGHTS_DIR = DATA_ROOT

# All 7 systems x 5 real latents x 3 couplings + null, matching V6_Gen.m folder naming
SYSTEMS  = ['goodwin', 'brusselator', 'repressilator', 'van_der_pol',
            'fitzhugh_nagumo', 'rosenzweig_macarthur', 'toggle_switch']
N_SYS    = len(SYSTEMS)

data_dirs = []
sys_idx_lookup = {}  # folder -> system index, saved as metadata only (not a model input)
for sys_i in range(1, N_SYS + 1):
    for lat in range(5):       # Lat0-4
        for ch in range(3):    # Ch0-2
            d = f'{DATA_ROOT}/V6Lat{lat}_Ch{ch}_Sys{sys_i}_NNdata'
            data_dirs.append(d)
            sys_idx_lookup[d] = sys_i - 1
    # Null class per system
    d_null = f'{DATA_ROOT}/V6Lat5_Ch3_Sys{sys_i}_NNdata'
    data_dirs.append(d_null)
    sys_idx_lookup[d_null] = sys_i - 1

latent_names   = ['C0 Lin-x', 'C1 miRNA', 'C2 IncoFF',
                  'C3 GK-ph', 'C4 HRep-y', 'C5 Null']
coupling_names = ['Ch0 HillRep', 'Ch1 Additive', 'Ch2 Multiplicative', 'Ch3 None']


# ── Dataset ───────────────────────────────────────────────────────────────
class LTInetV6Dataset(Dataset):
    def __init__(self, data_dirs, sys_idx_lookup, nullcline_dim):
        self.samples = []
        skipped_no_anf   = 0
        skipped_no_xcorr = 0
        skipped_no_v6new = 0

        for d in data_dirs:
            if not os.path.exists(d):
                print(f'  [MISSING] {d}')
                continue
            files = sorted([os.path.join(d, f)
                             for f in os.listdir(d) if f.endswith('.mat')])
            sys_id = sys_idx_lookup[d]

            for fp in files:
                try:
                    mat = scipy.io.loadmat(fp)
                    if 'structure_label' not in mat or 'coupling_label' not in mat:
                        continue
                    sl = int(mat['structure_label'].squeeze())
                    cl = int(mat['coupling_label'].squeeze())
                    if not (0 <= sl < N_LATENT_CLASSES):   continue
                    if not (0 <= cl < N_COUPLING_CLASSES): continue

                    if 'xcorr_features' not in mat:
                        skipped_no_xcorr += 1; continue
                    if 'all_new_features' not in mat:
                        skipped_no_anf += 1; continue
                    if 'v6_new_features' not in mat:
                        skipped_no_v6new += 1; continue

                    nc = np.clip(mat['nullcline_features'].squeeze(),
                                 -50., 50.).astype(np.float32)
                    if nc.shape[0] != nullcline_dim or not np.all(np.isfinite(nc)):
                        continue

                    xc = np.clip(mat['xcorr_features'].squeeze(),
                                 -50., 50.).astype(np.float32)
                    if xc.shape[0] != XCORR_DIM or not np.all(np.isfinite(xc)):
                        continue

                    anf = np.clip(mat['all_new_features'].squeeze(),
                                  -50., 50.).astype(np.float32)
                    if anf.shape[0] != ALL_NEW_DIM or not np.all(np.isfinite(anf)):
                        continue

                    v6n = np.clip(mat['v6_new_features'].squeeze(),
                                  -50., 50.).astype(np.float32)
                    if v6n.shape[0] != V6_NEW_DIM or not np.all(np.isfinite(v6n)):
                        continue

                    # fp_multi_features may be absent on older files or all-zero
                    # for non-toggle systems — treat missing as zeros rather
                    # than skipping, since it's a legitimate "not applicable" case.
                    if 'fp_multi_features' in mat:
                        fpm = np.clip(mat['fp_multi_features'].squeeze(),
                                     -50., 50.).astype(np.float32)
                        if fpm.shape[0] != FP_MULTI_DIM or not np.all(np.isfinite(fpm)):
                            fpm = np.zeros(FP_MULTI_DIM, dtype=np.float32)
                    else:
                        fpm = np.zeros(FP_MULTI_DIM, dtype=np.float32)

                    # Concatenate into the single enlarged Branch1c input
                    anf_full = np.concatenate([anf, v6n, fpm])  # 88+12+10=110

                    xi = mat['Xi_ternary'].astype(np.float32)
                    if xi.shape[0] < 9:
                        xi = np.vstack([xi,
                            np.zeros((9-xi.shape[0], 2), dtype=np.float32)])
                    elif xi.shape[0] > 9:
                        xi = xi[:9, :]

                    topo = topology_map.get(str(mat['topology'][0]).strip(), 3)

                    system_name = str(mat['system_name'][0]).strip() if 'system_name' in mat else SYSTEMS[sys_id]

                    self.samples.append((nc, xc, anf_full, xi, topo, sl, cl, sys_id, system_name))

                except Exception:
                    continue

        if skipped_no_xcorr > 0:
            print(f'  [INFO] {skipped_no_xcorr} files missing xcorr_features.')
        if skipped_no_anf > 0:
            print(f'  [INFO] {skipped_no_anf} files missing all_new_features '
                  f'— run V6_annex_full.m first.')
        if skipped_no_v6new > 0:
            print(f'  [INFO] {skipped_no_v6new} files missing v6_new_features '
                  f'— these were generated before V6_Gen.m added FNN/TE/ResidSpec, '
                  f'or generation failed to compute them.')
        print(f'Loaded {len(self.samples)} valid examples')

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        nc, xc, anf_full, xi, topo, sl, cl, sys_id, system_name = self.samples[idx]
        return (torch.tensor(nc), torch.tensor(xc), torch.tensor(anf_full),
                torch.tensor(xi), torch.tensor(topo, dtype=torch.long),
                torch.tensor(sl,  dtype=torch.long),
                torch.tensor(cl,  dtype=torch.long),
                sys_id, system_name)  # metadata only — not fed to the model


# ── Model ─────────────────────────────────────────────────────────────────
class Branch1_NullclineMLP(nn.Module):
    def __init__(self, input_dim=49, output_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(128,96),nn.BatchNorm1d(96),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(96,output_dim),nn.ReLU())
    def forward(self,x): return self.net(x)

class Branch1b_XcorrMLP(nn.Module):
    def __init__(self, input_dim=10, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(32,output_dim),nn.ReLU())
    def forward(self,x): return self.net(x)

class Branch1c_AllNewMLP(nn.Module):
    # V6: input_dim enlarged from 88 -> 110 (88 legacy + 12 v6_new + 10 fp_multi)
    def __init__(self, input_dim=110, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,160),nn.BatchNorm1d(160),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(160,80),nn.BatchNorm1d(80),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(80,output_dim),nn.ReLU())
    def forward(self,x): return self.net(x)

class Branch2_TermAttention(nn.Module):
    def __init__(self, n_terms=9, n_equations=2, embed_dim=32, output_dim=24):
        super().__init__()
        self.asym_proj=nn.Sequential(nn.Linear(n_terms,16),nn.ReLU(),nn.Linear(16,8))
        self.embed=nn.Linear(n_terms,embed_dim)
        self.attention=nn.MultiheadAttention(embed_dim=embed_dim,num_heads=4,batch_first=True)
        self.projection=nn.Sequential(
            nn.Linear(n_equations*embed_dim+8,64),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(64,output_dim),nn.ReLU())
    def forward(self,x):
        if x.shape[-1]!=2: x=x.transpose(-1,-2)
        x=x.reshape(x.shape[0],9,2)
        asym=self.asym_proj(x[:,:,0]-x[:,:,1])
        tokens=self.embed(x.permute(0,2,1))
        attn_out,_=self.attention(tokens,tokens,tokens)
        flat=attn_out.reshape(attn_out.shape[0],-1)
        return self.projection(torch.cat([flat,asym],dim=1))

class Branch3_TopologyEmbed(nn.Module):
    def __init__(self, n_classes=4, embed_dim=8, output_dim=8):
        super().__init__()
        self.embed=nn.Embedding(n_classes,embed_dim)
        self.proj=nn.Sequential(nn.Linear(embed_dim,output_dim),nn.ReLU())
    def forward(self,x): return self.proj(self.embed(x))

class LTInetV6(nn.Module):
    """
    merge_dim = 48 (nullcline) + 16 (xcorr) + 32 (allnew-enlarged) + 24 (term attn) + 8 (topology) = 128
    Same merge width as V5 — only Branch1c's INPUT grew (88->110); its
    OUTPUT stays 32, so downstream merge/head dimensions are unchanged
    and V5 weights are not directly loadable (Branch1c's first layer
    shape differs) but every other branch's architecture is identical.
    head_latent:   128->96->48->6   (Lat0-4 + Lat5 null)
    head_coupling: 128->64->32->4   (Ch0-2  + Ch3 null)
    """
    def __init__(self):
        super().__init__()
        self.branch1  = Branch1_NullclineMLP(49,48)
        self.branch1b = Branch1b_XcorrMLP(10,16)
        self.branch1c = Branch1c_AllNewMLP(BRANCH1C_DIM,32)
        self.branch2  = Branch2_TermAttention(9,output_dim=24)
        self.branch3  = Branch3_TopologyEmbed(4,output_dim=8)
        self.head_latent = nn.Sequential(
            nn.Linear(128,96),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(96,48),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(48,N_LATENT_CLASSES))
        self.head_coupling = nn.Sequential(
            nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,N_COUPLING_CLASSES))
    def forward(self,nc,xc,anf,xi,topo):
        merged=torch.cat([self.branch1(nc),self.branch1b(xc),
                          self.branch1c(anf),self.branch2(xi),
                          self.branch3(topo)],dim=1)
        return self.head_latent(merged),self.head_coupling(merged)


# ── Train / Evaluate ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, loss_fn_lat, loss_fn_coup, device,
                nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std):
    model.train()
    total_loss = lat_correct = coup_correct = total = 0
    for nc,xc,anf,xi,topo,slbl,clbl,_,_ in loader:
        nc   = (nc.to(device)  - nc_mean)  / nc_std
        xc   = (xc.to(device)  - xc_mean)  / xc_std
        anf  = (anf.to(device) - anf_mean) / anf_std
        xi   = xi.to(device); topo=topo.to(device)
        slbl = slbl.to(device); clbl=clbl.to(device)
        optimizer.zero_grad()
        ll, lc = model(nc,xc,anf,xi,topo)
        loss = loss_fn_lat(ll,slbl) + COUPLING_LOSS_WEIGHT*loss_fn_coup(lc,clbl)
        loss.backward(); optimizer.step()
        total_loss  += loss.item()
        lat_correct  += (ll.argmax(1)==slbl).sum().item()
        coup_correct += (lc.argmax(1)==clbl).sum().item()
        total        += slbl.size(0)
    return total_loss/len(loader), lat_correct/total, coup_correct/total


def evaluate(model, loader, loss_fn_lat, loss_fn_coup, device,
             nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std):
    model.eval()
    total_loss = lat_correct = coup_correct = total = 0
    with torch.no_grad():
        for nc,xc,anf,xi,topo,slbl,clbl,_,_ in loader:
            nc   = (nc.to(device)  - nc_mean)  / nc_std
            xc   = (xc.to(device)  - xc_mean)  / xc_std
            anf  = (anf.to(device) - anf_mean) / anf_std
            xi   = xi.to(device); topo=topo.to(device)
            slbl = slbl.to(device); clbl=clbl.to(device)
            ll, lc = model(nc,xc,anf,xi,topo)
            loss = loss_fn_lat(ll,slbl) + COUPLING_LOSS_WEIGHT*loss_fn_coup(lc,clbl)
            total_loss  += loss.item()
            lat_correct  += (ll.argmax(1)==slbl).sum().item()
            coup_correct += (lc.argmax(1)==clbl).sum().item()
            total        += slbl.size(0)
    return total_loss/len(loader), lat_correct/total, coup_correct/total


def evaluate_per_system(model, loader, device, nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std):
    """V6-specific diagnostic: accuracy broken out by system, since system
    identity is metadata-only (not a model input) but you'll want to know
    if e.g. toggle_switch or fitzhugh_nagumo underperforms relative to the
    original four systems — that's the actual test of whether the
    architecture generalizes rather than just memorizing per-system quirks."""
    model.eval()
    sys_correct_lat = Counter(); sys_correct_coup = Counter(); sys_total = Counter()
    with torch.no_grad():
        for nc,xc,anf,xi,topo,slbl,clbl,sys_id,sys_name in loader:
            nc   = (nc.to(device)  - nc_mean)  / nc_std
            xc   = (xc.to(device)  - xc_mean)  / xc_std
            anf  = (anf.to(device) - anf_mean) / anf_std
            xi   = xi.to(device); topo=topo.to(device)
            ll, lc = model(nc,xc,anf,xi,topo)
            lat_pred = ll.argmax(1).cpu(); coup_pred = lc.argmax(1).cpu()
            for i, name in enumerate(sys_name):
                sys_total[name] += 1
                if lat_pred[i] == slbl[i]: sys_correct_lat[name] += 1
                if coup_pred[i] == clbl[i]: sys_correct_coup[name] += 1
    print('\nPer-system validation accuracy:')
    for name in sorted(sys_total.keys()):
        n = sys_total[name]
        print(f'  {name:<24} n={n:4d}  Lat={sys_correct_lat[name]/n:.3f}  Coup={sys_correct_coup[name]/n:.3f}')


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    dataset = LTInetV6Dataset(data_dirs, sys_idx_lookup, NULLCLINE_DIM)
    if len(dataset) == 0:
        raise RuntimeError('No valid examples found. '
                           'Check data_dirs and run V6_annex_full.m first.')

    struct_counts   = Counter(s[5] for s in dataset.samples)
    coupling_counts = Counter(s[6] for s in dataset.samples)
    system_counts   = Counter(s[8] for s in dataset.samples)

    print(f'\nTotal examples: {len(dataset)}')
    print('Latent distribution:')
    for i in range(N_LATENT_CLASSES):
        print(f'  {latent_names[i]}: {struct_counts.get(i,0)}')
    print('Coupling distribution:')
    for i in range(N_COUPLING_CLASSES):
        print(f'  {coupling_names[i]}: {coupling_counts.get(i,0)}')
    print('System distribution:')
    for name in sorted(system_counts.keys()):
        print(f'  {name}: {system_counts[name]}')

    total_lat = sum(struct_counts.values())
    lat_weights = torch.tensor(
        [total_lat / (N_LATENT_CLASSES * max(struct_counts.get(i,1), 1))
         for i in range(N_LATENT_CLASSES)], dtype=torch.float32)
    lat_weights = lat_weights / lat_weights.mean()

    total_coup = sum(coupling_counts.values())
    coup_weights = torch.tensor(
        [total_coup / (N_COUPLING_CLASSES * max(coupling_counts.get(i,1), 1))
         for i in range(N_COUPLING_CLASSES)], dtype=torch.float32)
    coup_weights = coup_weights / coup_weights.mean()

    print(f'\nLatent class weights:   {lat_weights.numpy().round(3)}')
    print(f'Coupling class weights: {coup_weights.numpy().round(3)}')

    all_nc  = np.stack([s[0] for s in dataset.samples])
    all_xc  = np.stack([s[1] for s in dataset.samples])
    all_anf = np.stack([s[2] for s in dataset.samples])

    def make_norm(arr):
        mean = torch.tensor(arr.mean(0), dtype=torch.float32)
        std  = torch.tensor(arr.std(0),  dtype=torch.float32)
        std[std < 1e-6] = 1.0
        return mean, std

    nc_mean,  nc_std  = make_norm(all_nc)
    xc_mean,  xc_std  = make_norm(all_xc)
    anf_mean, anf_std = make_norm(all_anf)

    n_train = int(0.8*len(dataset))
    n_val   = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train,n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False, drop_last=False)
    print(f'Train: {n_train}  Val: {n_val}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    nc_mean  = nc_mean.to(device);  nc_std  = nc_std.to(device)
    xc_mean  = xc_mean.to(device);  xc_std  = xc_std.to(device)
    anf_mean = anf_mean.to(device); anf_std = anf_std.to(device)
    lat_weights  = lat_weights.to(device)
    coup_weights = coup_weights.to(device)

    model     = LTInetV6().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)
    loss_fn_lat  = nn.CrossEntropyLoss(weight=lat_weights,  label_smoothing=0.05)
    loss_fn_coup = nn.CrossEntropyLoss(weight=coup_weights, label_smoothing=0.05)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params:,}')

    SAVE_PATH    = f'{WEIGHTS_DIR}/LTInetV6_full_best.pth'
    best_avg_acc = 0.0
    n_epochs     = 300
    es_patience  = 60
    no_improve   = 0

    train_losses=[]; val_losses=[]
    train_lat_accs=[]; val_lat_accs=[]
    train_coup_accs=[]; val_coup_accs=[]

    for epoch in range(1, n_epochs+1):
        tr_loss,tr_lat,tr_coup = train_epoch(
            model,train_loader,optimizer,loss_fn_lat,loss_fn_coup,device,
            nc_mean,nc_std,xc_mean,xc_std,anf_mean,anf_std)
        va_loss,va_lat,va_coup = evaluate(
            model,val_loader,loss_fn_lat,loss_fn_coup,device,
            nc_mean,nc_std,xc_mean,xc_std,anf_mean,anf_std)
        scheduler.step(va_loss)
        va_avg = (va_lat+va_coup)/2.0

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_lat_accs.append(tr_lat); val_lat_accs.append(va_lat)
        train_coup_accs.append(tr_coup); val_coup_accs.append(va_coup)

        if va_avg > best_avg_acc:
            best_avg_acc = va_avg; no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            np.save(f'{WEIGHTS_DIR}/V6_nc_mean.npy',  nc_mean.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V6_nc_std.npy',   nc_std.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V6_xc_mean.npy',  xc_mean.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V6_xc_std.npy',   xc_std.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V6_anf_mean.npy', anf_mean.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V6_anf_std.npy',  anf_std.cpu().numpy())
        else:
            no_improve += 1

        print(f'Epoch {epoch:3d} | Loss {tr_loss:.4f}/{va_loss:.4f} | '
              f'Lat {tr_lat:.3f}/{va_lat:.3f} | '
              f'Coup {tr_coup:.3f}/{va_coup:.3f} | '
              f'Avg {va_avg:.3f} | LR {optimizer.param_groups[0]["lr"]:.1e}')

        if epoch % 20 == 0:
            ep = list(range(1, epoch+1))
            fig, axes = plt.subplots(1,3,figsize=(15,4))
            axes[0].plot(ep,train_losses,'b-',label='Train')
            axes[0].plot(ep,val_losses,'r--',label='Val')
            axes[0].set_title('Combined loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
            axes[1].plot(ep,train_lat_accs,'b-',label='Train')
            axes[1].plot(ep,val_lat_accs,'r--',label='Val')
            axes[1].axhline(1/N_LATENT_CLASSES,color='gray',linestyle=':',
                            label=f'Chance ({100/N_LATENT_CLASSES:.0f}%)')
            axes[1].set_ylim([0,1]); axes[1].set_title('Latent accuracy')
            axes[1].legend(); axes[1].grid(alpha=0.3)
            axes[2].plot(ep,train_coup_accs,'b-',label='Train')
            axes[2].plot(ep,val_coup_accs,'r--',label='Val')
            axes[2].axhline(1/N_COUPLING_CLASSES,color='gray',linestyle=':',
                            label=f'Chance ({100/N_COUPLING_CLASSES:.0f}%)')
            axes[2].set_ylim([0,1]); axes[2].set_title('Coupling accuracy')
            axes[2].legend(); axes[2].grid(alpha=0.3)
            fig.suptitle(f'LTInetV6 | Epoch {epoch} | '
                         f'Lat {va_lat:.3f} | Coup {va_coup:.3f} | '
                         f'Best avg {best_avg_acc:.3f}')
            plt.tight_layout(); plt.show()

        if no_improve >= es_patience:
            print(f'\nEarly stopping at epoch {epoch}'); break

    print(f'\nTraining complete. Best avg val acc: {best_avg_acc:.3f}')
    print(f'Weights: {SAVE_PATH}')

    # V6-specific: per-system breakdown, since this is the actual test of
    # whether the architecture generalizes across genuinely different
    # dynamical regimes rather than just performing well on average.
    evaluate_per_system(model, val_loader, device, nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std)