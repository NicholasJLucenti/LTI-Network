import os
import glob
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import savgol_filter, hilbert

# ================================================================
# V9_UDE_Annex.py
#
# Replaces the SINDy-dependent portions of the existing feature files
# with UDE-derived equivalents — a direct in-place substitute for what
# V9_annex_full.m did, computed in Python instead of MATLAB since UDE
# fitting needs a differentiable optimizer.
#
# KEY DESIGN FACT (checked directly against the existing pipeline
# before building this): most of `all_new_features` (88-dim) does NOT
# depend on SINDy at all — RQA, cross-RQA, wavelet energy, Poincare,
# and amplitude-envelope features (47 of 88 dims) are computed straight
# from the raw trajectory. ONLY the phase-resolved residual features
# (32 dims: indices 0-31) and SINDy coefficient magnitudes (9 dims:
# indices 79-87) depend on which mechanistic model was used. This
# script recomputes ONLY those two blocks from the UDE fit and copies
# indices 32-78 (amp+rqa+crqa+wavelet+poincare) straight through
# UNCHANGED from what MATLAB already computed — lower risk than
# reimplementing those feature functions in Python, and means
# LTInetV9.py itself needs ZERO changes: it just transparently trains
# on whichever features are actually saved in the files.
#
# `v6_new_features` (12-dim, previously FNN+TransferEntropy+
# ResidualSpectrum, computed from the SINDy residual) is REPLACED
# ENTIRELY with 12 new UDE-specific diagnostic summary statistics —
# more directly targeted at "how much/where is the mechanistic model
# failing" than the features it replaces, per the stated goal.
#
# `fp_multi_features` (10-dim, toggle-switch-only geometry) is
# untouched — doesn't depend on the mechanistic model choice.
#
# COMPUTATIONAL STRATEGY: the expensive part (per-trajectory gradient-
# descent fitting) is BATCHED — many independent UDEs trained in
# parallel via torch.einsum, not a Python loop over tens of thousands
# of files one at a time. At real dataset scale (V9: ~50-60k examples),
# a naive one-at-a-time loop with ~1000+ Adam epochs each would take on
# the order of days; batched fitting on GPU brings this down to a
# tractable range. Post-fit diagnostic computation (phase binning, etc)
# is cheap relative to training and left as a straightforward per-file
# loop for correctness/simplicity.
#
# Run this AFTER V9_ProductionGen.m + V9_annex_full.m have already
# produced the standard feature files (this script needs the EXISTING
# all_new_features to copy the unchanged portion from, plus
# x_data_all/y_data_all which V9_ProductionGen.m already saves).
# ================================================================

# ── Config ──────────────────────────────────────────────────────────
DATA_ROOT = r'C:/Users/nickj/LTInetV9 Local Data Drive'
N = 300           # observed-window length used throughout the pipeline
DT = 0.05
SG_POLYORDER = 3
SG_WINDOW = 11

HIDDEN_DIM = 8
HILL_N = 3
N_EPOCHS = 800     # fewer than the standalone diagnostic tool's 3000 —
                   # convergence was already fast by epoch ~500 in
                   # testing; kept modest here since this runs per-chunk
                   # across the whole dataset, not once per trajectory
LR = 1e-2
L1_PENALTY_XI = 5e-5     # see LTInetV9_UDE_diagnostic_standalone.py's
                         # comment on this value — an approximation of
                         # STRidge sparsity, not equivalent to it
CORRECTION_REG = 1e-3

CHUNK_SIZE = 2000        # number of trajectories fit together per batch
N_BINS = 8                # phase bins, matches existing convention
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ANNEX_FLAG = 'ude_annex_complete'


# ── Batched UDE model ─────────────────────────────────────────────────
class BatchedKnownLibrary(nn.Module):
    """Same 9-term 'standard' library as SINDy used, but with B
    independent sets of coefficients (one per trajectory in the batch),
    fit simultaneously via batched tensor ops instead of one at a time."""
    def __init__(self, batch_size, hill_n, device):
        super().__init__()
        self.xi_x = nn.Parameter(torch.zeros(batch_size, 9, device=device))
        self.xi_y = nn.Parameter(torch.zeros(batch_size, 9, device=device))
        self.raw_k0 = nn.Parameter(torch.zeros(batch_size, device=device))
        self.raw_km = nn.Parameter(torch.zeros(batch_size, device=device))
        self.hill_n = hill_n

    def theta(self, x, y):
        # x, y: (B, N)
        k0 = F.softplus(self.raw_k0).unsqueeze(1) + 1e-3   # (B,1)
        km = F.softplus(self.raw_km).unsqueeze(1) + 1e-3
        n = self.hill_n
        hrep = k0**n / (k0**n + y**n + 1e-8)
        hdeg = y / (km + y + 1e-8)
        return torch.stack([torch.ones_like(x), x, x**2, x**3,
                             y, y**2, y**3, hrep, hdeg], dim=-1)  # (B,N,9)

    def forward(self, x, y):
        th = self.theta(x, y)                                  # (B,N,9)
        dx = (th * self.xi_x.unsqueeze(1)).sum(-1)              # (B,N)
        dy = (th * self.xi_y.unsqueeze(1)).sum(-1)
        return dx, dy


class BatchedCorrection(nn.Module):
    """B independent small MLPs, trained in parallel via batched
    matmul (einsum) rather than B separate nn.Linear layers — the
    standard trick for training many small independent models
    efficiently on GPU."""
    def __init__(self, batch_size, hidden_dim, device):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(batch_size, 2, hidden_dim, device=device) * 0.3)
        self.b1 = nn.Parameter(torch.zeros(batch_size, hidden_dim, device=device))
        self.W2 = nn.Parameter(torch.randn(batch_size, hidden_dim, 2, device=device) * 0.3)
        self.b2 = nn.Parameter(torch.zeros(batch_size, 2, device=device))

    def forward(self, x, y):
        # x, y: (B, N)
        inp = torch.stack([x, y], dim=-1)                       # (B,N,2)
        h = torch.tanh(torch.einsum('bnf,bfh->bnh', inp, self.W1) + self.b1.unsqueeze(1))
        out = torch.einsum('bnh,bho->bno', h, self.W2) + self.b2.unsqueeze(1)  # (B,N,2)
        return out[..., 0], out[..., 1]


class BatchedUDE(nn.Module):
    def __init__(self, batch_size, hill_n, hidden_dim, device):
        super().__init__()
        self.known = BatchedKnownLibrary(batch_size, hill_n, device)
        self.correction = BatchedCorrection(batch_size, hidden_dim, device)

    def drift(self, x, y):
        dx_k, dy_k = self.known(x, y)
        dx_c, dy_c = self.correction(x, y)
        return dx_k + dx_c, dy_k + dy_c, dx_k, dy_k, dx_c, dy_c


# ── Collect all example files ──────────────────────────────────────
def find_all_example_files(data_root):
    files = []
    for folder in glob.glob(os.path.join(data_root, '*_NNdata')):
        files.extend(glob.glob(os.path.join(folder, 'example_*.mat')))
    return sorted(files)


def process_chunk(filepaths):
    """Loads a chunk of trajectories, fits a batched UDE, computes UDE-
    based replacement features, and writes them back into each file.
    Returns (n_processed, n_skipped)."""
    B = len(filepaths)

    x_batch = np.zeros((B, N), dtype=np.float32)
    y_batch = np.zeros((B, N), dtype=np.float32)
    mats = []
    valid_mask = np.ones(B, dtype=bool)

    for i, fp in enumerate(filepaths):
        try:
            mat = scipy.io.loadmat(fp)
        except Exception:
            mats.append(None); valid_mask[i] = False; continue

        if ANNEX_FLAG in mat:
            mats.append(mat)  # already done, will be skipped below but keep for count
            valid_mask[i] = False
            continue

        if 'x_data_all' not in mat or 'y_data_all' not in mat or 'all_new_features' not in mat:
            mats.append(None); valid_mask[i] = False; continue

        xa = mat['x_data_all'].squeeze().astype(np.float32)
        ya = mat['y_data_all'].squeeze().astype(np.float32)
        if len(xa) < N or len(ya) < N:
            mats.append(None); valid_mask[i] = False; continue

        x_batch[i] = xa[:N]
        y_batch[i] = ya[:N]
        mats.append(mat)

    n_already_done = sum(1 for i in range(B) if mats[i] is not None and ANNEX_FLAG in mats[i])
    n_bad = sum(1 for i in range(B) if mats[i] is None)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return 0, n_bad, n_already_done

    Bv = len(valid_idx)
    x_v = x_batch[valid_idx]
    y_v = y_batch[valid_idx]

    # Smoothed observed derivatives, batched via savgol_filter's axis arg
    dx_obs = savgol_filter(np.gradient(x_v, DT, axis=1), SG_WINDOW, SG_POLYORDER, axis=1)
    dy_obs = savgol_filter(np.gradient(y_v, DT, axis=1), SG_WINDOW, SG_POLYORDER, axis=1)

    x_t = torch.tensor(x_v, device=DEVICE)
    y_t = torch.tensor(y_v, device=DEVICE)
    dx_obs_t = torch.tensor(dx_obs, device=DEVICE)
    dy_obs_t = torch.tensor(dy_obs, device=DEVICE)

    model = BatchedUDE(Bv, HILL_N, HIDDEN_DIM, DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        dx_pred, dy_pred, _, _, dx_c, dy_c = model.drift(x_t, y_t)
        loss_fit = ((dx_pred - dx_obs_t)**2).mean(dim=1) + ((dy_pred - dy_obs_t)**2).mean(dim=1)  # (Bv,)
        loss_reg = CORRECTION_REG * ((dx_c**2).mean(dim=1) + (dy_c**2).mean(dim=1))
        loss_sparsity = L1_PENALTY_XI * (model.known.xi_x.abs().sum(dim=1) + model.known.xi_y.abs().sum(dim=1))
        loss = (loss_fit + loss_reg + loss_sparsity).sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        dx_pred, dy_pred, dx_known, dy_known, dx_corr, dy_corr = model.drift(x_t, y_t)

    dx_pred_np = dx_pred.cpu().numpy()
    dy_pred_np = dy_pred.cpu().numpy()
    known_mag = torch.sqrt(dx_known**2 + dy_known**2).cpu().numpy()
    corr_mag = torch.sqrt(dx_corr**2 + dy_corr**2).cpu().numpy()
    ratio = corr_mag / (known_mag + 1e-6)

    xi_x_np = model.known.xi_x.detach().cpu().numpy()
    xi_y_np = model.known.xi_y.detach().cpu().numpy()
    k0_np = F.softplus(model.known.raw_k0).detach().cpu().numpy()
    km_np = F.softplus(model.known.raw_km).detach().cpu().numpy()

    resid_dx = dx_obs - dx_pred_np    # (Bv, N)
    resid_dy = dy_obs - dy_pred_np

    n_written = 0
    for local_i, orig_i in enumerate(valid_idx):
        mat = mats[orig_i]
        x_ex = x_v[local_i]
        y_ex = y_v[local_i]

        # ── Phase-resolved UDE-residual features (32 dims, replacing
        #    indices 0-31 of all_new_features) ──────────────────────
        x_cent = x_ex - x_ex.mean()
        inst_phase = np.mod(np.angle(hilbert(x_cent)), 2*np.pi)
        bin_edges = np.linspace(0, 2*np.pi, N_BINS+1)

        r_std_x = resid_dx[local_i].std()
        if r_std_x < 1e-8:
            n_bad += 1
            continue
        phase_mean_dx = np.zeros(N_BINS); phase_std_dx = np.zeros(N_BINS)
        for b in range(N_BINS):
            mask = (inst_phase >= bin_edges[b]) & (inst_phase < bin_edges[b+1])
            if mask.sum() >= 3:
                phase_mean_dx[b] = resid_dx[local_i][mask].mean() / r_std_x
                phase_std_dx[b] = resid_dx[local_i][mask].std() / r_std_x
        phase_x_features = np.clip(np.concatenate([phase_mean_dx, phase_std_dx]), -10, 10)

        r_std_y = resid_dy[local_i].std()
        if r_std_y < 1e-8:
            r_std_y = 1.0
        phase_mean_dy = np.zeros(N_BINS); phase_std_dy = np.zeros(N_BINS)
        for b in range(N_BINS):
            mask = (inst_phase >= bin_edges[b]) & (inst_phase < bin_edges[b+1])
            if mask.sum() >= 3:
                phase_mean_dy[b] = resid_dy[local_i][mask].mean() / r_std_y
                phase_std_dy[b] = resid_dy[local_i][mask].std() / r_std_y
        phase_y_features = np.clip(np.concatenate([phase_mean_dy, phase_std_dy]), -10, 10)

        # ── UDE known-library coefficient magnitudes (9 dims,
        #    replacing indices 79-87) — direct analog of old xi_mag ──
        xi_norm = np.linalg.norm(xi_x_np[local_i], 2)
        if xi_norm < 1e-10:
            xi_norm = 1.0
        xi_mag_features = np.clip(np.abs(xi_x_np[local_i]) / xi_norm, 0, 5)

        # ── Assemble new all_new_features: UDE phase (0-31) + UNCHANGED
        #    amp/rqa/crqa/wavelet/poincare (32-78, copied straight from
        #    the existing MATLAB-computed array) + UDE xi_mag (79-87) ──
        old_anf = mat['all_new_features'].squeeze().astype(np.float32)
        if old_anf.shape[0] != 88:
            n_bad += 1
            continue
        new_anf = old_anf.copy()
        new_anf[0:32] = phase_x_features.tolist() + phase_y_features.tolist()
        new_anf[79:88] = xi_mag_features
        if not np.all(np.isfinite(new_anf)):
            n_bad += 1
            continue

        # ── New UDE diagnostic summary (12 dims, replacing
        #    v6_new_features entirely) — the actual "mechanistic model
        #    failure" signal this framework exists to produce ─────────
        ratio_ex = ratio[local_i]
        rel_rmse_dx = np.sqrt(((resid_dx[local_i])**2).mean()) / (dx_obs[local_i].std() + 1e-8)
        rel_rmse_dy = np.sqrt(((resid_dy[local_i])**2).mean()) / (dy_obs[local_i].std() + 1e-8)
        ude_summary = np.array([
            ratio_ex.mean(),
            ratio_ex.max(),
            ratio_ex.std(),
            corr_mag[local_i].mean(),
            corr_mag[local_i].max(),
            rel_rmse_dx,
            rel_rmse_dy,
            k0_np[local_i],
            km_np[local_i],
            (ratio_ex > 0.5).mean(),      # fraction of trajectory where correction is substantial
            (ratio_ex > 1.0).mean(),      # fraction where correction DOMINATES the known term
            corr_mag[local_i].mean() / (dx_obs[local_i].std() + 1e-8),  # correction scale vs observed dynamics scale
        ], dtype=np.float32)
        ude_summary = np.clip(ude_summary, -50, 50)
        if not np.all(np.isfinite(ude_summary)):
            n_bad += 1
            continue

        # ── Write back, preserving every other existing field exactly ──
        # Exclude 'params_saved' — the generation scripts attach actual
        # MATLAB function handles onto it (p.hill_rep, p.hill_deg), which
        # scipy.io can read fine but cannot write back out
        # (MatWriteError: 'Cannot write matlab functions'). Not used
        # anywhere in LTInetV9.py's training pipeline, so dropping it is
        # safe rather than trying to sanitize function handles out of a
        # nested struct.
        EXCLUDE_FIELDS = {'params_saved'}
        mat_out = {k: v for k, v in mat.items()
                   if not k.startswith('__') and k not in EXCLUDE_FIELDS}
        mat_out['all_new_features'] = new_anf.reshape(1, -1)
        mat_out['v6_new_features'] = ude_summary.reshape(1, -1)
        mat_out[ANNEX_FLAG] = True
        mat_out['ude_xi_x'] = xi_x_np[local_i].reshape(1, -1)
        mat_out['ude_xi_y'] = xi_y_np[local_i].reshape(1, -1)

        scipy.io.savemat(filepaths[orig_i], mat_out)
        n_written += 1

    return n_written, n_bad, n_already_done


if __name__ == '__main__':
    print(f'Using device: {DEVICE}')
    all_files = find_all_example_files(DATA_ROOT)
    print(f'Found {len(all_files)} example files under {DATA_ROOT}')
    if len(all_files) == 0:
        raise RuntimeError('No example_*.mat files found — check DATA_ROOT.')

    total_written = total_bad = total_skipped_done = 0
    n_chunks = (len(all_files) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for c in range(n_chunks):
        chunk = all_files[c*CHUNK_SIZE:(c+1)*CHUNK_SIZE]
        print(f'\n--- Chunk {c+1}/{n_chunks} ({len(chunk)} files) ---')
        written, bad, already_done = process_chunk(chunk)
        total_written += written
        total_bad += bad
        total_skipped_done += already_done
        print(f'  Written: {written}  Bad/skipped: {bad}  Already annexed: {already_done}')

    print(f'\n=== DONE ===')
    print(f'Total written: {total_written}')
    print(f'Total bad/skipped: {total_bad}')
    print(f'Total already annexed (from a prior run): {total_skipped_done}')
    print('\nLTInetV9.py needs NO changes — it will transparently train on')
    print('these UDE-derived all_new_features/v6_new_features values.')