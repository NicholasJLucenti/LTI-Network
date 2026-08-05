import os
import glob
import numpy as np
import scipy.io
import torch
from scipy.signal import savgol_filter, hilbert
from collections import defaultdict

# ================================================================
# V11_DEN_Annex.py
#
# Trimmed from V10_DEN_Annex.py per the V11 feature cut ("error signals
# only, no mechanistic fitting"):
#
#   REMOVED ENTIRELY:
#     - Stage 2 (den_xi_w): library regression of w(t) onto the known
#       term library. denxi scored ~dead in permutation importance and
#       is cut on principle regardless of score (per the V11 spec).
#     - build_library_np / LIBID_BY_SYSTEM: only existed to support
#       Stage 2 — no remaining caller.
#     - den_w_x / den_w_y raw arrays: never a feature themselves, only
#       ever consumed to derive den_w_phase_features / den_w_xcorr_features
#       / den_energy / den_fit_rmse below, so they're no longer written
#       to the output .mat at all (still computed in-memory, just not saved).
#
#   SLICED (kept subset of the phase-binned block only — xcorr block and
#   rmse/energy scalars survive in full):
#     - den_w_phase_features: 32 -> 26 dims. phase_mean_wx/wy survive in
#       full (8 each); phase_std_wx keeps [1,2,3,4,6] of 8 (0-indexed,
#       matching the V11 trainer's index convention); phase_std_wy keeps
#       [2,3,5,6,7] of 8.
#     - den_w_xcorr_features: unchanged, all 12 lags survive.
#     - den_fit_rmse_x/y, den_energy_x/y: unchanged, all 4 survive.
#
# Fitting logic (DEN Stage 1 elastic-net w(t) reconstruction against each
# system's known base ODE) is UNCHANGED — it's what den_w_x/w_y, and
# therefore every surviving DEN feature, is derived from.
# ================================================================

DATA_ROOT = r'C:/Users/nickj/LTInetV11 Local Data Drive'
N = 300
DT = 0.05
SG_POLYORDER = 3
SG_WINDOW = 11

N_EPOCHS = 600
LR = 2e-2
LAMBDA1 = 5e-3          # L1 sparsity on w itself
LAMBDA2 = 1e-3          # L2 shrinkage on w itself
LAMBDA_SMOOTH = 1e-2    # L2 penalty on finite-difference dw/dt

CHUNK_SIZE = 1500        # trajectories fit together per batch, WITHIN one system
N_BINS = 8                # phase bins, matches existing convention
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ANNEX_FLAG = 'den_annex_v11_complete'

# Kept phase-block indices (0-indexed, into each 8-length sub-array)
KEEP_PHASE_STD_WX = [1, 2, 3, 4, 6]
KEEP_PHASE_STD_WY = [2, 3, 5, 6, 7]

# Params required per system, in the exact order used when constructing
# the batched parameter tensor dict — must match V11_SystemLib.m field names.
PARAMS_BY_SYSTEM = {
    'goodwin': ['alpha', 'd1', 'ks', 'Vmax', 'hill_k0', 'hill_n', 'hill_km'],
    'brusselator': ['a', 'b', 'hill_k0', 'hill_n', 'hill_km'],
    'repressilator': ['alpha', 'delta', 'beta_rep', 'gamma_rep', 'hill_k0', 'hill_n', 'hill_km'],
    'van_der_pol': ['mu', 'omega', 'hill_k0', 'hill_n', 'hill_km'],
    'fitzhugh_nagumo': ['a', 'b', 'tau', 'I_ext', 'hill_k0', 'hill_n', 'hill_km'],
    'rosenzweig_macarthur': ['r', 'K', 'a_rate', 'h', 'e', 'm', 'hill_k0', 'hill_n', 'hill_km'],
    'toggle_switch': ['alpha1', 'alpha2', 'beta', 'gamma', 'hill_k0', 'hill_n', 'hill_km'],
}


def hill_rep(z, k, n):
    return k**n / (k**n + z**n + 1e-8)


def hill_deg(y, km):
    return y / (km + y + 1e-8)


def base_ode_batched(system_name, x, y, p):
    """x, y: (B,N). p: dict of (B,1) tensors, broadcasts over N."""
    if system_name == 'goodwin':
        dx = p['alpha'] * hill_rep(y, p['hill_k0'], p['hill_n']) - p['d1'] * x
        dy = p['ks'] * x - p['Vmax'] * hill_deg(y, p['hill_km'])
    elif system_name == 'brusselator':
        dx = p['a'] - (p['b'] + 1) * x + x**2 * y
        dy = p['b'] * x - x**2 * y
    elif system_name == 'repressilator':
        dx = p['alpha'] * hill_rep(y, p['hill_k0'], p['hill_n']) - p['delta'] * hill_deg(x, p['hill_km'])
        dy = p['beta_rep'] * x - p['gamma_rep'] * y
    elif system_name == 'van_der_pol':
        dx = p['mu'] * (1 - y**2) * x - p['omega'] * y
        dy = x
    elif system_name == 'fitzhugh_nagumo':
        dx = x - x**3 / 3 - y + p['I_ext']
        dy = (x + p['a'] - p['b'] * y) / p['tau']
    elif system_name == 'rosenzweig_macarthur':
        denom = 1 + p['a_rate'] * p['h'] * x
        dx = p['r'] * x * (1 - x / p['K']) - p['a_rate'] * x * y / denom
        dy = p['e'] * p['a_rate'] * x * y / denom - p['m'] * y
    elif system_name == 'toggle_switch':
        x_safe = x.clamp(min=1e-6)
        y_safe = y.clamp(min=1e-6)
        dx = p['alpha1'] / (1 + y_safe**p['beta']) - x
        dy = p['alpha2'] / (1 + x_safe**p['gamma']) - y
    else:
        raise ValueError(f'Unknown system: {system_name}')
    return dx, dy


def extract_params(mat):
    if 'params_numeric' not in mat:
        raise KeyError(
            "params_numeric" ' — check V11_Gen.m; params_saved contains '
            'MATLAB function handles scipy cannot read.')
    ps = mat['params_numeric']
    for _ in range(10):  # bounded, not while-True
        if isinstance(ps, np.ndarray) and ps.dtype == object and ps.size == 1:
            ps = ps.reshape(-1)[0]
        else:
            break
    p = {}
    for name in ps.dtype.names:
        val = ps[name]
        # A 0-d numpy array (shape ()) stays an ndarray under .squeeze()
        # forever — np.asarray(...).flatten() always terminates in one step.
        arr = np.asarray(val).flatten()
        if arr.size == 0:
            continue
        try:
            p[name] = float(arr[0])
        except (TypeError, ValueError):
            continue
    return p


SYSTEM_BY_SYS_IDX = {
    1: 'goodwin', 2: 'brusselator', 3: 'repressilator', 4: 'van_der_pol',
    5: 'fitzhugh_nagumo', 6: 'rosenzweig_macarthur', 7: 'toggle_switch',
}


def group_files_by_system(data_root):
    """Derives system_name from the folder name (V11Lat{..}_..._Sys{idx}_NNdata)
    instead of opening every .mat file — Sys{idx} maps deterministically to
    V11_SystemLib.m's ordering, so no file I/O is needed for this step."""
    groups = defaultdict(list)
    folders = sorted(glob.glob(os.path.join(data_root, '*_NNdata')))
    for folder in folders:
        base = os.path.basename(folder)
        if '_Sys' not in base:
            continue
        try:
            sys_idx = int(base.split('_Sys')[1].split('_')[0])
        except (IndexError, ValueError):
            continue
        sysname = SYSTEM_BY_SYS_IDX.get(sys_idx)
        if sysname is None:
            continue
        groups[sysname].extend(sorted(glob.glob(os.path.join(folder, '*.mat'))))
    return groups


def fit_chunk(system_name, filepaths, skip_reasons):
    param_names = PARAMS_BY_SYSTEM[system_name]
    mats = []
    for fp in filepaths:
        try:
            mats.append(scipy.io.loadmat(fp))
        except Exception as e:
            skip_reasons[f'load_failed: {e}'] += 1
            mats.append(None)

    valid_idx, x_list, y_list, p_list = [], [], [], []
    n_bad = 0
    for i, mat in enumerate(mats):
        if mat is None:
            n_bad += 1
            continue
        if mat.get(ANNEX_FLAG, False):
            continue
        if 'x_data_all' not in mat or 'y_data_all' not in mat:
            skip_reasons['missing_x_or_y'] += 1
            n_bad += 1
            continue
        try:
            p = extract_params(mat)
        except Exception as e:
            skip_reasons[f'param_extract_failed: {e}'] += 1
            n_bad += 1
            continue
        if any(pn not in p for pn in param_names):
            skip_reasons['missing_required_param'] += 1
            n_bad += 1
            continue
        x = mat['x_data_all'].squeeze().astype(np.float64)
        y = mat['y_data_all'].squeeze().astype(np.float64)
        if len(x) != N or len(y) != N:
            skip_reasons[f'bad_length_{len(x)}_{len(y)}'] += 1
            n_bad += 1
            continue
        valid_idx.append(i)
        x_list.append(x)
        y_list.append(y)
        p_list.append(p)

    Bv = len(valid_idx)
    if Bv == 0:
        return 0, n_bad, skip_reasons

    x_v = np.stack(x_list, axis=0)
    y_v = np.stack(y_list, axis=0)
    p_valid = p_list

    dx_obs = savgol_filter(np.gradient(x_v, DT, axis=1), SG_WINDOW, SG_POLYORDER, axis=1)
    dy_obs = savgol_filter(np.gradient(y_v, DT, axis=1), SG_WINDOW, SG_POLYORDER, axis=1)

    x_t = torch.tensor(x_v, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_v, dtype=torch.float32, device=DEVICE)
    dx_obs_t = torch.tensor(dx_obs, dtype=torch.float32, device=DEVICE)
    dy_obs_t = torch.tensor(dy_obs, dtype=torch.float32, device=DEVICE)

    p_batched = {}
    for pn in param_names:
        vals = np.array([p_valid[j][pn] for j in range(Bv)], dtype=np.float32).reshape(-1, 1)
        p_batched[pn] = torch.tensor(vals, device=DEVICE)

    w_x = torch.zeros(Bv, N, device=DEVICE, requires_grad=True)
    w_y = torch.zeros(Bv, N, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([w_x, w_y], lr=LR)

    print(f'    fitting {Bv} trajectories x {N_EPOCHS} epochs on {DEVICE}...')
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        dx_known, dy_known = base_ode_batched(system_name, x_t, y_t, p_batched)
        resid_x = dx_obs_t - dx_known - w_x
        resid_y = dy_obs_t - dy_known - w_y
        loss_fit = (resid_x**2).mean(dim=1) + (resid_y**2).mean(dim=1)          # (Bv,)
        loss_l1 = LAMBDA1 * (w_x.abs().mean(dim=1) + w_y.abs().mean(dim=1))
        loss_l2 = LAMBDA2 * ((w_x**2).mean(dim=1) + (w_y**2).mean(dim=1))
        loss_smooth = LAMBDA_SMOOTH * (((w_x[:, 1:] - w_x[:, :-1])**2).mean(dim=1) +
                                        ((w_y[:, 1:] - w_y[:, :-1])**2).mean(dim=1))
        loss = (loss_fit + loss_l1 + loss_l2 + loss_smooth).sum()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == N_EPOCHS - 1:
            print(f'      epoch {epoch:4d}/{N_EPOCHS}  mean_fit_loss={loss_fit.mean().item():.5f}')

    with torch.no_grad():
        dx_known, dy_known = base_ode_batched(system_name, x_t, y_t, p_batched)
        resid_x_final = dx_obs_t - dx_known - w_x
        resid_y_final = dy_obs_t - dy_known - w_y

    w_x_np = w_x.detach().cpu().numpy()
    w_y_np = w_y.detach().cpu().numpy()
    rmse_x = np.sqrt((resid_x_final.cpu().numpy()**2).mean(axis=1))
    rmse_y = np.sqrt((resid_y_final.cpu().numpy()**2).mean(axis=1))
    energy_x = np.sqrt((w_x_np**2).mean(axis=1))
    energy_y = np.sqrt((w_y_np**2).mean(axis=1))

    print(f'    fitting done, writing {Bv} files...')
    n_written = 0
    for local_i, orig_i in enumerate(valid_idx):
        if local_i > 0 and local_i % 200 == 0:
            print(f'      writing {local_i}/{Bv}...')
        mat = mats[orig_i]
        x_ex = x_v[local_i]
        y_ex = y_v[local_i]
        wx_ex = w_x_np[local_i]
        wy_ex = w_y_np[local_i]

        # ── Phase-resolved w_x/w_y features, sliced to V11-kept dims ────
        x_cent = x_ex - x_ex.mean()
        inst_phase = np.mod(np.angle(hilbert(x_cent)), 2 * np.pi)
        bin_edges = np.linspace(0, 2 * np.pi, N_BINS + 1)

        std_wx = wx_ex.std() + 1e-8
        std_wy = wy_ex.std() + 1e-8
        phase_mean_wx = np.zeros(N_BINS); phase_std_wx = np.zeros(N_BINS)
        phase_mean_wy = np.zeros(N_BINS); phase_std_wy = np.zeros(N_BINS)
        for b in range(N_BINS):
            mask = (inst_phase >= bin_edges[b]) & (inst_phase < bin_edges[b + 1])
            if mask.sum() >= 3:
                phase_mean_wx[b] = wx_ex[mask].mean() / std_wx
                phase_std_wx[b] = wx_ex[mask].std() / std_wx
                phase_mean_wy[b] = wy_ex[mask].mean() / std_wy
                phase_std_wy[b] = wy_ex[mask].std() / std_wy

        den_w_phase_features = np.clip(np.concatenate([
            phase_mean_wx,
            phase_std_wx[KEEP_PHASE_STD_WX],
            phase_mean_wy,
            phase_std_wy[KEEP_PHASE_STD_WY],
        ]), -10, 10).astype(np.float32)  # 8+5+8+5 = 26

        # ── Cross-correlation of w_x/w_y against x/y at 3 lags (12, all kept) ─
        lags = [-5, 0, 5]

        def xcorr_at_lags(a, b, lags):
            a = (a - a.mean()) / (a.std() + 1e-8)
            b = (b - b.mean()) / (b.std() + 1e-8)
            out = []
            for lag in lags:
                if lag == 0:
                    out.append(np.mean(a * b))
                elif lag > 0:
                    out.append(np.mean(a[lag:] * b[:-lag]) if lag < len(a) else 0.0)
                else:
                    out.append(np.mean(a[:lag] * b[-lag:]) if -lag < len(a) else 0.0)
            return out

        den_w_xcorr_features = np.array(
            xcorr_at_lags(wx_ex, x_ex, lags) + xcorr_at_lags(wx_ex, y_ex, lags) +
            xcorr_at_lags(wy_ex, x_ex, lags) + xcorr_at_lags(wy_ex, y_ex, lags),
            dtype=np.float32)
        den_w_xcorr_features = np.clip(den_w_xcorr_features, -10, 10)

        if not np.all(np.isfinite(den_w_phase_features)) or not np.all(np.isfinite(den_w_xcorr_features)):
            skip_reasons['nonfinite_phase_or_xcorr_features'] += 1
            n_bad += 1
            continue

        EXCLUDE_FIELDS = {'params_saved'}
        mat_out = {k: v for k, v in mat.items()
                   if not k.startswith('__') and k not in EXCLUDE_FIELDS}
        mat_out['den_fit_rmse_x'] = np.float32(rmse_x[local_i])
        mat_out['den_fit_rmse_y'] = np.float32(rmse_y[local_i])
        mat_out['den_energy_x'] = np.float32(energy_x[local_i])
        mat_out['den_energy_y'] = np.float32(energy_y[local_i])
        mat_out['den_w_phase_features'] = den_w_phase_features.reshape(1, -1)
        mat_out['den_w_xcorr_features'] = den_w_xcorr_features.reshape(1, -1)
        mat_out[ANNEX_FLAG] = True

        scipy.io.savemat(filepaths[orig_i], mat_out)
        n_written += 1

    return n_written, n_bad, skip_reasons


if __name__ == '__main__':
    print(f'Using device: {DEVICE}')
    groups = group_files_by_system(DATA_ROOT)
    for sysname, files in groups.items():
        print(f'  {sysname}: {len(files)} unannexed files')
    if sum(len(v) for v in groups.values()) == 0:
        raise RuntimeError('No unannexed example_*.mat files found — check DATA_ROOT.')

    total_written = total_bad = 0
    all_skip_reasons = defaultdict(int)

    for sysname, files in groups.items():
        n_chunks = (len(files) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f'\n=== {sysname}: {len(files)} files, {n_chunks} chunks ===')
        for c in range(n_chunks):
            chunk = files[c * CHUNK_SIZE:(c + 1) * CHUNK_SIZE]
            print(f'  --- chunk {c + 1}/{n_chunks} ({len(chunk)} files) ---')
            skip_reasons = defaultdict(int)
            written, bad, skip_reasons = fit_chunk(sysname, chunk, skip_reasons)
            total_written += written
            total_bad += bad
            print(f'    written={written}  bad/skipped={bad}')
            for reason, count in sorted(skip_reasons.items(), key=lambda kv: -kv[1])[:5]:
                print(f'      [{count:5d}x] {reason}')
                all_skip_reasons[reason] += count

    print(f'\n=== DONE ===')
    print(f'Total written: {total_written}')
    print(f'Total bad/skipped: {total_bad}')
    if total_written == 0 and all_skip_reasons:
        print('\nTop skip reasons across the whole run:')
        for reason, count in sorted(all_skip_reasons.items(), key=lambda kv: -kv[1])[:10]:
            print(f'  [{count:6d}x] {reason}')
    print('\nV11 data generation complete: nullcline(24) + xcorr(5) + b1c(40) +')
    print('b4(40) + den_energy_x/y + topology. Ready for LTInetV11.py.')
