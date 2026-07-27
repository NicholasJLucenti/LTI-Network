import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt
from collections import Counter

# ================================================================
# DUAL-BRANCH, EQ-GATED ARCHITECTURE
#
# Diagnosis chain that led here:
#   1. Pooled Eq0+Eq1 training: Latent 51%, Coupling 76% (vs V8's 83%/92%)
#   2. EQ_LOSS_WEIGHT=0 ablation: same result -> not gradient interference
#   3. Isolated Eq0-only / Eq1-only ablations: both recover to ~80-91%,
#      matching V8 -> the y-equation regime is NOT intrinsically harder
#   4. Option A (head-level conditioning on PREDICTED eq): no recovery
#      (~52%/76%), despite head_eq itself hitting ~92% almost immediately
#   5. Ground-truth-eq head-level conditioning (oracle, upper bound test):
#      STILL no recovery (~54%/78%), fully converged over 440 epochs
#
# (4) and (5) together prove the failure is not about eq signal quality
# or where in training it becomes available — it's that head-level
# conditioning (concatenation after the shared trunk) cannot recover
# separability that the shared branches never encoded. Before any
# eq-conditioning existed, head_latent/head_coupling only ever saw ONE
# decision boundary, so branch1/1b/1c/2 had no incentive to preserve
# eq-specific latent/coupling-discriminating detail — they converged on
# a compromise representation averaged across both regimes. No amount of
# downstream conditioning, however accurate, can reconstruct detail that
# was never encoded upstream.
#
# FIX: duplicate the four eq-sensitive branches (branch1, branch1b,
# branch1c, branch2) into two full specialized stacks — one free to
# specialize for Eq0-shaped data, one for Eq1-shaped data — and combine
# their outputs via a soft mixture weighted by predicted eq probability,
# computed from BOTH stacks' raw output (not a compressed single-vector
# afterthought). branch3 (topology) stays shared/undupli-cated since
# topology classification has no principled dependency on which
# equation absorbed the coupling term. Each stack should behave close to
# the already-validated single-eq specialists (~80-91%) since it never
# has to compromise with the other regime's data — the gate, not the
# classification heads, is what has to handle the two-regime split, and
# gating on eq has already been shown to be reliably learnable (~90%+
# eq accuracy from epoch ~20 in every prior ablation).
# ================================================================

# ================================================================
# SHARPENED-GATE + WIDENED-BRANCH1C VARIANT
#
# Follow-up to the dual-branch result (Latent 67.3%, Coupling 84.2%,
# Eq 93.8% val) — three targeted changes based on that run's diagnosis:
#   1. Entropy penalty on the gate's eq_probs, encouraging confident
#      (near-one-hot) gating once it's reasonably sure. A 93-94%
#      accurate-but-soft gate still blends ~6-7% of the "wrong" stack's
#      representation into core_gated for many examples; pushing the
#      gate toward sharper decisions should reduce that residual
#      cross-stack contamination without changing what the gate is
#      allowed to decide.
#   2. Widened branch1c (Branch1c_AllNewMLP) — it processes the largest,
#      most information-dense feature block (110-dim: RQA, wavelet,
#      phase-residuals, SINDy coefficients) and is the most plausible
#      branch under-provisioned to resolve the L_gk-attractor and
#      L_bistable/L_delay confusion clusters identified in the
#      confusion matrix. VRAM/param footprint was trivial (~185K params
#      on a 6GB card), so there's real headroom to widen it cheaply.
#   3. N_EPOCHS/ES_PATIENCE promoted to top-level config and raised —
#      train/val curves hadn't clearly plateaued at epoch 500 (no
#      overfitting gap anywhere), so the run may simply have been cut
#      short of its actual ceiling.
# ================================================================

# ================================================================
# NO-OVERSHOOT VARIANT — L_overshoot removed from the latent library
#
# Rationale (both numerical and biological):
#   Numerically, L_overshoot had the lowest recall of the six real
#   classes (69.1%) in the sharpened dual-branch run, and unlike the
#   L_bistable/L_delay confusion pair (a clean, interpretable mutual
#   ambiguity between two well-grounded neighbors), its errors were
#   diffusely scattered across four+ other classes — the signature of
#   a class with no clean signature at all, not a class with a
#   well-defined but confusable one.
#   Biologically, "overshoot/adaptation" (dI/dt = kp*x - kp2*x^2 - kd*I)
#   does not actually implement genuine biochemical adaptation by the
#   field's own formal definition (Ma, Trusina, El-Samad, Lim & Tang,
#   Cell 2009): true adaptation requires the steady state to be
#   independent of sustained input magnitude, needing a minimum
#   three-node topology (negative feedback + buffer node, or incoherent
#   feedforward + proportioner node). This single-ODE form's steady
#   state I_ss(x) = (kp*x - kp2*x^2)/kd is a fixed function OF x, not an
#   x-independent baseline — it cannot adapt. It's structurally closer
#   to Haldane-type substrate-inhibition enzyme kinetics (describing a
#   reaction rate, not a regulator's own abundance dynamics), a much
#   narrower and less-established template than the other five classes,
#   which each map onto canonical, ubiquitous regulatory motifs
#   (miRNA/ceRNA titration, Goldbeter-Koshland ultrasensitivity, Hill
#   repression, positive-autoregulation bistability, delay-relay).
#
# L_overshoot's data is EXCLUDED AT THE FOLDER LEVEL — its files are
# never listed in data_dirs and therefore never read from disk, not
# merely filtered out after loading. NULL's raw label (6 in the
# original 7-class scheme) is remapped to 5 to keep the label space
# compact (0-5, six classes total) rather than leaving a gap at index 5.
# ================================================================

# Config
N_LATENT_CLASSES   = 6    # L_mirna, L_gk, L_yhill, L_bistable, L_delay, NULL (L_overshoot removed)
N_COUPLING_CLASSES = 4    # Ch0-2 + Ch3 null (unaffected by latent removal)
N_EQ_CLASSES       = 2    # Eq0 (coupling -> dx), Eq1 (coupling -> dy). Undefined/masked for NULL.
NULLCLINE_DIM      = 49
XCORR_DIM          = 10
ALL_NEW_DIM        = 88
V6_NEW_DIM         = 12   # FNN(4) + TransferEntropy(4) + ResidualSpectrum(4)
FP_MULTI_DIM       = 10   # multi-fixed-point geometry (nonzero only for toggle switch)
BRANCH1C_DIM       = ALL_NEW_DIM + V6_NEW_DIM + FP_MULTI_DIM  # 110
COUPLING_LOSS_WEIGHT = 1.0
EQ_LOSS_WEIGHT       = 1.0

# Per-branch output dims (each duplicated into an "_a"/"_b" stack).
# BRANCH1C_OUT widened 32 -> 48 (~50% larger), with proportionally wider
# hidden layers inside Branch1c_AllNewMLP itself (see class below).
BRANCH1_OUT   = 48
BRANCH1B_OUT  = 16
BRANCH1C_OUT  = 48   # was 32
BRANCH2_OUT   = 24
TOPO_OUT      = 8    # branch3, shared/undupli-cated
CORE_DIM      = BRANCH1_OUT + BRANCH1B_OUT + BRANCH1C_OUT + BRANCH2_OUT  # 136 (was 120)
MERGE_DIM     = CORE_DIM + TOPO_OUT                                      # 144 (was 128)

# Entropy penalty on the gate's softmax(eq_logits), added to the total
# loss during training only (not backpropagated during eval, though the
# raw entropy value is still computed there for monitoring). Encourages
# eq_probs toward one-hot once the gate is reasonably confident, without
# changing what it's allowed to predict. Max possible entropy for a
# 2-class softmax is ln(2)=0.693, so start modest (this is a nudge, not
# a hard constraint) and increase if the gate still looks soft in the
# printed per-epoch entropy value.
ENTROPY_PENALTY_WEIGHT = 0.05

# Cap on how many files are loaded per (latent, coupling, eq, system)
# folder (and per-system for NULL). Tune directly; this is the full
# pooled dataset (Eq0+Eq1+NULL), so start generous and reduce if
# overfitting appears, same as in the other V9 scripts.
MAX_FILES_PER_CONFIG = 300

# Training budget. Raised from 500/60 -> both Latent and Coupling
# accuracy were still climbing (no train/val gap) at epoch 500 in the
# prior run, suggesting the ceiling hadn't actually been reached yet.
N_EPOCHS    = 900
ES_PATIENCE = 120

topology_map = {
    'LIMIT CYCLE': 0, 'DAMPED OSCILLATION': 1,
    'STEADY STATE': 2, 'UNDETERMINED': 3
}

DATA_ROOT   = r'C:/Users/nickj/LTInetV9 Local Data Drive'
WEIGHTS_DIR = DATA_ROOT

SYSTEMS  = ['goodwin', 'brusselator', 'repressilator', 'van_der_pol',
            'fitzhugh_nagumo', 'rosenzweig_macarthur', 'toggle_switch']
N_SYS    = len(SYSTEMS)

# V9 latent library MINUS L_overshoot (5 real classes) x 4 coupling
# channels x 2 coupling-equation targets. NULL class has no Eq split.
# latent_names is index-aligned to the REMAPPED compact label space
# (0-5), not the original generator's raw structure_label (0-6).
latent_names   = ['L_mirna', 'L_gk', 'L_yhill', 'L_bistable', 'L_delay', 'NULL']
coupling_names = ['Ch0 HillRep', 'Ch1 Additive', 'Ch2 Multiplicative', 'Ch3 None']
eq_names       = ['Eq0 -> dx', 'Eq1 -> dy']

# Raw structure_label values as saved by V9_ProductionGen.m (0=L_mirna,
# 1=L_gk, 2=L_yhill, 3=L_bistable, 4=L_delay, 5=L_overshoot, 6=NULL) ->
# compact 0-5 label space used by this model. 5 (L_overshoot) is
# deliberately absent from this map — any file with that raw label
# would be a bug (its folders are never in data_dirs to begin with) and
# is defensively rejected at load time rather than silently remapped.
ACTIVE_RAW_LATENTS = [0, 1, 2, 3, 4]   # excludes 5 (L_overshoot)
RAW_TO_COMPACT_LABEL = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5}  # raw NULL(6) -> compact 5

data_dirs = []
sys_idx_lookup = {}
for sys_i in range(1, N_SYS + 1):
    for lat in ACTIVE_RAW_LATENTS:   # L_mirna, L_gk, L_yhill, L_bistable, L_delay — NOT L_overshoot
        for ch in range(3):     # Ch0-2
            for eq in range(2):  # Eq0, Eq1
                d = f'{DATA_ROOT}/V9Lat{lat}_Ch{ch}_Eq{eq}_Sys{sys_i}_NNdata'
                data_dirs.append(d)
                sys_idx_lookup[d] = sys_i - 1
    d_null = f'{DATA_ROOT}/V9Lat6_Ch3_Sys{sys_i}_NNdata'
    data_dirs.append(d_null)
    sys_idx_lookup[d_null] = sys_i - 1


class LTInetV9Dataset(Dataset):
    def __init__(self, data_dirs, sys_idx_lookup, nullcline_dim):
        self.samples = []
        skipped_no_anf = 0
        skipped_no_xcorr = 0
        skipped_no_v6new = 0

        for d in data_dirs:
            if not os.path.exists(d):
                print(f'  [MISSING] {d}')
                continue
            files = sorted([os.path.join(d, f)
                             for f in os.listdir(d) if f.endswith('.mat')])
            files = files[:MAX_FILES_PER_CONFIG]
            sys_id = sys_idx_lookup[d]

            for fp in files:
                try:
                    mat = scipy.io.loadmat(fp)
                    if 'structure_label' not in mat or 'coupling_label' not in mat:
                        continue
                    sl_raw = int(mat['structure_label'].squeeze())
                    cl = int(mat['coupling_label'].squeeze())

                    # Remap raw generator label -> compact 0-5 label space.
                    # sl_raw==5 (L_overshoot) should be structurally
                    # impossible here since its folders are never in
                    # data_dirs, but reject it defensively rather than
                    # silently mis-mapping if a stray file somehow exists.
                    if sl_raw not in RAW_TO_COMPACT_LABEL:
                        continue
                    sl = RAW_TO_COMPACT_LABEL[sl_raw]

                    if not (0 <= sl < N_LATENT_CLASSES):
                        continue
                    if not (0 <= cl < N_COUPLING_CLASSES):
                        continue

                    # coupling_eq: -1 (or absent, for older files) means
                    # undefined/NULL -> masked out of the eq loss.
                    if 'coupling_eq' in mat:
                        eq_raw = int(mat['coupling_eq'].squeeze())
                    else:
                        eq_raw = -1
                    eq_valid = 0 <= eq_raw < N_EQ_CLASSES
                    eq = eq_raw if eq_valid else 0  # placeholder value, ignored via mask
                    eq_mask = 1.0 if eq_valid else 0.0

                    if 'xcorr_features' not in mat:
                        skipped_no_xcorr += 1; continue
                    if 'all_new_features' not in mat:
                        skipped_no_anf += 1; continue
                    if 'v6_new_features' not in mat:
                        skipped_no_v6new += 1; continue

                    nc = np.clip(mat['nullcline_features'].squeeze(), -50., 50.).astype(np.float32)
                    if nc.shape[0] != nullcline_dim or not np.all(np.isfinite(nc)):
                        continue

                    xc = np.clip(mat['xcorr_features'].squeeze(), -50., 50.).astype(np.float32)
                    if xc.shape[0] != XCORR_DIM or not np.all(np.isfinite(xc)):
                        continue

                    anf = np.clip(mat['all_new_features'].squeeze(), -50., 50.).astype(np.float32)
                    if anf.shape[0] != ALL_NEW_DIM or not np.all(np.isfinite(anf)):
                        continue

                    v6n = np.clip(mat['v6_new_features'].squeeze(), -50., 50.).astype(np.float32)
                    if v6n.shape[0] != V6_NEW_DIM or not np.all(np.isfinite(v6n)):
                        continue

                    if 'fp_multi_features' in mat:
                        fpm = np.clip(mat['fp_multi_features'].squeeze(), -50., 50.).astype(np.float32)
                        if fpm.shape[0] != FP_MULTI_DIM or not np.all(np.isfinite(fpm)):
                            fpm = np.zeros(FP_MULTI_DIM, dtype=np.float32)
                    else:
                        fpm = np.zeros(FP_MULTI_DIM, dtype=np.float32)

                    anf_full = np.concatenate([anf, v6n, fpm])  # 88+12+10=110

                    xi = mat['Xi_ternary'].astype(np.float32)
                    if xi.shape[0] < 9:
                        xi = np.vstack([xi, np.zeros((9-xi.shape[0], 2), dtype=np.float32)])
                    elif xi.shape[0] > 9:
                        xi = xi[:9, :]

                    topo = topology_map.get(str(mat['topology'][0]).strip(), 3)
                    system_name = str(mat['system_name'][0]).strip() if 'system_name' in mat else SYSTEMS[sys_id]

                    self.samples.append((nc, xc, anf_full, xi, topo, sl, cl, eq, eq_mask, sys_id, system_name))

                except Exception:
                    continue

        if skipped_no_xcorr > 0:
            print(f'  [INFO] {skipped_no_xcorr} files missing xcorr_features.')
        if skipped_no_anf > 0:
            print(f'  [INFO] {skipped_no_anf} files missing all_new_features - run annexation first.')
        if skipped_no_v6new > 0:
            print(f'  [INFO] {skipped_no_v6new} files missing v6_new_features.')
        print(f'Loaded {len(self.samples)} valid examples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        nc, xc, anf_full, xi, topo, sl, cl, eq, eq_mask, sys_id, system_name = self.samples[idx]
        return (torch.tensor(nc), torch.tensor(xc), torch.tensor(anf_full),
                torch.tensor(xi), torch.tensor(topo, dtype=torch.long),
                torch.tensor(sl, dtype=torch.long),
                torch.tensor(cl, dtype=torch.long),
                torch.tensor(eq, dtype=torch.long),
                torch.tensor(eq_mask, dtype=torch.float32),
                sys_id, system_name)


class PreloadedTensorDataset(Dataset):
    """
    Same VRAM-preload approach as LTInetV8.py: dataset is small enough
    (well under the RTX PRO 500's 6GB) to stack into a few large tensors
    and move them to the GPU once, avoiding per-batch transfer entirely.
    """
    def __init__(self, dataset, device):
        n = len(dataset)
        nc_all = torch.empty((n, NULLCLINE_DIM), dtype=torch.float32)
        xc_all = torch.empty((n, XCORR_DIM), dtype=torch.float32)
        anf_all = torch.empty((n, BRANCH1C_DIM), dtype=torch.float32)
        xi_all = torch.empty((n, 9, 2), dtype=torch.float32)
        topo_all = torch.empty(n, dtype=torch.long)
        sl_all = torch.empty(n, dtype=torch.long)
        cl_all = torch.empty(n, dtype=torch.long)
        eq_all = torch.empty(n, dtype=torch.long)
        eqmask_all = torch.empty(n, dtype=torch.float32)

        sys_ids = []
        sys_names = []

        for i in range(n):
            nc, xc, anf, xi, topo, sl, cl, eq, eq_mask, sys_id, sys_name = dataset[i]
            nc_all[i] = nc; xc_all[i] = xc; anf_all[i] = anf
            xi_all[i] = xi; topo_all[i] = topo; sl_all[i] = sl; cl_all[i] = cl
            eq_all[i] = eq; eqmask_all[i] = eq_mask
            sys_ids.append(sys_id); sys_names.append(sys_name)

        self.nc = nc_all.to(device)
        self.xc = xc_all.to(device)
        self.anf = anf_all.to(device)
        self.xi = xi_all.to(device)
        self.topo = topo_all.to(device)
        self.sl = sl_all.to(device)
        self.cl = cl_all.to(device)
        self.eq = eq_all.to(device)
        self.eqmask = eqmask_all.to(device)
        self.sys_ids = sys_ids
        self.sys_names = sys_names
        self.n = n
        self.device = device

    def __len__(self):
        return self.n

    def get_batch(self, indices):
        idx_cpu = indices.cpu().numpy()
        return (self.nc[indices], self.xc[indices], self.anf[indices],
                self.xi[indices], self.topo[indices], self.sl[indices],
                self.cl[indices], self.eq[indices], self.eqmask[indices],
                [self.sys_ids[i] for i in idx_cpu],
                [self.sys_names[i] for i in idx_cpu])


def iterate_batches(preloaded, batch_size, shuffle, device):
    n = len(preloaded)
    if shuffle:
        perm = torch.randperm(n, device=device)
    else:
        perm = torch.arange(n, device=device)
    for start in range(0, n, batch_size):
        idx = perm[start:start+batch_size]
        yield preloaded.get_batch(idx)


class Branch1_NullclineMLP(nn.Module):
    def __init__(self, input_dim=49, output_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 96), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(96, output_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Branch1b_XcorrMLP(nn.Module):
    def __init__(self, input_dim=10, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, output_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Branch1c_AllNewMLP(nn.Module):
    # Widened: 160/80 -> 224/112 hidden, output 32 -> 48 (BRANCH1C_OUT).
    # This is the branch processing the largest, most information-dense
    # feature block (RQA, wavelet, phase-residuals, SINDy coefficients),
    # and the most plausible candidate for under-provisioned capacity
    # given the L_gk-attractor and L_bistable/L_delay confusion clusters.
    def __init__(self, input_dim=110, output_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 224), nn.BatchNorm1d(224), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(224, 112), nn.BatchNorm1d(112), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(112, output_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Branch2_TermAttention(nn.Module):
    def __init__(self, n_terms=9, n_equations=2, embed_dim=32, output_dim=24):
        super().__init__()
        self.asym_proj = nn.Sequential(nn.Linear(n_terms, 16), nn.ReLU(), nn.Linear(16, 8))
        self.embed = nn.Linear(n_terms, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(n_equations*embed_dim+8, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, output_dim), nn.ReLU())

    def forward(self, x):
        if x.shape[-1] != 2:
            x = x.transpose(-1, -2)
        x = x.reshape(x.shape[0], 9, 2)
        asym = self.asym_proj(x[:, :, 0] - x[:, :, 1])
        tokens = self.embed(x.permute(0, 2, 1))
        attn_out, _ = self.attention(tokens, tokens, tokens)
        flat = attn_out.reshape(attn_out.shape[0], -1)
        return self.projection(torch.cat([flat, asym], dim=1))


class Branch3_TopologyEmbed(nn.Module):
    def __init__(self, n_classes=4, embed_dim=8, output_dim=8):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.proj = nn.Sequential(nn.Linear(embed_dim, output_dim), nn.ReLU())

    def forward(self, x):
        return self.proj(self.embed(x))


class LTInetV9(nn.Module):
    """
    Dual-branch, eq-gated architecture (see header for full diagnosis).
    This variant widens branch1c (32->48 out) versus the original
    dual-branch script, so core_dim/merge_dim grew accordingly.

    core_dim = 48+16+48+24 = 136 (branch1+1b+1c+2 output, per stack)
    topo_dim = 8 (branch3, SHARED — not duplicated)
    merge_dim = 136 + 8 = 144

    Two full copies of branch1/1b/1c/2 ("_a" for Eq0-hypothesis,
    "_b" for Eq1-hypothesis) each produce a CORE_DIM-dim core vector from
    the SAME raw inputs. head_eq sees both cores concatenated plus
    topology — full information from both hypotheses, not a compressed
    single-vector summary — and predicts eq. Its softmax output
    soft-gates (weighted sum, not hard selection) between core_a and
    core_b, so both stacks receive gradient every batch in proportion to
    how much the gate trusts them for that example. An entropy penalty
    on eq_probs (applied in the training loop, not here) pushes this
    gate toward sharper, more one-hot decisions once it's confident,
    reducing residual cross-stack contamination in core_gated.

    head_latent/head_coupling operate on the gated MERGE_DIM-dim merge.
    """
    def __init__(self):
        super().__init__()
        # Stack A: specializes for Eq0-shaped (coupling -> dx) data
        self.branch1_a = Branch1_NullclineMLP(49, BRANCH1_OUT)
        self.branch1b_a = Branch1b_XcorrMLP(10, BRANCH1B_OUT)
        self.branch1c_a = Branch1c_AllNewMLP(BRANCH1C_DIM, BRANCH1C_OUT)
        self.branch2_a = Branch2_TermAttention(9, output_dim=BRANCH2_OUT)

        # Stack B: specializes for Eq1-shaped (coupling -> dy) data
        self.branch1_b = Branch1_NullclineMLP(49, BRANCH1_OUT)
        self.branch1b_b = Branch1b_XcorrMLP(10, BRANCH1B_OUT)
        self.branch1c_b = Branch1c_AllNewMLP(BRANCH1C_DIM, BRANCH1C_OUT)
        self.branch2_b = Branch2_TermAttention(9, output_dim=BRANCH2_OUT)

        # Shared, undupli-cated: topology has no principled dependency
        # on which equation absorbed the coupling term.
        self.branch3 = Branch3_TopologyEmbed(4, output_dim=TOPO_OUT)

        # Sees BOTH stacks' full CORE_DIM-dim cores plus topology — full
        # information from both hypotheses, not a bottlenecked summary,
        # since accurate gating is the one thing this whole design
        # depends on.
        self.head_eq = nn.Sequential(
            nn.Linear(CORE_DIM + CORE_DIM + TOPO_OUT, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, N_EQ_CLASSES))

        self.head_latent = nn.Sequential(
            nn.Linear(MERGE_DIM, 96), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(96, 48), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(48, N_LATENT_CLASSES))
        self.head_coupling = nn.Sequential(
            nn.Linear(MERGE_DIM, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, N_COUPLING_CLASSES))

    def forward(self, nc, xc, anf, xi, topo):
        core_a = torch.cat([self.branch1_a(nc), self.branch1b_a(xc),
                             self.branch1c_a(anf), self.branch2_a(xi)], dim=1)
        core_b = torch.cat([self.branch1_b(nc), self.branch1b_b(xc),
                             self.branch1c_b(anf), self.branch2_b(xi)], dim=1)
        topo_vec = self.branch3(topo)

        eq_logits = self.head_eq(torch.cat([core_a, core_b, topo_vec], dim=1))
        eq_probs = torch.softmax(eq_logits, dim=1)  # (B, 2)

        # Soft mixture: weight_a*core_a + weight_b*core_b. Gradient to
        # each stack scales with how much the gate currently trusts it
        # for that example, rather than a hard, non-differentiable pick.
        w_a = eq_probs[:, 0:1]
        w_b = eq_probs[:, 1:2]
        core_gated = w_a * core_a + w_b * core_b

        merged = torch.cat([core_gated, topo_vec], dim=1)
        lat_logits = self.head_latent(merged)
        coup_logits = self.head_coupling(merged)

        return lat_logits, coup_logits, eq_logits


def masked_eq_loss(eq_logits, eq_labels, eq_mask, loss_fn_eq):
    """
    Cross-entropy over Eq0/Eq1, computed only for examples where
    eq_mask==1 (i.e. real coupling classes with a defined equation
    target). NULL-class examples (eq_mask==0) contribute zero loss and
    zero gradient through this head, rather than being taught a fake
    fixed label.
    """
    if eq_mask.sum() < 1:
        return torch.tensor(0.0, device=eq_logits.device), torch.tensor(0.0, device=eq_logits.device), torch.tensor(0.0, device=eq_logits.device)
    per_example = loss_fn_eq(eq_logits, eq_labels)  # reduction='none' expected
    loss = (per_example * eq_mask).sum() / eq_mask.sum()
    pred = eq_logits.argmax(1)
    correct = ((pred == eq_labels).float() * eq_mask).sum()
    n_valid = eq_mask.sum()
    return loss, correct, n_valid


def eq_gate_entropy(eq_logits):
    """
    Mean entropy of softmax(eq_logits) across the batch, applied to ALL
    examples (real and NULL alike) — the gate has to produce SOME
    mixture weight for every example regardless of whether eq is
    meaningfully defined for it, so sharpening applies uniformly.
    Max possible value for a 2-class softmax is ln(2)=0.693 (fully
    uncertain); near 0 means the gate is close to one-hot.
    """
    probs = torch.softmax(eq_logits, dim=1)
    ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return ent.mean()


def train_epoch(model, preloaded_train, optimizer, loss_fn_lat, loss_fn_coup, loss_fn_eq, device,
                 nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std, batch_size=64):
    model.train()
    total_loss = lat_correct = coup_correct = total = 0
    eq_correct_sum = 0.0; eq_valid_sum = 0.0
    entropy_sum = 0.0; n_batches_seen = 0
    for nc, xc, anf, xi, topo, slbl, clbl, eqlbl, eqmask, _, _ in iterate_batches(preloaded_train, batch_size, True, device):
        if nc.shape[0] < 2:
            continue
        nc_n = (nc - nc_mean) / nc_std
        xc_n = (xc - xc_mean) / xc_std
        anf_n = (anf - anf_mean) / anf_std
        optimizer.zero_grad()
        ll, lc, le = model(nc_n, xc_n, anf_n, xi, topo)
        eq_loss, eq_corr, eq_n = masked_eq_loss(le, eqlbl, eqmask, loss_fn_eq)
        gate_entropy = eq_gate_entropy(le)
        loss = (loss_fn_lat(ll, slbl) + COUPLING_LOSS_WEIGHT * loss_fn_coup(lc, clbl)
                + EQ_LOSS_WEIGHT * eq_loss + ENTROPY_PENALTY_WEIGHT * gate_entropy)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        lat_correct += (ll.argmax(1) == slbl).sum().item()
        coup_correct += (lc.argmax(1) == clbl).sum().item()
        eq_correct_sum += eq_corr.item()
        eq_valid_sum += eq_n.item()
        entropy_sum += gate_entropy.item()
        n_batches_seen += 1
        total += slbl.size(0)
    n_batches = max((len(preloaded_train) + batch_size - 1) // batch_size, 1)
    eq_acc = eq_correct_sum / eq_valid_sum if eq_valid_sum > 0 else 0.0
    mean_entropy = entropy_sum / max(n_batches_seen, 1)
    return total_loss / n_batches, lat_correct / max(total, 1), coup_correct / max(total, 1), eq_acc, mean_entropy


def evaluate(model, preloaded_val, loss_fn_lat, loss_fn_coup, loss_fn_eq, device,
             nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std, batch_size=64):
    model.eval()
    total_loss = lat_correct = coup_correct = total = 0
    eq_correct_sum = 0.0; eq_valid_sum = 0.0
    entropy_sum = 0.0; n_batches_seen = 0
    with torch.no_grad():
        for nc, xc, anf, xi, topo, slbl, clbl, eqlbl, eqmask, _, _ in iterate_batches(preloaded_val, batch_size, False, device):
            nc_n = (nc - nc_mean) / nc_std
            xc_n = (xc - xc_mean) / xc_std
            anf_n = (anf - anf_mean) / anf_std
            ll, lc, le = model(nc_n, xc_n, anf_n, xi, topo)
            eq_loss, eq_corr, eq_n = masked_eq_loss(le, eqlbl, eqmask, loss_fn_eq)
            gate_entropy = eq_gate_entropy(le)
            loss = (loss_fn_lat(ll, slbl) + COUPLING_LOSS_WEIGHT * loss_fn_coup(lc, clbl)
                    + EQ_LOSS_WEIGHT * eq_loss + ENTROPY_PENALTY_WEIGHT * gate_entropy)
            total_loss += loss.item()
            lat_correct += (ll.argmax(1) == slbl).sum().item()
            coup_correct += (lc.argmax(1) == clbl).sum().item()
            eq_correct_sum += eq_corr.item()
            eq_valid_sum += eq_n.item()
            entropy_sum += gate_entropy.item()
            n_batches_seen += 1
            total += slbl.size(0)
    n_batches = max((len(preloaded_val) + batch_size - 1) // batch_size, 1)
    eq_acc = eq_correct_sum / eq_valid_sum if eq_valid_sum > 0 else 0.0
    mean_entropy = entropy_sum / max(n_batches_seen, 1)
    return total_loss / n_batches, lat_correct / max(total, 1), coup_correct / max(total, 1), eq_acc, mean_entropy


def evaluate_per_system(model, preloaded_val, device,
                         nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std, batch_size=64):
    model.eval()
    sys_correct_lat = Counter(); sys_correct_coup = Counter()
    sys_correct_eq = Counter(); sys_valid_eq = Counter(); sys_total = Counter()
    with torch.no_grad():
        for nc, xc, anf, xi, topo, slbl, clbl, eqlbl, eqmask, sys_id, sys_name in iterate_batches(preloaded_val, batch_size, False, device):
            nc_n = (nc - nc_mean) / nc_std
            xc_n = (xc - xc_mean) / xc_std
            anf_n = (anf - anf_mean) / anf_std
            ll, lc, le = model(nc_n, xc_n, anf_n, xi, topo)
            lat_pred = ll.argmax(1).cpu(); coup_pred = lc.argmax(1).cpu(); eq_pred = le.argmax(1).cpu()
            slbl_cpu = slbl.cpu(); clbl_cpu = clbl.cpu(); eqlbl_cpu = eqlbl.cpu(); eqmask_cpu = eqmask.cpu()
            for i, name in enumerate(sys_name):
                sys_total[name] += 1
                if lat_pred[i] == slbl_cpu[i]:
                    sys_correct_lat[name] += 1
                if coup_pred[i] == clbl_cpu[i]:
                    sys_correct_coup[name] += 1
                if eqmask_cpu[i] > 0:
                    sys_valid_eq[name] += 1
                    if eq_pred[i] == eqlbl_cpu[i]:
                        sys_correct_eq[name] += 1
    print('\nPer-system validation accuracy:')
    for name in sorted(sys_total.keys()):
        n = sys_total[name]
        n_eq = max(sys_valid_eq[name], 1)
        print(f'  {name:<24} n={n:4d}  Lat={sys_correct_lat[name]/n:.3f}  '
              f'Coup={sys_correct_coup[name]/n:.3f}  Eq={sys_correct_eq[name]/n_eq:.3f} (n_eq={sys_valid_eq[name]})')


if __name__ == '__main__':

    dataset = LTInetV9Dataset(data_dirs, sys_idx_lookup, NULLCLINE_DIM)
    if len(dataset) == 0:
        raise RuntimeError('No valid examples found. Check data_dirs and run annexation first.')

    struct_counts = Counter(s[5] for s in dataset.samples)
    coupling_counts = Counter(s[6] for s in dataset.samples)
    eq_counts = Counter(s[7] for s in dataset.samples if s[8] > 0)  # only valid-eq examples
    system_counts = Counter(s[10] for s in dataset.samples)

    print(f'\nTotal examples: {len(dataset)}')
    print(f'L_overshoot EXCLUDED — {len(ACTIVE_RAW_LATENTS)} real latent classes + NULL = '
          f'{N_LATENT_CLASSES} total. Folders never listed: '
          f'V9Lat5_Ch*_Eq*_Sys*_NNdata (all systems/couplings/eqs).')
    print(f'MAX_FILES_PER_CONFIG={MAX_FILES_PER_CONFIG}  N_EPOCHS={N_EPOCHS}  '
          f'ES_PATIENCE={ES_PATIENCE}  ENTROPY_PENALTY_WEIGHT={ENTROPY_PENALTY_WEIGHT}  '
          f'BRANCH1C_OUT={BRANCH1C_OUT}  CORE_DIM={CORE_DIM}  MERGE_DIM={MERGE_DIM}')
    print(f'MAX_FILES_PER_CONFIG = {MAX_FILES_PER_CONFIG}')
    print('Latent distribution:')
    for i in range(N_LATENT_CLASSES):
        print(f'  {latent_names[i]}: {struct_counts.get(i, 0)}')
    print('Coupling distribution:')
    for i in range(N_COUPLING_CLASSES):
        print(f'  {coupling_names[i]}: {coupling_counts.get(i, 0)}')
    print('Coupling-equation distribution (excludes NULL, masked):')
    for i in range(N_EQ_CLASSES):
        print(f'  {eq_names[i]}: {eq_counts.get(i, 0)}')
    print('System distribution:')
    for name in sorted(system_counts.keys()):
        print(f'  {name}: {system_counts[name]}')

    total_lat = sum(struct_counts.values())
    lat_weights = torch.tensor(
        [total_lat / (N_LATENT_CLASSES * max(struct_counts.get(i, 1), 1))
         for i in range(N_LATENT_CLASSES)], dtype=torch.float32)
    lat_weights = lat_weights / lat_weights.mean()

    total_coup = sum(coupling_counts.values())
    coup_weights = torch.tensor(
        [total_coup / (N_COUPLING_CLASSES * max(coupling_counts.get(i, 1), 1))
         for i in range(N_COUPLING_CLASSES)], dtype=torch.float32)
    coup_weights = coup_weights / coup_weights.mean()

    total_eq = sum(eq_counts.values())
    eq_weights = torch.tensor(
        [total_eq / (N_EQ_CLASSES * max(eq_counts.get(i, 1), 1))
         for i in range(N_EQ_CLASSES)], dtype=torch.float32)
    eq_weights = eq_weights / eq_weights.mean() if total_eq > 0 else torch.ones(N_EQ_CLASSES)

    print(f'\nLatent class weights:   {lat_weights.numpy().round(3)}')
    print(f'Coupling class weights: {coup_weights.numpy().round(3)}')
    print(f'Eq class weights:       {eq_weights.numpy().round(3)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    all_nc = np.stack([s[0] for s in dataset.samples])
    all_xc = np.stack([s[1] for s in dataset.samples])
    all_anf = np.stack([s[2] for s in dataset.samples])

    def make_norm(arr):
        mean = torch.tensor(arr.mean(0), dtype=torch.float32)
        std = torch.tensor(arr.std(0), dtype=torch.float32)
        std[std < 1e-6] = 1.0
        return mean, std

    nc_mean, nc_std = make_norm(all_nc)
    xc_mean, xc_std = make_norm(all_xc)
    anf_mean, anf_std = make_norm(all_anf)
    nc_mean, nc_std = nc_mean.to(device), nc_std.to(device)
    xc_mean, xc_std = xc_mean.to(device), xc_std.to(device)
    anf_mean, anf_std = anf_mean.to(device), anf_std.to(device)

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f'Train: {n_train}  Val: {n_val}')

    print('\nPreloading dataset into VRAM...')
    if device.type == 'cuda':
        t_preload_start = torch.cuda.Event(enable_timing=True)
        t_preload_end = torch.cuda.Event(enable_timing=True)
        t_preload_start.record()
    else:
        t_preload_start = None

    preloaded_train = PreloadedTensorDataset(train_set, device)
    preloaded_val = PreloadedTensorDataset(val_set, device)

    if device.type == 'cuda':
        t_preload_end.record()
        torch.cuda.synchronize()
        print(f'Preload complete in {t_preload_start.elapsed_time(t_preload_end)/1000:.1f}s')
        print(f'VRAM allocated after preload: {torch.cuda.memory_allocated(0)/1e6:.1f} MB')

    lat_weights = lat_weights.to(device)
    coup_weights = coup_weights.to(device)
    eq_weights = eq_weights.to(device)

    model = LTInetV9().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)
    loss_fn_lat = nn.CrossEntropyLoss(weight=lat_weights, label_smoothing=0.05)
    loss_fn_coup = nn.CrossEntropyLoss(weight=coup_weights, label_smoothing=0.05)
    # reduction='none' so masked_eq_loss can zero out NULL-class rows
    # before averaging, rather than letting them contribute at a fixed
    # (meaningless) label.
    loss_fn_eq = nn.CrossEntropyLoss(weight=eq_weights, label_smoothing=0.05, reduction='none')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params:,}')

    SAVE_PATH = f'{WEIGHTS_DIR}/LTInetV9_DualBranchEqGated_Sharpened_NoOvershoot_best.pth'
    best_avg_acc = 0.0
    n_epochs = N_EPOCHS
    es_patience = ES_PATIENCE
    no_improve = 0
    BATCH_SIZE = 64

    train_losses = []; val_losses = []
    train_lat_accs = []; val_lat_accs = []
    train_coup_accs = []; val_coup_accs = []
    train_eq_accs = []; val_eq_accs = []
    train_entropies = []; val_entropies = []

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_lat, tr_coup, tr_eq, tr_ent = train_epoch(
            model, preloaded_train, optimizer, loss_fn_lat, loss_fn_coup, loss_fn_eq, device,
            nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std, BATCH_SIZE)
        va_loss, va_lat, va_coup, va_eq, va_ent = evaluate(
            model, preloaded_val, loss_fn_lat, loss_fn_coup, loss_fn_eq, device,
            nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std, BATCH_SIZE)
        scheduler.step(va_loss)
        va_avg = (va_lat + va_coup + va_eq) / 3.0

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_lat_accs.append(tr_lat); val_lat_accs.append(va_lat)
        train_coup_accs.append(tr_coup); val_coup_accs.append(va_coup)
        train_eq_accs.append(tr_eq); val_eq_accs.append(va_eq)
        train_entropies.append(tr_ent); val_entropies.append(va_ent)

        if va_avg > best_avg_acc:
            best_avg_acc = va_avg
            no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            np.save(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_Sharpened_NoOvershoot_nc_mean.npy', nc_mean.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_Sharpened_NoOvershoot_nc_std.npy', nc_std.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_Sharpened_NoOvershoot_xc_mean.npy', xc_mean.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_Sharpened_NoOvershoot_xc_std.npy', xc_std.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_Sharpened_NoOvershoot_anf_mean.npy', anf_mean.cpu().numpy())
            np.save(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_Sharpened_NoOvershoot_anf_std.npy', anf_std.cpu().numpy())
        else:
            no_improve += 1

        print(f'Epoch {epoch:3d} | Loss {tr_loss:.4f}/{va_loss:.4f} | '
              f'Lat {tr_lat:.3f}/{va_lat:.3f} | '
              f'Coup {tr_coup:.3f}/{va_coup:.3f} | '
              f'Eq {tr_eq:.3f}/{va_eq:.3f} | '
              f'GateEnt {tr_ent:.3f}/{va_ent:.3f} (max {np.log(N_EQ_CLASSES):.3f}) | '
              f'Avg {va_avg:.3f} | LR {optimizer.param_groups[0]["lr"]:.1e}')

        if epoch % 20 == 0:
            ep = list(range(1, epoch + 1))
            fig, axes = plt.subplots(1, 5, figsize=(23, 4))
            axes[0].plot(ep, train_losses, 'b-', label='Train')
            axes[0].plot(ep, val_losses, 'r--', label='Val')
            axes[0].set_title('Combined loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
            axes[1].plot(ep, train_lat_accs, 'b-', label='Train')
            axes[1].plot(ep, val_lat_accs, 'r--', label='Val')
            axes[1].axhline(1/N_LATENT_CLASSES, color='gray', linestyle=':',
                             label=f'Chance ({100/N_LATENT_CLASSES:.0f}%)')
            axes[1].set_ylim([0, 1]); axes[1].set_title('Latent accuracy')
            axes[1].legend(); axes[1].grid(alpha=0.3)
            axes[2].plot(ep, train_coup_accs, 'b-', label='Train')
            axes[2].plot(ep, val_coup_accs, 'r--', label='Val')
            axes[2].axhline(1/N_COUPLING_CLASSES, color='gray', linestyle=':',
                             label=f'Chance ({100/N_COUPLING_CLASSES:.0f}%)')
            axes[2].set_ylim([0, 1]); axes[2].set_title('Coupling accuracy')
            axes[2].legend(); axes[2].grid(alpha=0.3)
            axes[3].plot(ep, train_eq_accs, 'b-', label='Train')
            axes[3].plot(ep, val_eq_accs, 'r--', label='Val')
            axes[3].axhline(1/N_EQ_CLASSES, color='gray', linestyle=':',
                             label=f'Chance ({100/N_EQ_CLASSES:.0f}%)')
            axes[3].set_ylim([0, 1]); axes[3].set_title('Coupling-eq accuracy (non-null)')
            axes[3].legend(); axes[3].grid(alpha=0.3)
            axes[4].plot(ep, train_entropies, 'b-', label='Train')
            axes[4].plot(ep, val_entropies, 'r--', label='Val')
            axes[4].axhline(np.log(N_EQ_CLASSES), color='gray', linestyle=':',
                             label=f'Max entropy ({np.log(N_EQ_CLASSES):.3f})')
            axes[4].set_ylim([0, np.log(N_EQ_CLASSES) * 1.1])
            axes[4].set_title('Gate entropy (lower = sharper)')
            axes[4].legend(); axes[4].grid(alpha=0.3)
            fig.suptitle(f'LTInetV9 Dual-Branch Eq-Gated [Sharpened+Widened, NoOvershoot] | Epoch {epoch} | '
                         f'Lat {va_lat:.3f} | Coup {va_coup:.3f} | Eq {va_eq:.3f} | '
                         f'GateEnt {va_ent:.3f} | Best avg {best_avg_acc:.3f}')
            plt.tight_layout(); plt.show()

        if no_improve >= es_patience:
            print(f'\nEarly stopping at epoch {epoch}')
            break

    print(f'\nTraining complete. Best avg val acc: {best_avg_acc:.3f}')
    print(f'Weights: {SAVE_PATH}')

    evaluate_per_system(model, preloaded_val, device,
                         nc_mean, nc_std, xc_mean, xc_std, anf_mean, anf_std, BATCH_SIZE)