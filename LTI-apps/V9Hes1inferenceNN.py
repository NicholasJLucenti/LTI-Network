import scipy.io
import numpy as np
import torch
import torch.nn as nn

from LTInetV9 import (
    LTInetV9, N_LATENT_CLASSES, N_COUPLING_CLASSES, N_EQ_CLASSES,
    NULLCLINE_DIM, XCORR_DIM, ALL_NEW_DIM, V6_NEW_DIM, FP_MULTI_DIM,
    latent_names, coupling_names, eq_names, topology_map, WEIGHTS_DIR
)

# This uses the DUAL-BRANCH, EQ-GATED model — the full 7-latent/4-coupling
# class structure, trained on the complete pooled dataset (Eq0+Eq1+NULL),
# with two specialized branch stacks combined via soft eq-gating. Unlike
# every prior V9 script, eq is a genuine THIRD joint output here (not
# fixed by which data subset the model was trained on) — so this script
# reports latent, coupling, AND eq predictions together, each with its
# own MC-dropout confidence interval.

HES1_DATA_DIR = r'C:/Users/nickj/LTInetV8 Local Data Drive/Hes1'
MAT_PATH = f'{HES1_DATA_DIR}/Hes1_V8_inference.mat'

latent_descriptions = {
    'L_mirna':     'miRNA-style titration: dI/dt = kp*y - (kd + km*x)*I',
    'L_gk':        'Goldbeter-Koshland dual saturation: dI/dt = kp*x*(1-I/Imax) - kcat*I/(Km+I)',
    'L_yhill':     'Hill repression by y: dI/dt = kp*Hrep(y,k0,n) - kd*I',
    'L_bistable':  'Positive autoregulation: dI/dt = kp*x + kfb*I^2/(k0^2+I^2) - kd*I',
    'L_delay':     'Delayed relay: dI/dt = kp*z - kd*I, dz/dt = kz*(x-z)',
    'L_overshoot': 'Overshoot/adaptation: dI/dt = kp*x - kp2*x^2 - kd*I',
    'NULL':        'Null - no latent variable (2D model is complete)',
}
coupling_descriptions = {
    'Ch0 HillRep': 'Hill repressor: -alpha_c * Hrep(I,k0,n)',
    'Ch1 Additive': 'Linear additive: -beta_c * I',
    'Ch2 Multiplicative': 'Multiplicative: -gamma_c * I * (x or y, depending on eq)',
    'Ch3 None': 'None (null - no coupling term)',
}
eq_descriptions = {
    'Eq0 -> dx': 'Coupling acts on the x-equation (mRNA dynamics)',
    'Eq1 -> dy': 'Coupling acts on the y-equation (protein dynamics) - '
                 'matches the hypothesized Stat3 phosphorylation-on-degradation mechanism',
}

# ── Load Hes1 feature vector ────────────────────────────────────────────
data = scipy.io.loadmat(MAT_PATH)

nc_raw = data['nullcline_features'].squeeze()
assert nc_raw.shape[0] == NULLCLINE_DIM, f"nullcline_features shape mismatch: {nc_raw.shape}"
nc = torch.tensor(np.clip(nc_raw, -50, 50).astype(np.float32)).unsqueeze(0)

xc_raw = data['xcorr_features'].squeeze()
assert xc_raw.shape[0] == XCORR_DIM, f"xcorr_features shape mismatch: {xc_raw.shape}"
xc = torch.tensor(np.clip(xc_raw, -50, 50).astype(np.float32)).unsqueeze(0)

if 'all_new_features' not in data:
    raise KeyError("'all_new_features' missing - re-run Hes1_V8_Generate_Inference.m")
anf_raw = data['all_new_features'].squeeze()
assert anf_raw.shape[0] == ALL_NEW_DIM, f"all_new_features shape mismatch: {anf_raw.shape}"

if 'v6_new_features' not in data:
    raise KeyError("'v6_new_features' missing - re-run Hes1_V8_Generate_Inference.m")
v6n_raw = data['v6_new_features'].squeeze()
assert v6n_raw.shape[0] == V6_NEW_DIM, f"v6_new_features shape mismatch: {v6n_raw.shape}"

if 'fp_multi_features' in data:
    fpm_raw = data['fp_multi_features'].squeeze()
    if fpm_raw.shape[0] != FP_MULTI_DIM:
        fpm_raw = np.zeros(FP_MULTI_DIM, dtype=np.float32)
else:
    fpm_raw = np.zeros(FP_MULTI_DIM, dtype=np.float32)

anf_full_raw = np.concatenate([
    np.clip(anf_raw, -50, 50).astype(np.float32),
    np.clip(v6n_raw, -50, 50).astype(np.float32),
    np.clip(fpm_raw, -50, 50).astype(np.float32),
])
anf = torch.tensor(anf_full_raw).unsqueeze(0)

xi = data['Xi_ternary'].astype(np.float32)
if xi.shape[0] < 9:
    xi = np.vstack([xi, np.zeros((9-xi.shape[0], 2), dtype=np.float32)])
elif xi.shape[0] > 9:
    xi = xi[:9, :]
xi = torch.tensor(xi).unsqueeze(0)

topo_str = str(data['topology'][0]).strip()
topo = torch.tensor([topology_map.get(topo_str, 3)], dtype=torch.long)

print(f'Loaded Hes1 feature vector. Topology: {topo_str}')

# ── Normalization — use the DualBranchEqGated-tagged stats ─────────────
try:
    nc_mean  = torch.tensor(np.load(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_nc_mean.npy'),  dtype=torch.float32)
    nc_std   = torch.tensor(np.load(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_nc_std.npy'),   dtype=torch.float32)
    xc_mean  = torch.tensor(np.load(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_xc_mean.npy'),  dtype=torch.float32)
    xc_std   = torch.tensor(np.load(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_xc_std.npy'),   dtype=torch.float32)
    anf_mean = torch.tensor(np.load(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_anf_mean.npy'), dtype=torch.float32)
    anf_std  = torch.tensor(np.load(f'{WEIGHTS_DIR}/V9_DualBranchEqGated_anf_std.npy'),  dtype=torch.float32)
    nc_std[nc_std < 1e-6] = 1.
    xc_std[xc_std < 1e-6] = 1.
    anf_std[anf_std < 1e-6] = 1.
except FileNotFoundError as e:
    raise FileNotFoundError(f'DualBranchEqGated norm files not found in {WEIGHTS_DIR}. Missing: {e}')

# Sanity check: raw Hes1 features vs training normalization stats.
nc_z = ((nc.squeeze() - nc_mean) / nc_std).abs()
xc_z = ((xc.squeeze() - xc_mean) / xc_std).abs()
anf_z = ((anf.squeeze() - anf_mean) / anf_std).abs()
max_z = max(nc_z.max().item(), xc_z.max().item(), anf_z.max().item())
print(f'Max |normalized feature value| across all blocks: {max_z:.2f}')
if max_z > 6:
    print('  [WARNING] Some Hes1 feature(s) fall >6 std devs from the synthetic')
    print('  training distribution. This is a real out-of-distribution signal -')
    print('  treat any resulting prediction with proportionally more caution,')
    print('  and consider inspecting which specific feature(s) are extreme.')

nc  = (nc  - nc_mean)  / nc_std
xc  = (xc  - xc_mean)  / xc_std
anf = (anf - anf_mean) / anf_std

# ── Load model ────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LTInetV9().to(device)
model.load_state_dict(torch.load(f'{WEIGHTS_DIR}/LTInetV9_DualBranchEqGated_best.pth', map_location=device))
model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.train()  # MC dropout: keep dropout active at inference for uncertainty estimates

nc, xc, anf, xi, topo = [t.to(device) for t in (nc, xc, anf, xi, topo)]

# ── MC dropout inference ──────────────────────────────────────────────
# Dual-branch model returns (lat_logits, coup_logits, eq_logits) — eq is
# a genuine third joint output here, not fixed by data subset.
n_samples = 300
all_lat_probs, all_coup_probs, all_eq_probs = [], [], []
with torch.no_grad():
    for _ in range(n_samples):
        ll, lc, le = model(nc, xc, anf, xi, topo)
        all_lat_probs.append(torch.softmax(ll, dim=1).squeeze())
        all_coup_probs.append(torch.softmax(lc, dim=1).squeeze())
        all_eq_probs.append(torch.softmax(le, dim=1).squeeze())

lat_probs  = torch.stack(all_lat_probs)
coup_probs = torch.stack(all_coup_probs)
eq_probs   = torch.stack(all_eq_probs)
lat_mean   = lat_probs.mean(0);  lat_std  = lat_probs.std(0)
coup_mean  = coup_probs.mean(0); coup_std = coup_probs.std(0)
eq_mean    = eq_probs.mean(0);   eq_std   = eq_probs.std(0)
lat_entropy  = (-lat_mean  * torch.log(lat_mean  + 1e-8)).sum().item()
coup_entropy = (-coup_mean * torch.log(coup_mean + 1e-8)).sum().item()
eq_entropy   = (-eq_mean   * torch.log(eq_mean   + 1e-8)).sum().item()

# ── Results ───────────────────────────────────────────────────────────
print(f'\nLTInetV9 [Dual-Branch, Eq-Gated] Hes1 Inference (real experimental data)')
print(f'Topology: {topo_str}  |  MC samples: {n_samples}')
print('=' * 90)

print('\n-- Latent kinetic form --')
print(f'  {"Latent":<14} {"P":>7}  {"+/-":>7}  {"Description"}')
print('  ' + '-'*86)
for i in range(N_LATENT_CLASSES):
    name = latent_names[i]
    p_val = lat_mean[i].item(); s = lat_std[i].item()
    bar = '#' * int(p_val*30)
    best = ' <-' if i == lat_mean.argmax().item() else ''
    null_note = '  [NULL - model may be complete as-is]' if name == 'NULL' and best else ''
    desc = latent_descriptions.get(name, '')
    print(f'  {name:<14} {p_val:>7.4f}  {s:>7.4f}  |{bar}{best}{null_note}')
    print(f'  {"":<14} {"":>7}  {"":>7}   {desc}')
print(f'\n  Entropy: {lat_entropy:.4f} / {np.log(N_LATENT_CLASSES):.4f} '
      f'({100*lat_entropy/np.log(N_LATENT_CLASSES):.1f}% of max)')

print('\n-- Coupling channel --')
print(f'  {"Channel":<20} {"P":>7}  {"+/-":>7}  {"Description"}')
print('  ' + '-'*86)
for i in range(N_COUPLING_CLASSES):
    name = coupling_names[i]
    p_val = coup_mean[i].item(); s = coup_std[i].item()
    bar = '#' * int(p_val*30)
    best = ' <-' if i == coup_mean.argmax().item() else ''
    desc = coupling_descriptions.get(name, '')
    print(f'  {name:<20} {p_val:>7.4f}  {s:>7.4f}  |{bar}{best}')
    print(f'  {"":<20} {"":>7}  {"":>7}   {desc}')
print(f'\n  Entropy: {coup_entropy:.4f} / {np.log(N_COUPLING_CLASSES):.4f} '
      f'({100*coup_entropy/np.log(N_COUPLING_CLASSES):.1f}% of max)')

print('\n-- Coupling equation target (NEW: genuine joint output) --')
print(f'  {"Target":<14} {"P":>7}  {"+/-":>7}  {"Description"}')
print('  ' + '-'*86)
for i in range(N_EQ_CLASSES):
    name = eq_names[i]
    p_val = eq_mean[i].item(); s = eq_std[i].item()
    bar = '#' * int(p_val*30)
    best = ' <-' if i == eq_mean.argmax().item() else ''
    desc = eq_descriptions.get(name, '')
    print(f'  {name:<14} {p_val:>7.4f}  {s:>7.4f}  |{bar}{best}')
    print(f'  {"":<14} {"":>7}  {"":>7}   {desc}')
print(f'\n  Entropy: {eq_entropy:.4f} / {np.log(N_EQ_CLASSES):.4f} '
      f'({100*eq_entropy/np.log(N_EQ_CLASSES):.1f}% of max)')
print('  NOTE: this model was trained on VALIDATION accuracy of 93.8% for this head,')
print('  the strongest of the three outputs - weight this prediction accordingly')
print('  relative to the noisier latent/coupling calls above.')

print('\n-- Joint probability table (Latent x Coupling, marginalized over Eq) --')
print(f"  {'':>12}" + ''.join(f'{s[:10]:>12}' for s in coupling_names))
print('  ' + '-'*(12 + 12*N_COUPLING_CLASSES))
joint = torch.outer(lat_mean, coup_mean)
for i in range(N_LATENT_CLASSES):
    row = f'  {latent_names[i][:11]:>12}'
    for j in range(N_COUPLING_CLASSES):
        v = joint[i,j].item()
        cell = f'{v:.3f}'
        if i == lat_mean.argmax().item() and j == coup_mean.argmax().item():
            cell = f'[{v:.3f}]'
        row += f'{cell:>12}'
    print(row)

best_lat = lat_mean.argmax().item()
best_coup = coup_mean.argmax().item()
best_eq = eq_mean.argmax().item()
print('\n' + '='*90)
if latent_names[best_lat] == 'NULL':
    print('RESULT: NULL - the 2D Hes1 model appears complete. No latent variable')
    print('        is required to explain the observed mRNA/protein dynamics.')
    print('        (This would be surprising given the prior anomalous-term finding')
    print('         from earlier RI-SINDy fits - worth double-checking feature')
    print('         quality/OOD warnings above before accepting this at face value.)')
else:
    print(f'Most likely latent:    {latent_names[best_lat]}')
    print(f'  {latent_descriptions[latent_names[best_lat]]}')
    print(f'  Confidence: {lat_mean[best_lat].item():.4f} +/- {lat_std[best_lat].item():.4f}')
    print(f'Most likely coupling:  {coupling_names[best_coup]}')
    print(f'  {coupling_descriptions[coupling_names[best_coup]]}')
    print(f'  Confidence: {coup_mean[best_coup].item():.4f} +/- {coup_std[best_coup].item():.4f}')
    print(f'Most likely eq target: {eq_names[best_eq]}')
    print(f'  {eq_descriptions[eq_names[best_eq]]}')
    print(f'  Confidence: {eq_mean[best_eq].item():.4f} +/- {eq_std[best_eq].item():.4f}')
    print(f'Joint probability (latent x coupling): {joint[best_lat,best_coup].item():.4f}')
print('='*90)