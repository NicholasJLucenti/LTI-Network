import scipy.io
import numpy as np
import torch
import torch.nn as nn

# ── Label remap — must match exactly what the trainer used ───────────────
# Original Lat indices → contiguous 0-6 for the 7-class model
ACTIVE_CLASSES = [0, 1, 8, 9, 10, 11, 12]
LABEL_REMAP    = {orig: new for new, orig in enumerate(ACTIVE_CLASSES)}

# ── Model definition — copied from 7-class trainer ────────────────────────
# Must match saved weights exactly. Inline to avoid import version mismatch.

class Branch1_NullclineMLP(nn.Module):
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
    def __init__(self, n_structure_classes=7):
        super().__init__()
        self.branch1 = Branch1_NullclineMLP(input_dim=49, output_dim=48)
        self.branch2 = Branch2_TermAttention(n_terms=9, output_dim=24)
        self.branch3 = Branch3_TopologyEmbed(n_classes=4, output_dim=8)
        self.head = nn.Sequential(
            nn.Linear(80, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, n_structure_classes)
        )

    def forward(self, nullcline, xi_ternary, topo_label):
        b1 = self.branch1(nullcline)
        b2 = self.branch2(xi_ternary)
        b3 = self.branch3(topo_label)
        return self.head(torch.cat([b1, b2, b3], dim=1))


# ── Load inference data ───────────────────────────────────────────────────
# Run Hes1_inference_generator.m first to produce this file.
# The .mat file must contain: nullcline_features, Xi_ternary, topology.
MAT_PATH = r'C:/Users/nickj/MATLAB Drive/Hes1_inference.mat'

data = scipy.io.loadmat(MAT_PATH)

assert 'nullcline_features' in data, \
    'nullcline_features missing — re-run Hes1_inference_generator.m ' \
    'with the V3 nullcline extraction block'

# ── Nullcline features (49,) ──────────────────────────────────────────────
nc_raw = data['nullcline_features'].squeeze()
nc_raw = np.clip(nc_raw, -50.0, 50.0).astype(np.float32)
assert nc_raw.shape[0] == 49, \
    f'Expected 49 nullcline features, got {nc_raw.shape[0]}. ' \
    f'Regenerate with the updated MATLAB script.'

# ── Xi ternary (9 x 2) ───────────────────────────────────────────────────
xi_raw = data['Xi_ternary'].astype(np.float32)
if xi_raw.shape[0] < 9:
    xi_raw = np.vstack([xi_raw,
                        np.zeros((9 - xi_raw.shape[0], 2), dtype=np.float32)])

# ── Topology label ────────────────────────────────────────────────────────
topology_map = {
    'LIMIT CYCLE':        0,
    'DAMPED OSCILLATION': 1,
    'STEADY STATE':       2,
    'UNDETERMINED':       3
}
topo_str   = str(data['topology'][0]).strip()
topo_label = topology_map.get(topo_str, 3)

# ── Load normalisation stats ──────────────────────────────────────────────
# nc_mean and nc_std must come from the training dataset.
# Two options:
#   A) Save them from training (recommended — add lines below to trainer):
#        np.save('nc_mean.npy', nc_mean.numpy())
#        np.save('nc_std.npy',  nc_std.numpy())
#   B) Re-scan the training folders here (slow but self-contained).
# This script uses Option A — save the .npy files from your trainer first.
NORM_ROOT = r'C:/Users/nickj/MATLAB Drive'
try:
    nc_mean = torch.tensor(np.load(f'{NORM_ROOT}/nc_mean.npy'), dtype=torch.float32)
    nc_std  = torch.tensor(np.load(f'{NORM_ROOT}/nc_std.npy'),  dtype=torch.float32)
    nc_std[nc_std < 1e-6] = 1.0
except FileNotFoundError:
    raise FileNotFoundError(
        'nc_mean.npy and nc_std.npy not found. '
        'Add these two lines at the end of your trainer and re-run it:\n'
        '    import numpy as np\n'
        '    np.save(r"C:/Users/nickj/MATLAB Drive/nc_mean.npy", nc_mean.cpu().numpy())\n'
        '    np.save(r"C:/Users/nickj/MATLAB Drive/nc_std.npy",  nc_std.cpu().numpy())'
    )

# ── Assemble tensors ──────────────────────────────────────────────────────
nc_tensor   = torch.tensor(nc_raw).unsqueeze(0)
nc_tensor   = (nc_tensor - nc_mean) / nc_std
xi_tensor   = torch.tensor(xi_raw).unsqueeze(0)
topo_tensor = torch.tensor([topo_label], dtype=torch.long)

# ── Load weights ──────────────────────────────────────────────────────────
WEIGHTS_PATH = r'C:/Users/nickj/MATLAB Drive/LTInetV3_8class_x_best.pth'

model = LTInetV3(n_structure_classes=7)
model.load_state_dict(torch.load(WEIGHTS_PATH,
                                  map_location=torch.device('cpu')))

# ── MC dropout inference ──────────────────────────────────────────────────
model.train()   # keep dropout active for uncertainty estimation
n_samples = 200
all_probs  = []

with torch.no_grad():
    for _ in range(n_samples):
        logits = model(nc_tensor, xi_tensor, topo_tensor)
        all_probs.append(torch.softmax(logits, dim=1).squeeze())

probs    = torch.stack(all_probs).mean(dim=0)
prob_std = torch.stack(all_probs).std(dim=0)
entropy  = (-probs * torch.log(probs + 1e-8)).sum().item()
max_entropy = np.log(7)

# ── Class definitions ─────────────────────────────────────────────────────
# Remapped index → original Lat → equation
structure_names = [
    'Class 0 (Lat0)  — Linear x          dI/dt = kp·x - kd·I',
    'Class 1 (Lat1)  — Hill act. x       dI/dt = kp·H_act(x,ka,na) - kd·I',
    'Class 2 (Lat8)  — MM degradation    dI/dt = kp·x - kcat·I/(Km+I)',
    'Class 3 (Lat9)  — Coop. Hill x      dI/dt = kp·H_act(x,ka,n≥4) - kd·I',
    'Class 4 (Lat10) — Quadratic x       dI/dt = kp·x² - kd·I',
    'Class 5 (Lat11) — Incoherent FF     dI/dt = kp·x·H_rep(x,k0,n) - kd·I',
    'Class 6 (Lat12) — Delayed linear x  dI/dt = kp·x_filt - kd·I',
]

# ── Print results ─────────────────────────────────────────────────────────
print(f'\nLTInetV3 Hes1 Inference — topology: {topo_str}')
print(f'7-class x-driven model | MC samples: {n_samples}')
print('=' * 72)
for name, p, s in zip(structure_names, probs, prob_std):
    bar = '█' * int(p.item() * 30)
    print(f'{name}')
    print(f'  Probability: {p.item():.4f} ± {s.item():.4f}  |{bar}')
print('=' * 72)
best_idx = probs.argmax().item()
print(f'Most likely: {structure_names[best_idx]}')
print(f'Confidence:  {probs[best_idx].item():.4f} ± {prob_std[best_idx].item():.4f}')
print(f'Entropy:     {entropy:.4f} / {max_entropy:.4f} '
      f'({100*entropy/max_entropy:.1f}% of max)')