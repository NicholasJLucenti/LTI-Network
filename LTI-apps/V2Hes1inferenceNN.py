import scipy.io
import numpy as np
import torch
import torch.nn as nn

# ── Label remap — must match exactly what the trainer used ───────────────
ACTIVE_CLASSES = [0, 1, 5, 6, 7]
LABEL_REMAP    = {orig: new for new, orig in enumerate(ACTIVE_CLASSES)}

# ── Model definition — copied from 5-class Colab trainer ─────────────────
# Must match the saved weights exactly. Do not import from an older file.

class Branch1_Hybrid(nn.Module):
    def __init__(self, n_filters=32, output_dim=64):
        super().__init__()
        self.fft_stream_small = nn.Sequential(
            nn.Conv1d(3, n_filters, kernel_size=3,  padding=1),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_filters), nn.ReLU())
        self.fft_stream_medium = nn.Sequential(
            nn.Conv1d(3, n_filters, kernel_size=7,  padding=3),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(n_filters), nn.ReLU())
        self.fft_stream_large = nn.Sequential(
            nn.Conv1d(3, n_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(n_filters), nn.ReLU())
        self.raw_stream_small = nn.Sequential(
            nn.Conv1d(2, n_filters, kernel_size=5,  padding=2),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_filters), nn.ReLU())
        self.raw_stream_medium = nn.Sequential(
            nn.Conv1d(2, n_filters, kernel_size=21, padding=10),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=21, padding=10),
            nn.BatchNorm1d(n_filters), nn.ReLU())
        self.raw_stream_large = nn.Sequential(
            nn.Conv1d(2, n_filters, kernel_size=51, padding=25),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=51, padding=25),
            nn.BatchNorm1d(n_filters), nn.ReLU())
        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(6 * n_filters, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim), nn.ReLU())

    def forward(self, fft, raw):
        fs = self.gap(self.fft_stream_small(fft)).squeeze(-1)
        fm = self.gap(self.fft_stream_medium(fft)).squeeze(-1)
        fl = self.gap(self.fft_stream_large(fft)).squeeze(-1)
        rs = self.gap(self.raw_stream_small(raw)).squeeze(-1)
        rm = self.gap(self.raw_stream_medium(raw)).squeeze(-1)
        rl = self.gap(self.raw_stream_large(raw)).squeeze(-1)
        return self.projection(torch.cat([fs, fm, fl, rs, rm, rl], dim=1))


class Branch2_TermAttention(nn.Module):
    def __init__(self, n_terms=9, n_equations=2, embed_dim=32, output_dim=64):
        super().__init__()
        self.asym_proj = nn.Sequential(
            nn.Linear(n_terms, 32), nn.ReLU(),
            nn.Linear(32, 16))
        self.embed     = nn.Linear(n_terms, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(n_equations * embed_dim + 16, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim), nn.ReLU())

    def forward(self, x):
        asym     = self.asym_proj(x[:, :, 0] - x[:, :, 1])
        tokens   = self.embed(x.permute(0, 2, 1))
        attn_out, _ = self.attention(tokens, tokens, tokens)
        flat     = attn_out.reshape(attn_out.shape[0], -1)
        return self.projection(torch.cat([flat, asym], dim=1))


class Branch3_TopologyEmbed(nn.Module):
    def __init__(self, n_classes=4, embed_dim=16, output_dim=16):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.proj  = nn.Sequential(nn.Linear(embed_dim, output_dim), nn.ReLU())

    def forward(self, x):
        return self.proj(self.embed(x))


class LTInetV2(nn.Module):
    def __init__(self, n_structure_classes=5):
        super().__init__()
        self.branch1 = Branch1_Hybrid(n_filters=32, output_dim=64)
        self.branch2 = Branch2_TermAttention(n_terms=9, output_dim=64)
        self.branch3 = Branch3_TopologyEmbed(n_classes=4, output_dim=16)
        # merge_dim = 64 (B1) + 64 (B2) + 16 (B3) = 144
        self.head = nn.Sequential(
            nn.Linear(144, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_structure_classes))

    def forward(self, fft_features, residual, xi_ternary, topo_label):
        b1 = self.branch1(fft_features, residual)
        b2 = self.branch2(xi_ternary)
        b3 = self.branch3(topo_label)
        return self.head(torch.cat([b1, b2, b3], dim=1))


# ── Load inference data ───────────────────────────────────────────────────
data = scipy.io.loadmat(r'C:/Users/nickj/MATLAB Drive/Hes1_inference.mat')
assert 'fft_features' in data, \
    'fft_features missing — re-run Hes1_inference_generator.m'

# ── FFT features (3 x 251) ───────────────────────────────────────────────
fft_features = torch.tensor(
    data['fft_features'].astype(np.float32)).unsqueeze(0)

# ── Raw residuals (2 x 501) ──────────────────────────────────────────────
rx = data['resid_dx'].squeeze()
ry = data['resid_dy'].squeeze()

rx = np.clip(rx, -1e6, 1e6).astype(np.float32)
ry = np.clip(ry, -1e6, 1e6).astype(np.float32)
rx = np.nan_to_num(rx, nan=0.0, posinf=0.0, neginf=0.0)
ry = np.nan_to_num(ry, nan=0.0, posinf=0.0, neginf=0.0)
rx = (rx - rx.mean()) / (rx.std() + 1e-8)
ry = (ry - ry.mean()) / (ry.std() + 1e-8)

if len(rx) < 501:
    pad = 501 - len(rx)
    rx  = np.concatenate([rx, np.zeros(pad)])
    ry  = np.concatenate([ry, np.zeros(pad)])

residual = torch.tensor(
    np.stack([rx, ry], axis=0).astype(np.float32)).unsqueeze(0)

# ── Xi ternary (9 x 2) ───────────────────────────────────────────────────
xi_raw = data['Xi_ternary'].astype(np.float32)
if xi_raw.shape[0] < 9:
    xi_raw = np.vstack([xi_raw,
                        np.zeros((9 - xi_raw.shape[0], 2), dtype=np.float32)])
xi_ternary = torch.tensor(xi_raw).unsqueeze(0)

# ── Topology label ────────────────────────────────────────────────────────
topology_map = {
    'LIMIT CYCLE':        0,
    'DAMPED OSCILLATION': 1,
    'STEADY STATE':       2,
    'UNDETERMINED':       3
}
topo_str   = str(data['topology'][0]).strip()
topo_label = torch.tensor([topology_map.get(topo_str, 3)], dtype=torch.long)

# ── Load weights ──────────────────────────────────────────────────────────
WEIGHTS_PATH = r'C:/Users/nickj/MATLAB Drive/LTInetV2_5class_best.pth'

model = LTInetV2(n_structure_classes=5)
model.load_state_dict(torch.load(WEIGHTS_PATH,
                                  map_location=torch.device('cpu')))

# ── MC dropout inference ──────────────────────────────────────────────────
model.train()
n_samples = 100
all_probs  = []

with torch.no_grad():
    for _ in range(n_samples):
        logits = model(fft_features, residual, xi_ternary, topo_label)
        all_probs.append(torch.softmax(logits, dim=1).squeeze())

probs    = torch.stack(all_probs).mean(dim=0)
prob_std = torch.stack(all_probs).std(dim=0)

# ── Class definitions ─────────────────────────────────────────────────────
# Remapped 0-4 → original Lat index → equation
structure_names = [
    'Class 0 (Lat0) — Linear x          (dI/dt = kp*x - kd*I)',
    'Class 1 (Lat1) — Hill act. x       (dI/dt = kp*H_act(x,ka,na) - kd*I)',
    'Class 2 (Lat5) — Hill rep. x       (dI/dt = kp*H_rep(x,k0,n) - kd*I)',
    'Class 3 (Lat6) — Additive x+y      (dI/dt = kp*x + kq*y - kd*I)',
    'Class 4 (Lat7) — Multiplicative xy (dI/dt = kp*x*y - kd*I)',
]

# ── Print results ─────────────────────────────────────────────────────────
print(f'\nLTInetV2 Hes1 Inference — topology: {topo_str}')
print(f'5-class model | Active latents: Lat0, Lat1, Lat5, Lat6, Lat7')
print('=' * 68)
for name, p, s in zip(structure_names, probs, prob_std):
    bar = '█' * int(p.item() * 30)
    print(f'{name}')
    print(f'  Probability: {p.item():.4f} ± {s.item():.4f}  |{bar}')
print('=' * 68)
best_idx = probs.argmax().item()
print(f'Most likely: {structure_names[best_idx]}')
print(f'Confidence:  {probs[best_idx].item():.4f} ± {prob_std[best_idx].item():.4f}')
print(f'Entropy:     {(-probs * torch.log(probs + 1e-8)).sum().item():.4f} '
      f'(max = {torch.log(torch.tensor(5.0)).item():.4f})')