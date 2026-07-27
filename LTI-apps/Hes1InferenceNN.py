import scipy.io
import numpy as np
import torch
from LTInet_model import LTInet

data = scipy.io.loadmat(r'C:/Users/nickj/MATLAB Drive/Hes1_inference.mat')

rx = data['resid_dx'].squeeze()
ry = data['resid_dy'].squeeze()
rx = (rx - rx.mean()) / (rx.std() + 1e-8)
ry = (ry - ry.mean()) / (ry.std() + 1e-8)

target_length = 501
if len(rx) < target_length:
    pad = target_length - len(rx)
    rx  = np.concatenate([rx, np.zeros(pad)])
    ry  = np.concatenate([ry, np.zeros(pad)])

residual = torch.tensor(
    np.stack([rx, ry], axis=0).astype(np.float32)
).unsqueeze(0)

xi_raw = data['Xi_ternary'].astype(np.float32)
if xi_raw.shape[0] < 9:
    pad    = np.zeros((9 - xi_raw.shape[0], 2), dtype=np.float32)
    xi_raw = np.vstack([xi_raw, pad])
xi_ternary = torch.tensor(xi_raw).unsqueeze(0)

topology_map = {
    'LIMIT CYCLE':        0,
    'DAMPED OSCILLATION': 1,
    'STEADY STATE':       2,
    'UNDETERMINED':       3
}

topo_str   = str(data['topology'][0]).strip()
topo_label = torch.tensor(
    [topology_map.get(topo_str, 3)], dtype=torch.long
)

model = LTInet(n_structure_classes=6)
model.load_state_dict(torch.load('LTInet_best.pth',
    map_location=torch.device('cpu')))
model.eval()

model.train()
n_samples = 100
all_probs  = []
with torch.no_grad():
    for _ in range(n_samples):
        logits = model(residual, xi_ternary, topo_label)
        all_probs.append(torch.softmax(logits, dim=1).squeeze())

probs    = torch.stack(all_probs).mean(dim=0)
prob_std = torch.stack(all_probs).std(dim=0)

structure_names = [
    'Class 0 — Linear driven x        (dI/dt = a*x - b*I)',
    'Class 1 — Saturating driven       (dI/dt = a*H(x) - b*I)',
    'Class 2 — Coupled degradation     (dI/dt = a*x - b*x*I - c*I)',
    'Class 3 — Linear driven y         (dI/dt = a*y - b*I)',
    'Class 4 — Michaelis-Menten        (dI/dt = kf*x*(E-I) - kr*I - kcat*I)',
    'Class 5 — Nuclear transport       (dI/dt = ki*x - ke*I - kb*I*y)',
]

print(f'\nLTInet Inference — {topo_str}')
print('=' * 65)
for i, (name, p, s) in enumerate(zip(structure_names, probs, prob_std)):
    bar = '█' * int(p.item() * 30)
    print(f'{name}')
    print(f'  Probability: {p.item():.4f} ± {s.item():.4f}  |{bar}')
print('=' * 65)
print(f'Most likely: {structure_names[probs.argmax().item()]}')