import scipy.io
import numpy as np
import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────
MAT_PATH    = r'C:/Users/nickj/MATLAB Drive/Hes1_V4_inference.mat'
WEIGHTS_PATH= r'C:/Users/nickj/MATLAB Drive/LTInetV4_dual_best.pth'
NORM_ROOT   = r'C:/Users/nickj/MATLAB Drive'

N_LATENT_CLASSES   = 5
N_COUPLING_CLASSES = 3
NULLCLINE_DIM      = 49
XCORR_DIM          = 10

latent_names = [
    'Lat0  Linear mRNA readout       dI/dt = kp·x - kd·I',
    'Lat1  miRNA titration           dI/dt = kp·y - (kd + km·x)·I',
    'Lat2  Incoherent feedforward    dI/dt = kp·x·H_rep(x) - kd·I',
    'Lat3  GK phospho cycle          dI/dt = kp·x·(1-I/Imax) - kcat·I/(Km+I)',
    'Lat4  Hill repression by y      dI/dt = kp·H_rep(y) - kd·I',
]
coupling_names = [
    'Ch0  Hill repressor    -alpha_c · H_rep(I, k0, n)',
    'Ch1  Linear additive   -beta_c · I',
    'Ch2  Multiplicative    -gamma_c · I · x',
]
latent_short   = ['Lin-x', 'miRNA', 'IncoFF', 'GK-ph', 'HRep-y']
coupling_short = ['HillRep', 'Additive', 'Mult']

topology_map = {
    'LIMIT CYCLE':        0,
    'DAMPED OSCILLATION': 1,
    'STEADY STATE':       2,
    'UNDETERMINED':       3
}

# ── Model definition (must match trainer exactly) ─────────────────────────
class Branch1_NullclineMLP(nn.Module):
    def __init__(self, input_dim=49, output_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(128,96),nn.BatchNorm1d(96),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(96,output_dim),nn.ReLU())
    def forward(self,x): return self.net(x)

class Branch1b_XcorrMLP(nn.Module):
    def __init__(self, input_dim=10, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(32,output_dim),nn.ReLU())
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

class LTInetV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1  = Branch1_NullclineMLP(49,48)
        self.branch1b = Branch1b_XcorrMLP(10,16)
        self.branch2  = Branch2_TermAttention(9,output_dim=24)
        self.branch3  = Branch3_TopologyEmbed(4,output_dim=8)
        self.head_latent = nn.Sequential(
            nn.Linear(96,64),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(32,N_LATENT_CLASSES))
        self.head_coupling = nn.Sequential(
            nn.Linear(96,32),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(32,16),nn.ReLU(),
            nn.Linear(16,N_COUPLING_CLASSES))
    def forward(self,nc,xc,xi,topo):
        merged=torch.cat([self.branch1(nc),self.branch1b(xc),
                          self.branch2(xi),self.branch3(topo)],dim=1)
        return self.head_latent(merged),self.head_coupling(merged)

# ── Load inference data ───────────────────────────────────────────────────
data = scipy.io.loadmat(MAT_PATH)

nc_raw = data['nullcline_features'].squeeze()
assert nc_raw.shape[0] == NULLCLINE_DIM, \
    f'Expected {NULLCLINE_DIM} nullcline features, got {nc_raw.shape[0]}'
nc = torch.tensor(np.clip(nc_raw,-50,50).astype(np.float32)).unsqueeze(0)

xc_raw = data['xcorr_features'].squeeze()
assert xc_raw.shape[0] == XCORR_DIM, \
    f'Expected {XCORR_DIM} xcorr features, got {xc_raw.shape[0]}'
xc = torch.tensor(np.clip(xc_raw,-50,50).astype(np.float32)).unsqueeze(0)

xi = data['Xi_ternary'].astype(np.float32)
if xi.shape[0] < 9:
    xi = np.vstack([xi, np.zeros((9-xi.shape[0],2),dtype=np.float32)])
xi = torch.tensor(xi).unsqueeze(0)

topo_str  = str(data['topology'][0]).strip()
topo_lbl  = topology_map.get(topo_str, 3)
topo      = torch.tensor([topo_lbl], dtype=torch.long)

# ── Load normalisation ────────────────────────────────────────────────────
try:
    nc_mean = torch.tensor(np.load(f'{NORM_ROOT}/V4_nc_mean.npy'),dtype=torch.float32)
    nc_std  = torch.tensor(np.load(f'{NORM_ROOT}/V4_nc_std.npy'), dtype=torch.float32)
    xc_mean = torch.tensor(np.load(f'{NORM_ROOT}/V4_xc_mean.npy'),dtype=torch.float32)
    xc_std  = torch.tensor(np.load(f'{NORM_ROOT}/V4_xc_std.npy'), dtype=torch.float32)
    nc_std[nc_std < 1e-6] = 1.0
    xc_std[xc_std < 1e-6] = 1.0
except FileNotFoundError as e:
    raise FileNotFoundError(
        'Normalisation .npy files not found. '
        'These are saved automatically during training when a new best val acc is reached. '
        f'Missing: {e}')

nc = (nc - nc_mean) / nc_std
xc = (xc - xc_mean) / xc_std

# ── Load model ────────────────────────────────────────────────────────────
model = LTInetV4()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))

# ── MC dropout inference (dropout active = uncertainty estimation) ─────────
model.eval()    # BatchNorm uses running stats (works with batch size 1)
# Manually enable dropout layers for MC uncertainty estimation
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.train()
n_samples = 300
all_lat_probs  = []
all_coup_probs = []

with torch.no_grad():
    for _ in range(n_samples):
        logits_lat, logits_coup = model(nc, xc, xi, topo)
        all_lat_probs.append(torch.softmax(logits_lat,  dim=1).squeeze())
        all_coup_probs.append(torch.softmax(logits_coup, dim=1).squeeze())

lat_probs  = torch.stack(all_lat_probs)
coup_probs = torch.stack(all_coup_probs)

lat_mean  = lat_probs.mean(dim=0)
lat_std   = lat_probs.std(dim=0)
coup_mean = coup_probs.mean(dim=0)
coup_std  = coup_probs.std(dim=0)

lat_entropy  = (-lat_mean  * torch.log(lat_mean  + 1e-8)).sum().item()
coup_entropy = (-coup_mean * torch.log(coup_mean + 1e-8)).sum().item()

# ── Print results ─────────────────────────────────────────────────────────
print(f'\nLTInetV4 Hes1 Inference')
print(f'Topology: {topo_str}  |  MC samples: {n_samples}')
print('=' * 72)

print('\n── Latent kinetic form ──')
print(f'  {"Latent":<50} {"P":>7}  {"±":>7}')
print('  ' + '-'*66)
for i in range(N_LATENT_CLASSES):
    p = lat_mean[i].item(); s = lat_std[i].item()
    bar  = '█' * int(p * 30)
    best = ' ◄' if i == lat_mean.argmax().item() else ''
    print(f'  {latent_names[i]:<50} {p:>7.4f}  {s:>7.4f}  |{bar}{best}')
print(f'  Entropy: {lat_entropy:.4f} / {np.log(N_LATENT_CLASSES):.4f} '
      f'({100*lat_entropy/np.log(N_LATENT_CLASSES):.1f}% of max)')

print('\n── Coupling channel ──')
print(f'  {"Channel":<50} {"P":>7}  {"±":>7}')
print('  ' + '-'*66)
for i in range(N_COUPLING_CLASSES):
    p = coup_mean[i].item(); s = coup_std[i].item()
    bar  = '█' * int(p * 30)
    best = ' ◄' if i == coup_mean.argmax().item() else ''
    print(f'  {coupling_names[i]:<50} {p:>7.4f}  {s:>7.4f}  |{bar}{best}')
print(f'  Entropy: {coup_entropy:.4f} / {np.log(N_COUPLING_CLASSES):.4f} '
      f'({100*coup_entropy/np.log(N_COUPLING_CLASSES):.1f}% of max)')

# ── Joint probability table ───────────────────────────────────────────────
print('\n── Joint probability table (Latent × Coupling) ──')
print(f"  {'':>10}" + ''.join(f'{s:>12}' for s in coupling_short))
print('  ' + '-'*46)
joint = torch.outer(lat_mean, coup_mean)
for i in range(N_LATENT_CLASSES):
    row = f'  {latent_short[i]:>10}'
    for j in range(N_COUPLING_CLASSES):
        v = joint[i, j].item()
        cell = f'{v:.3f}'
        if i == lat_mean.argmax().item() and j == coup_mean.argmax().item():
            cell = f'[{v:.3f}]'
        row += f'{cell:>12}'
    print(row)

# ── Top prediction summary ────────────────────────────────────────────────
best_lat  = lat_mean.argmax().item()
best_coup = coup_mean.argmax().item()
print('\n' + '=' * 72)
print(f'Most likely latent:   {latent_names[best_lat].strip()}')
print(f'  Confidence: {lat_mean[best_lat].item():.4f} ± {lat_std[best_lat].item():.4f}')
print(f'Most likely coupling: {coupling_names[best_coup].strip()}')
print(f'  Confidence: {coup_mean[best_coup].item():.4f} ± {coup_std[best_coup].item():.4f}')
print(f'Joint probability:    {joint[best_lat, best_coup].item():.4f}')
print('=' * 72)