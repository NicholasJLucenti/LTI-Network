function S = V6_SystemLib()
% V6_SystemLib — single source of truth for all 7 base systems.
% Returns a struct array, one entry per system, each with:
%   .name         string identifier
%   .psamp        @() -> param struct (includes hill_rep/hill_deg handles + noise)
%   .odefn_base   @(t,s,p) -> [dx;dy]   UNCOUPLED base dynamics (2-state)
%   .ic_fun       @(p) -> [x0;y0]
%   .libid        'standard' | 'brusselator' | 'fhn' | 'rma' | 'toggle'
%   .lib_ncols    number of library terms (for build_lib/lib_grid dispatch)
%   .coupling_site  'x_eq' — which equation the latent I couples into (V6: always x-eq,
%                   consistent with V5 convention; see coupling injector below)
%   .multi_fp     true if this system can have >1 fixed point (toggle switch)
%
% Coupling injection (how latent I modifies the x-equation) is handled
% separately by V6_CouplingInject.m — this file only defines UNCOUPLED
% base dynamics, matching the V5 convention where coupling is applied
% as an additive term on top of odefn_base's first component.

hill_rep = @(z,k,n)  k.^n ./ (k.^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);
hill_act = @(x,k,n)  x.^n ./ (k.^n + x.^n);

S = struct('name',{},'psamp',{},'odefn_base',{},'ic_fun',{},'libid',{}, ...
           'lib_ncols',{},'coupling_site',{},'multi_fp',{});

% ── 1. GOODWIN (existing) ──────────────────────────────────────────────
S(1).name = 'goodwin';
S(1).psamp = @() struct('alpha',5+7*rand(),'d1',0.3+0.7*rand(), ...
    'ks',1+3*rand(),'Vmax',3+5*rand(), ...
    'hill_n',round(3+5*rand()),'hill_k0',1.5+2.5*rand(), ...
    'hill_km',0.4+1.6*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
S(1).odefn_base = @(t,s,p) [p.alpha*p.hill_rep(s(2),p.hill_k0,p.hill_n)-p.d1*s(1);
                            p.ks*s(1)-p.Vmax*p.hill_deg(s(2),p.hill_km)];
S(1).ic_fun = @(p) [1.5; 0.5];
S(1).libid = 'standard'; S(1).lib_ncols = 9;
S(1).coupling_site = 'x_eq'; S(1).multi_fp = false;

% ── 2. BRUSSELATOR (existing) ──────────────────────────────────────────
S(2).name = 'brusselator';
S(2).psamp = @() sample_brussel(hill_rep,hill_deg);
S(2).odefn_base = @(t,s,p) [p.a-(p.b+1)*s(1)+s(1)^2*s(2);
                            p.b*s(1)-s(1)^2*s(2)];
S(2).ic_fun = @(p) [p.a; p.b/p.a];
S(2).libid = 'brusselator'; S(2).lib_ncols = 7;
S(2).coupling_site = 'x_eq'; S(2).multi_fp = false;

% ── 3. REPRESSILATOR (2-node reduction, existing) ──────────────────────
S(3).name = 'repressilator';
S(3).psamp = @() struct('alpha',4+6*rand(),'delta',1+2*rand(), ...
    'beta_rep',1+2*rand(),'gamma_rep',0.5+1.5*rand(), ...
    'hill_n',round(2+5*rand()),'hill_k0',1+3*rand(), ...
    'hill_km',0.4+1.6*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
S(3).odefn_base = @(t,s,p) [p.alpha*p.hill_rep(s(2),p.hill_k0,p.hill_n)-p.delta*p.hill_deg(s(1),p.hill_km);
                            p.beta_rep*s(1)-p.gamma_rep*s(2)];
S(3).ic_fun = @(p) [1.0; 0.5];
S(3).libid = 'standard'; S(3).lib_ncols = 9;
S(3).coupling_site = 'x_eq'; S(3).multi_fp = false;

% ── 4. VAN DER POL (existing) ──────────────────────────────────────────
S(4).name = 'van_der_pol';
S(4).psamp = @() struct('mu',0.3+0.7*rand(),'omega',0.8+0.4*rand(), ...
    'hill_k0',1+2*rand(),'hill_n',round(2+3*rand()), ...
    'hill_km',0.5+1.5*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
S(4).odefn_base = @(t,s,p) [p.mu*(1-s(2)^2)*s(1)-p.omega*s(2); s(1)];
S(4).ic_fun = @(p) [0.5; 0.0];
S(4).libid = 'standard'; S(4).lib_ncols = 9;
S(4).coupling_site = 'x_eq'; S(4).multi_fp = false;

% ── 5. FITZHUGH-NAGUMO (new — excitable regime) ────────────────────────
% dv/dt = v - v^3/3 - w + I_ext
% dw/dt = (v + a - b*w) / tau
% Standard params: a=0.7, b=0.8, tau=12.5 (Zillmer / FitzHugh 1961)
S(5).name = 'fitzhugh_nagumo';
S(5).psamp = @() struct('a',0.6+0.3*rand(),'b',0.7+0.3*rand(), ...
    'tau',9+7*rand(),'I_ext',0.2+0.5*rand(), ...
    'hill_k0',1+2*rand(),'hill_n',round(2+3*rand()), ...
    'hill_km',0.5+1.5*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
S(5).odefn_base = @(t,s,p) [s(1)-s(1)^3/3-s(2)+p.I_ext;
                            (s(1)+p.a-p.b*s(2))/p.tau];
S(5).ic_fun = @(p) [0.1; 0.05];
S(5).libid = 'fhn'; S(5).lib_ncols = 8;
S(5).coupling_site = 'x_eq'; S(5).multi_fp = false;

% ── 6. ROSENZWEIG-MACARTHUR (new — saturating-limit-cycle regime) ──────
% dx/dt = r*x*(1-x/K) - a*x*y/(1+a*h*x)     [prey]
% dy/dt = e*a*x*y/(1+a*h*x) - m*y           [predator]
S(6).name = 'rosenzweig_macarthur';
S(6).psamp = @() struct('r',0.8+0.6*rand(),'K',8+6*rand(), ...
    'a_rate',0.4+0.5*rand(),'h',0.3+0.5*rand(),'e',0.4+0.4*rand(), ...
    'm',0.2+0.3*rand(), ...
    'hill_k0',1+2*rand(),'hill_n',round(2+3*rand()), ...
    'hill_km',0.5+1.5*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
S(6).odefn_base = @(t,s,p) [ ...
    p.r*s(1)*(1-s(1)/p.K) - p.a_rate*s(1)*s(2)/(1+p.a_rate*p.h*s(1)); ...
    p.e*p.a_rate*s(1)*s(2)/(1+p.a_rate*p.h*s(1)) - p.m*s(2)];
S(6).ic_fun = @(p) [p.K*0.5; p.K*0.2];
S(6).libid = 'rma'; S(6).lib_ncols = 8;
S(6).coupling_site = 'x_eq'; S(6).multi_fp = false;

% ── 7. GENETIC TOGGLE SWITCH (new — bistable regime, multi-fixed-point) ─
% dx/dt = alpha1/(1+y^beta) - x
% dy/dt = alpha2/(1+x^gamma) - y
S(7).name = 'toggle_switch';
S(7).psamp = @() struct('alpha1',8+8*rand(),'alpha2',8+8*rand(), ...
    'beta',1.5+1.0*rand(),'gamma',1.5+1.0*rand(), ...
    'hill_k0',1+2*rand(),'hill_n',round(2+3*rand()), ...
    'hill_km',0.5+1.5*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.05*rand());
S(7).odefn_base = @(t,s,p) [p.alpha1/(1+s(2)^p.beta) - s(1);
                            p.alpha2/(1+s(1)^p.gamma) - s(2)];
% IC sampled per-draw near one of the two expected basins (upper-left or
% lower-right of phase plane) — randomized per attempt in the gen script
% so both stable branches get represented across the dataset.
S(7).ic_fun = @(p) [];   % filled in by caller (needs randomized basin choice)
S(7).libid = 'toggle'; S(7).lib_ncols = 6;
S(7).coupling_site = 'x_eq'; S(7).multi_fp = true;

end

function p=sample_brussel(hill_rep,hill_deg)
    max_tries = 1000; a = 1; b = 5;
    for k=1:max_tries
        a=1+2*rand(); b=4+6*rand();
        if b>a^2+1; break; end
    end
    p=struct('a',a,'b',b,'hill_k0',1.5+2.5*rand(),'hill_n',round(2+3*rand()), ...
        'hill_km',0.5+1.5*rand(),'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
end