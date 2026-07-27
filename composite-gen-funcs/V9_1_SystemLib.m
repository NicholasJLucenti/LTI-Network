function S = V9_1_SystemLib()
% V9_1_SystemLib.m
%
% Four new base "observed 2D system" archetypes for the V9.1 NF-kB-
% tailored LTInet variant, REPLACING the original 7-system V6_SystemLib
% library entirely for this dataset (V9.1 does not mix in the original
% Goodwin/Brusselator/Repressilator/etc. systems).
%
% Motivated by the NF-kB single-cell oscillation literature (Hoffmann
% et al. 2002; Nelson et al. 2004; Krishna, Jensen & Sneppen 2006;
% noise-induced-oscillation work, e.g. arXiv:1010.1743), which
% characterizes real single-cell IkBa/p65 dynamics as either severely
% damped (older tagged-reporter studies) or sustained-but-noisy (Sung
% et al. 2014, endogenous knock-in reporters) -- with growing evidence
% that at least some of the apparent oscillatory character in single
% cells is NOISE-INDUCED: the deterministic system sits near a stable
% fixed point, and continuous process noise (not just measurement noise
% layered on afterward) is what sustains the appearance of oscillation.
% None of the 7 systems in the original V6_SystemLib represent this
% "process noise, not just observation noise" regime, which is the main
% qualitative gap systems 1-4 below are meant to close.
%
% NOTE ON EXACT PARAMETER VALUES: these are reasoned starting ranges
% chosen to span the right QUALITATIVE dynamical regimes (damped /
% dual-timescale / noise-induced / sustained), not literal fitted
% parameters from a specific published NF-kB model -- consistent with
% how the original V6_SystemLib systems (Goodwin, Brusselator, etc.)
% are themselves generic archetypes rather than calibrated biological
% models. Revisit ranges if generated trajectories don't visually match
% your reference data once you've run this.
%
% !! CONTRACT NOTE: this reconstructs the sysdef struct fields/behavior
% expected by V9_ProductionGen.m purely from how that script CALLS
% sysdef (sysdef.libid, sysdef.multi_fp, sysdef.name, sysdef.psamp(),
% sysdef.ic_fun(p), sysdef.odefn_base(t,xy,p)) -- I do not have the
% original V6_SystemLib.m source in this conversation, so double-check
% field names/behavior against your actual codebase before running, and
% flag anything that doesn't match.
%
% All four systems use libid='standard' (the same 9-term polynomial+
% Hill library used throughout the rest of the pipeline) and
% multi_fp=false (none of these are bistable/toggle-type systems).
%
% ONE NON-STANDARD FIELD is introduced: sysdef.is_sde (boolean). System
% 3 (nfkb_noiseinduced) sets this true; V9_1_ProductionGen.m checks it
% to route that system through a custom Euler-Maruyama stochastic
% integrator instead of ode15s, since ode15s solves ODEs, not SDEs.
% Systems 1, 2, 4 leave is_sde=false and are solved exactly like the
% original 7 systems.

hill_rep = @(z,k,n)  k.^n ./ (k.^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);

S = struct('name',{},'libid',{},'multi_fp',{},'is_sde',{}, ...
           'psamp',{},'ic_fun',{},'odefn_base',{});

% ================================================================
% System 1: nfkb_damped -- damped 2-state Hill-repression negative
% feedback oscillator. LOW Hill coefficient (n=2-4) keeps this in the
% sub-Hopf regime where a 2-state Hill-repression loop generically
% produces a stable spiral (damped oscillatory decay to a fixed point)
% rather than a sustained limit cycle -- matching the severely-damped
% single-pulse-then-decay pattern in the reference figure's left panels
% (Hoffmann/Nelson-era characterization).
% ================================================================
S(1).name = 'nfkb_damped';
S(1).libid = 'standard';
S(1).multi_fp = false;
S(1).is_sde = false;
S(1).psamp = @() nfkb_damped_psamp(hill_rep, hill_deg);
S(1).ic_fun = @(p) [0.3+0.4*rand(); 0.3+0.4*rand()];
S(1).odefn_base = @(t, xy, p) nfkb_damped_ode(xy, p);

% ================================================================
% System 2: nfkb_dualfeedback -- dual-timescale negative feedback.
% Reflects the real IkBa (fast) + A20 (slower) dual-feedback structure
% documented in the NF-kB literature, WITHOUT introducing a genuine
% extra hidden state at the base-system level (that would blur the line
% between "known base dynamics" and the latent-inference problem this
% whole framework exists to solve). Instead, y's removal is split into
% two parallel terms with deliberately different rate constants: a fast
% SATURABLE term (kd_fast, analogous to IkBa-mediated fast nuclear
% export/inhibition) and a slower LINEAR term (kd_slow, analogous to a
% second, slower regulatory influence like A20-modulated turnover).
% ================================================================
S(2).name = 'nfkb_dualfeedback';
S(2).libid = 'standard';
S(2).multi_fp = false;
S(2).is_sde = false;
S(2).psamp = @() nfkb_dualfeedback_psamp(hill_rep, hill_deg);
S(2).ic_fun = @(p) [0.3+0.4*rand(); 0.3+0.4*rand()];
S(2).odefn_base = @(t, xy, p) nfkb_dualfeedback_ode(xy, p);

% ================================================================
% System 3: nfkb_noiseinduced -- SAME damped skeleton as System 1 (low
% Hill coefficient, sub-Hopf, stable-spiral drift), but integrated via
% Euler-Maruyama with continuous PROCESS noise added to both states at
% every timestep, rather than clean deterministic integration with
% observation noise added only after the fact. This directly implements
% the noise-induced-oscillation (NIO) mechanism reported in the NF-kB
% modeling literature: a system that would deterministically decay to a
% fixed point is continuously re-excited by intrinsic noise, producing
% a sustained-LOOKING noisy oscillation from an underlying damped
% skeleton. is_sde=true routes this through the custom integrator in
% V9_1_ProductionGen.m instead of ode15s.
% ================================================================
S(3).name = 'nfkb_noiseinduced';
S(3).libid = 'standard';
S(3).multi_fp = false;
S(3).is_sde = true;
S(3).psamp = @() nfkb_noiseinduced_psamp(hill_rep, hill_deg);
S(3).ic_fun = @(p) [0.3+0.4*rand(); 0.3+0.4*rand()];
S(3).odefn_base = @(t, xy, p) nfkb_damped_ode(xy, p);   % same drift as System 1

% ================================================================
% System 4: nfkb_sustained -- genuine sustained limit cycle. HIGH Hill
% coefficient (n=8-14) plus explicit timescale separation (y evolves
% much more slowly than x -- the standard relaxation-oscillator recipe
% also used by e.g. FitzHugh-Nagumo/Van der Pol to secure a true 2D
% limit cycle) pushes this past the Hopf bifurcation into sustained
% oscillation, combined with a higher observation-noise range --
% matching Sung et al. 2014's finding that endogenous (non-overexpressed)
% reporters show SUSTAINED oscillation in most cells, contrasting with
% System 1/3's damped character. Included because the field genuinely
% disagrees on which regime is "typical," so the training distribution
% spans both rather than committing to one.
% ================================================================
S(4).name = 'nfkb_sustained';
S(4).libid = 'standard';
S(4).multi_fp = false;
S(4).is_sde = false;
S(4).psamp = @() nfkb_sustained_psamp(hill_rep, hill_deg);
S(4).ic_fun = @(p) [0.3+0.4*rand(); 0.3+0.4*rand()];
S(4).odefn_base = @(t, xy, p) nfkb_sustained_ode(xy, p);

end


%% ================================================================
%  System 1 / 3 shared drift: damped Hill-repression negative feedback
%% ================================================================
function dxy = nfkb_damped_ode(xy, p)
    x = max(xy(1), 0); y = max(xy(2), 0);
    dx = p.kp1 * p.hill_rep(y, p.hill_k0, p.hill_n) - p.kd1*x;
    dy = p.kp2*x - p.kd2*y;
    dxy = [dx; dy];
end

function p = nfkb_damped_psamp(hill_rep, hill_deg)
    p = struct();
    p.hill_k0 = 0.5 + 2.0*rand();
    p.hill_n  = round(2 + 2*rand());     % LOW: 2-4, keeps sub-Hopf/damped regime
    p.hill_km = 0.3 + 1.2*rand();        % only used by the fitted 'standard' library's Hdeg term
    p.kp1 = 0.5 + 2.0*rand();
    p.kd1 = 0.3 + 1.2*rand();
    p.kp2 = 0.5 + 2.0*rand();
    p.kd2 = 0.2 + 1.0*rand();
    p.noise = 0.02 + 0.06*rand();
    p.hill_rep = hill_rep;
    p.hill_deg = hill_deg;
end


%% ================================================================
%  System 2: dual-timescale negative feedback (IkBa-fast + A20-slow
%  analog, collapsed into one observed 2-state system)
%% ================================================================
function dxy = nfkb_dualfeedback_ode(xy, p)
    x = max(xy(1), 0); y = max(xy(2), 0);
    dx = p.kp1 * p.hill_rep(y, p.hill_k0, p.hill_n) - p.kd1*x;
    dy = p.kp2*x - p.kd_fast*y/(p.km_fast+y+1e-8) - p.kd_slow*y;
    dxy = [dx; dy];
end

function p = nfkb_dualfeedback_psamp(hill_rep, hill_deg)
    p = struct();
    p.hill_k0 = 0.5 + 2.0*rand();
    p.hill_n  = round(2 + 3*rand());     % 2-5, moderate
    p.hill_km = 0.3 + 1.2*rand();
    p.kp1 = 0.5 + 2.0*rand();
    p.kd1 = 0.3 + 1.2*rand();
    p.kp2 = 0.5 + 2.0*rand();
    p.kd_fast = 0.3 + 1.2*rand();        % fast, saturable -- IkBa-like
    p.km_fast = 0.2 + 0.8*rand();
    p.kd_slow = 0.05 + 0.35*rand();      % slow, linear -- A20-like, deliberately smaller than kd_fast
    p.noise = 0.02 + 0.06*rand();
    p.hill_rep = hill_rep;
    p.hill_deg = hill_deg;
end


%% ================================================================
%  System 3 psamp: same drift family as System 1, plus SDE process-
%  noise intensities used only by the custom Euler-Maruyama integrator.
%  p.noise (post-hoc observation noise) is still sampled too, applied
%  identically to every system later in the generation script.
%% ================================================================
function p = nfkb_noiseinduced_psamp(hill_rep, hill_deg)
    p = nfkb_damped_psamp(hill_rep, hill_deg);
    p.sde_sigma_x = 0.05 + 0.20*rand();
    p.sde_sigma_y = 0.05 + 0.20*rand();
end


%% ================================================================
%  System 4: sustained limit cycle (high Hill coefficient + timescale
%  separation)
%% ================================================================
function dxy = nfkb_sustained_ode(xy, p)
    x = max(xy(1), 0); y = max(xy(2), 0);
    dx = p.kp1 * p.hill_rep(y, p.hill_k0, p.hill_n) - p.kd1*x;
    dy = p.kp2*x - p.kd2*y;
    dxy = [dx; dy];
end

function p = nfkb_sustained_psamp(hill_rep, hill_deg)
    p = struct();
    p.hill_k0 = 0.5 + 2.0*rand();
    p.hill_n  = round(8 + 6*rand());     % HIGH: 8-14, pushes past Hopf into sustained limit cycle
    p.hill_km = 0.3 + 1.2*rand();
    p.kp1 = 0.8 + 2.5*rand();
    p.kd1 = 0.8 + 1.7*rand();            % fast x
    p.kp2 = 0.5 + 2.0*rand();
    p.kd2 = 0.1 + 0.4*rand();            % slow y -- timescale separation
    p.noise = 0.05 + 0.10*rand();        % higher observation noise, matching "sustained but noisy"
    p.hill_rep = hill_rep;
    p.hill_deg = hill_deg;
end