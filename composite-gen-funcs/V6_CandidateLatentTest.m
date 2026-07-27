clear; close all; clc;
addpath(genpath('.'));
warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');

% ── CONFIG ────────────────────────────────────────────────────────────────
% Wider candidate pool than the previous 6-class test. Organized by
% DYNAMICAL MOTIF, not by cosmetic equation differences, since the last
% two rounds showed that equations differing only in surface form but
% sharing a motif (self-limiting growth) stay mutually confusable no
% matter how their constants are tuned. Each candidate below is chosen
% to represent a genuinely distinct qualitative behavior for I:
%
%   200 = L_prod     : dI/dt = kp*x - kd*I                          [plain relaxation]
%   201 = L_mirna    : dI/dt = kp*y - (kd + km*x)*I                 [x-modulated decay rate]
%   202 = L_selfact  : dI/dt = kp*x*Hact(I,k0,n) - kd*I             [positive autoregulation]
%   203 = L_gk       : dI/dt = kp*x*(1-I/Imax) - kcat*I/(Km+I)      [self-limiting: dual saturation]
%   204 = L_thresh   : dI/dt = kp*switch(x) - kd*I                  [threshold/switch-like production]
%   205 = L_overshoot: dI/dt = kp*x - kp2*x^2 - kd*I                [overshoot-adaptation proxy]
%   206 = L_bistable : dI/dt = kp*x + kfb*I^2/(k0^2+I^2) - kd*I     [I-driven positive feedback]
%   207 = L_delay    : dI/dt = kp*z - kd*I, dz/dt = kz*(x-z)         [explicit relay delay]
%
% NOTE: 205 (overshoot) approximates a derivative-feedback term with a
% quadratic in x rather than true d(x)/dt, to avoid needing a
% derivative-dependent solver — a genuine simplification, flagged here
% rather than hidden.
% NOTE: 207 (delay) uses a 2-stage linear relay (x -> z -> I) as a
% smooth proxy for discrete delay, standard in biological modeling
% (avoids DDE solver complexity while still producing delay-like phase lag).

DATA_ROOT = 'C:/Users/nickj/LTInetV6 Local Data Drive/CandidateLatentSearch';
if ~exist(DATA_ROOT,'dir'); mkdir(DATA_ROOT); end

dt = 0.05; N = 300; sg_p=3; sg_f=11; n_per_config = 60; base_seed = 88000;
ridge_lambda=0.05; sparsity_thresh=0.02; n_stridge_iters=10;
min_rel_amp=0.08; min_transient_ratio=0.3;
MAX_ATTEMPT_SECONDS = 8;

hill_rep = @(z,k,n) k.^n ./ (k.^n + z.^n);
hill_act = @(z,k,n) z.^n ./ (k.^n + z.^n);
hill_deg = @(y,km)  y   ./ (km  + y);

CANDIDATE_LATENTS = [200, 201, 202, 203, 204, 205, 206, 207];
latent_search_names = {'L_prod','L_mirna','L_selfact','L_gk','L_thresh', ...
                        'L_overshoot','L_bistable','L_delay'};

TEST_COUPLING = 1;

S = V6_SystemLib();
test_sys_indices = [1, 2, 5];
test_sys_names = {S(test_sys_indices).name};

fprintf('=== Expanded candidate latent search ===\n');
fprintf('Systems: %s\n', strjoin(test_sys_names, ', '));
fprintf('Latents: %s\n', strjoin(latent_search_names, ', '));
fprintf('n_per_config = %d\n\n', n_per_config);

total_saved = 0; total_skipped = 0;

for si = 1:length(test_sys_indices)
    sys_idx = test_sys_indices(si);
    sysdef = S(sys_idx);
    opts_ode = odeset('RelTol',1e-5,'AbsTol',1e-7,'MaxStep',dt);
    t_end = (N+200)*dt; t_span = 0:dt:t_end;

    for lat_idx = 1:length(CANDIDATE_LATENTS)
        lat = CANDIDATE_LATENTS(lat_idx);
        lat_name = latent_search_names{lat_idx};

        needs_extra_state = (lat == 207);

        save_dir = sprintf('%s/SearchLat%d_Sys%d_NNdata', DATA_ROOT, lat, sys_idx);
        if ~exist(save_dir,'dir'); mkdir(save_dir); end
        fprintf('--- %s | %s ---\n', sysdef.name, lat_name);

        rng(base_seed + sys_idx*1000 + lat_idx*10);
        saved = 0; skipped = 0; attempt = 0;

        while saved < n_per_config && attempt < n_per_config*40
            attempt = attempt + 1;
            p = sysdef.psamp();

            p.kp = 0.5+2*rand(); p.kd = 0.2+1.8*rand();
            p.km_lat = 0.1+1.9*rand();
            p.kcat = 0.3+1.7*rand();
            p.Imax = 0.8+0.4*rand();
            p.lat_hill_k0 = 0.5+2.5*rand(); p.lat_hill_n = round(2+4*rand());

            I_scale_est = (p.kp * 1.0) / p.kd;
            p.Km_lat = (0.15+0.35*rand()) * I_scale_est;
            p.lat_hill_k0_act = (0.3+0.7*rand()) * I_scale_est;
            p.lat_hill_n_act = round(3+4*rand());

            p.x_thresh = -0.5 + 1.0*rand();
            p.thresh_steepness = 5+15*rand();

            p.kp2_overshoot = (0.1+0.4*rand()) * p.kp;

            p.kfb_bistable = (0.5+1.5*rand()) * p.kp;
            p.k0_bistable = (0.3+0.7*rand()) * I_scale_est;

            p.kz_delay = 0.5+2.5*rand();
            p.beta_c = 0.3+1.7*rand();

            if sysdef.multi_fp
                if rand() < 0.5
                    ic2 = [p.alpha1*0.8; 0.5];
                else
                    ic2 = [0.5; p.alpha2*0.8];
                end
            else
                ic2 = sysdef.ic_fun(p);
            end

            if needs_extra_state
                ic_full = [ic2; 0.3; 0.3];
            else
                ic_full = [ic2; 0.3];
            end

            t_start = tic;
            odefn_full = @(t,s,p) search_latent_ode(t,s,p,sysdef,lat,hill_rep,hill_act,hill_deg,t_start,MAX_ATTEMPT_SECONDS);

            try
                [t_ode,S3] = ode15s(@(t,s) odefn_full(t,s,p), t_span, ic_full, opts_ode);
            catch
                skipped=skipped+1; continue;
            end
            if any(~isfinite(S3(:))) || size(S3,1) < N+201
                skipped=skipped+1; continue;
            end

            t_uni = (0:dt:t_end)';
            xa = interp1(t_ode,S3(:,1),t_uni,'linear');
            ya = interp1(t_ode,S3(:,2),t_uni,'linear');
            n_all = length(t_uni);

            if any(~isfinite(xa))||any(~isfinite(ya))||max(abs(xa))>1e5||max(abs(ya))>1e5
                skipped=skipped+1; continue;
            end

            tail = round(0.4*N);
            if var(xa(N-tail+1:N))<1e-6 || std(diff(xa(1:N)))<1e-4
                skipped=skipped+1; continue;
            end

            xa = xa + p.noise*std(xa)*randn(n_all,1);
            ya = ya + p.noise*std(ya)*randn(n_all,1);
            x_data = xa(1:N); y_data = ya(1:N);

            if (max(x_data)-min(x_data))/(mean(abs(x_data))+1e-8) < min_rel_amp
                skipped=skipped+1; continue;
            end
            if var(x_data(1:round(N/2))) < min_transient_ratio*var(x_data(round(N/2):N))
                skipped=skipped+1; continue;
            end

            [ok, savepack] = V6_ProcessFeatures( ...
                x_data, y_data, xa, ya, N, n_all, dt, sg_p, sg_f, p, ...
                sysdef, ridge_lambda, sparsity_thresh, n_stridge_iters);
            if ~ok
                skipped=skipped+1; continue;
            end

            savepack.structure_label = lat;
            savepack.coupling_label  = TEST_COUPLING;
            savepack.system_name     = sysdef.name;
            savepack.latent_name     = lat_name;
            savepack.params_saved    = p;

            fname = fullfile(save_dir, sprintf('example_%04d.mat', saved+1));
            save(fname, '-struct', 'savepack');

            saved = saved+1;
        end
        fprintf('  saved=%d skipped=%d\n', saved, skipped);
        total_saved = total_saved + saved; total_skipped = total_skipped + skipped;
    end
end

fprintf('\n=== Expanded search generation complete ===\n');
fprintf('Total saved: %d   Total skipped: %d\n', total_saved, total_skipped);
fprintf('Run V6_LatentSearch_LDA.py next for pairwise screening + greedy selection.\n');


%% ══════════════════════════════════════════════════════════════════════
%  Expanded candidate ODE dispatch
%% ══════════════════════════════════════════════════════════════════════
function dsdt = search_latent_ode(t, s, p, sysdef, lat, hill_rep, hill_act, hill_deg, t_start, max_seconds)
    if toc(t_start) > max_seconds
        error('V6:ODETimeout', 'Attempt exceeded %.0fs wall-clock limit', max_seconds);
    end

    if sysdef.multi_fp
        s_eval = max(s, 0);
    else
        s_eval = s;
    end
    xy = s_eval(1:2); I = s_eval(3);
    has_z = length(s) >= 4;
    if has_z; z = s_eval(4); else; z = 0; end
    dz = 0;

    base = sysdef.odefn_base(t, xy, p);

    switch lat
        case 200  % L_prod
            dI = p.kp*xy(1) - p.kd*I;
        case 201  % L_mirna
            dI = p.kp*xy(2) - (p.kd + p.km_lat*xy(1))*I;
        case 202  % L_selfact — positive autoregulation
            I_pos = max(I,0);
            dI = p.kp*xy(1)*hill_act(I_pos,p.lat_hill_k0_act,p.lat_hill_n_act) - p.kd*I;
        case 203  % L_gk — self-limiting dual saturation
            I_clamped = max(min(I,p.Imax),0);
            dI = p.kp*xy(1)*(1-I_clamped/p.Imax) - p.kcat*I_clamped/(p.Km_lat+I_clamped+1e-8);
        case 204  % L_thresh — smoothed threshold/switch production
            switch_on = 0.5*(1+tanh(p.thresh_steepness*(xy(1)-p.x_thresh)));
            dI = p.kp*switch_on - p.kd*I;
        case 205  % L_overshoot — quadratic-in-x proxy for derivative feedback
            dI = p.kp*xy(1) - p.kp2_overshoot*xy(1)^2 - p.kd*I;
        case 206  % L_bistable — I-driven positive feedback
            dI = p.kp*xy(1) + p.kfb_bistable*I^2/(p.k0_bistable^2+I^2+1e-8) - p.kd*I;
        case 207  % L_delay — 2-stage relay (x -> z -> I) as smooth delay proxy
            dz = p.kz_delay*(xy(1) - z);
            dI = p.kp*z - p.kd*I;
        otherwise
            dI = -I;
    end

    coup_term = -p.beta_c * I;
    dx = base(1) + coup_term;

    if sysdef.multi_fp
        if s(1) <= 0 && dx < 0; dx = 0; end
        dy = base(2);
        if s(2) <= 0 && dy < 0; dy = 0; end
        if s(3) <= 0 && dI < 0; dI = 0; end
    else
        dy = base(2);
    end

    if has_z
        dsdt = [dx; dy; dI; dz];
    else
        dsdt = [dx; dy; dI];
    end
end