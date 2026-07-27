clear; close all; clc;
addpath(genpath('.'));
warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');

% ================================================================
% V9.1 PRODUCTION GENERATOR (NF-kB-tailored variant)
%
% Uses V9_1_SystemLib() (4 new NF-kB-relevant base systems: damped,
% dual-timescale, noise-induced/SDE, sustained) INSTEAD OF V6_SystemLib
% (the original 7 generic archetypes) -- this dataset does not mix the
% two libraries.
%
% Latent library (5 real classes, matching the already-established
% NoOvershoot decision -- see LTInetV9_DualBranch_EqGated_Sharpened_
% NoOvershoot.py's header for the numerical + biological rationale for
% dropping L_overshoot) + NULL:
%   0 = L_mirna    : dI/dt = kp*y - (kd+km*x)*I
%   1 = L_gk       : dI/dt = kp*x*(1-I/Imax) - kcat*I/(Km+I)
%   2 = L_yhill    : dI/dt = kp*Hrep(y,k0,n) - kd*I
%   3 = L_bistable : dI/dt = kp*x + kfb*I^2/(k0^2+I^2) - kd*I
%   4 = L_delay    : dI/dt = kp*z - kd*I, dz/dt = kz*(x-z)
%   5 = (L_overshoot -- deliberately never generated, matching V9's
%        NoOvershoot precedent; raw label 5 kept unused for consistency
%        with the existing raw-label -> compact-label remap convention)
%   6 = NULL
%
% Coupling: same 4 channels x 2 eq-targets as V9 (Ch0-2 real + Ch3 null,
% Eq0=couples into dx, Eq1=couples into dy). NULL has no Eq split.
%
% SDE HANDLING: sysdef.is_sde (only true for System 3, nfkb_noiseinduced)
% routes generation through a custom Euler-Maruyama integrator
% (v9_1_sde_integrate) instead of ode15s, since ode15s solves ODEs, not
% SDEs. Both the per-latent loop AND the NULL-class loop check is_sde,
% so NULL examples for that system are ALSO genuinely noise-driven --
% otherwise "SDE-ness" itself would become a trivial giveaway that a
% latent is present, which would be a real methodological flaw.
% ================================================================
DATA_ROOT = 'C:/Users/nickj/LTInetV9.1 Local Data Drive';
if ~exist(DATA_ROOT,'dir'); mkdir(DATA_ROOT); end

dt = 0.05; N = 300; sg_p=3; sg_f=11; n_per_config = 500; base_seed = 40260;
ridge_lambda=0.05; sparsity_thresh=0.02; n_stridge_iters=10;
min_rel_amp=0.08; min_transient_ratio=0.3;
MAX_ATTEMPT_SECONDS = 8;

NULL_LAT = 6; NULL_COUP = 3;

hill_rep = @(z,k,n)  k.^n ./ (k.^n + z.^n);
hill_act = @(z,k,n)  z.^n ./ (k.^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);

active_latents = [0, 1, 2, 3, 4];   % L_overshoot (5) deliberately excluded
latent_names_v91 = {'L_mirna','L_gk','L_yhill','L_bistable','L_delay'};
coupling_channels = [0, 1, 2];
coupling_equations = [0, 1];

S = V9_1_SystemLib();
n_systems = length(S);

fprintf('=== V9.1 Production Generation: %d systems x %d latents x %d couplings x %d eq-targets + null ===\n', ...
    n_systems, length(active_latents), length(coupling_channels), length(coupling_equations));
fprintf('Latent library: %s\n', strjoin(latent_names_v91, ', '));
fprintf('Base systems: %s\n', strjoin({S.name}, ', '));

total_saved = 0; total_skipped = 0;

for sys_idx = 1:n_systems
    sysdef = S(sys_idx);

    if strcmp(sysdef.libid,'toggle')
        opts_ode_3state = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt/2, ...
                          'NonNegative',[1,2,3]);
        opts_ode_4state = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt/2, ...
                          'NonNegative',[1,2,3,4]);
        opts_ode_2state = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt/2, ...
                          'NonNegative',[1,2]);
        opts_ode = opts_ode_3state;
    elseif strcmp(sysdef.libid,'fhn')
        opts_ode = odeset('RelTol',1e-5,'AbsTol',1e-7,'MaxStep',dt);
    elseif strcmp(sysdef.name,'van_der_pol')
        opts_ode = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt);
    else
        opts_ode = odeset('RelTol',1e-5,'AbsTol',1e-7,'MaxStep',dt);
    end

    t_end = (N+200)*dt; t_span = 0:dt:t_end;
    n_steps_sde = round(t_end/dt);   % for the Euler-Maruyama path

    for lat = active_latents
        lat_name = latent_names_v91{lat+1};
        needs_extra_state = (lat == 4);  % L_delay needs the relay state z

        for ch = coupling_channels
            for eq = coupling_equations

            save_dir = sprintf('%s/V91Lat%d_Ch%d_Eq%d_Sys%d_NNdata', ...
                DATA_ROOT, lat, ch, eq, sys_idx);
            if ~exist(save_dir,'dir'); mkdir(save_dir); end

            existing_files = dir(fullfile(save_dir,'example_*.mat'));
            existing_count = length(existing_files);

            fprintf('\n--- %s | %s | Ch%d | Eq%d ---\n', sysdef.name, lat_name, ch, eq);
            if existing_count > 0
                fprintf('  Found %d existing files, targeting %d total\n', existing_count, n_per_config);
            end

            if existing_count >= n_per_config
                fprintf('  Already have %d/%d, skipping\n', existing_count, n_per_config);
                continue;
            end

            rng(base_seed + sys_idx*10000 + lat*100 + ch*10 + eq);
            saved = existing_count; skipped = 0; attempt = 0;

            while saved < n_per_config && attempt < (n_per_config-existing_count)*40
                attempt = attempt + 1;
                p = sysdef.psamp();

                p.kp = 0.5+2*rand(); p.kd = 0.2+1.8*rand();
                p.km_lat = 0.1+1.9*rand();          % L_mirna
                p.kcat = 0.3+1.7*rand();             % L_gk
                p.Imax = 0.8+0.4*rand();             % L_gk

                I_scale_est = (p.kp * 1.0) / p.kd;
                p.Km_lat = (0.15+0.35*rand()) * I_scale_est;         % L_gk
                p.lat_hill_k0 = 0.5+2.5*rand(); p.lat_hill_n = round(2+4*rand());  % L_yhill
                p.kfb_bistable = (0.5+1.5*rand()) * p.kp;            % L_bistable
                p.k0_bistable = (0.3+0.7*rand()) * I_scale_est;      % L_bistable
                p.kz_delay = 0.5+2.5*rand();                          % L_delay

                p.alpha_c = 0.3+1.7*rand();
                p.beta_c  = 0.3+1.7*rand();
                p.gamma_c = 0.1+0.9*rand();
                p.coup_k0 = 0.5+2.5*rand(); p.coup_n = round(2+4*rand());

                ic2 = sysdef.ic_fun(p);   % none of V9.1's 4 systems are multi_fp
                if needs_extra_state
                    ic_full = [ic2; 0.3; 0.3];
                else
                    ic_full = [ic2; 0.3];
                end

                if sysdef.is_sde
                    [ok_int, t_ode, S3] = v9_1_sde_integrate( ...
                        ic_full, dt, n_steps_sde, p, sysdef, lat, ch, eq, ...
                        hill_rep, hill_act, hill_deg);
                    if ~ok_int
                        skipped=skipped+1; continue;
                    end
                else
                    t_start = tic;
                    odefn_full = @(t,s,p) v9_1_coupled_ode(t,s,p,sysdef,lat,ch,eq, ...
                        hill_rep,hill_act,hill_deg,t_start,MAX_ATTEMPT_SECONDS);
                    try
                        [t_ode,S3] = ode15s(@(t,s) odefn_full(t,s,p), t_span, ic_full, opts_ode);
                    catch
                        skipped=skipped+1; continue;
                    end
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

                savepack.params_saved    = p;
                savepack.structure_label = lat;
                savepack.coupling_label  = ch;
                savepack.coupling_eq     = eq;
                savepack.system_name     = sysdef.name;
                savepack.latent_name     = lat_name;

                fname = fullfile(save_dir, sprintf('example_%04d.mat', saved+1));
                save(fname, '-struct', 'savepack');

                saved = saved+1;
                if mod(saved,50)==0
                    fprintf('  %d/%d (skip=%d)\n', saved, n_per_config, skipped);
                end
            end
            fprintf('[%s|%s|Ch%d|Eq%d] saved=%d skipped=%d\n', sysdef.name, lat_name, ch, eq, saved, skipped);
            total_saved = total_saved + saved; total_skipped = total_skipped + skipped;
            end
        end
    end

    % ================================================================
    %  NULL CLASS (Lat6/Ch3) — generated once per system, no Eq split
    % ================================================================
    save_dir = sprintf('%s/V91Lat%d_Ch%d_Sys%d_NNdata', DATA_ROOT, NULL_LAT, NULL_COUP, sys_idx);
    if ~exist(save_dir,'dir'); mkdir(save_dir); end
    fprintf('\n--- %s | NULL ---\n', sysdef.name);

    existing_files_null = dir(fullfile(save_dir,'example_*.mat'));
    existing_count_null = length(existing_files_null);
    if existing_count_null > 0
        fprintf('  Found %d existing files, targeting %d total\n', existing_count_null, n_per_config);
    end

    saved = existing_count_null; skipped = 0; attempt = 0;

    if existing_count_null >= n_per_config
        fprintf('  Already have %d/%d, skipping\n', existing_count_null, n_per_config);
    else

    rng(base_seed + sys_idx*10000 + 9999);

    while saved < n_per_config && attempt < (n_per_config-existing_count_null)*40
        attempt = attempt + 1;
        p = sysdef.psamp();
        p.kp = 0.5+2*rand(); p.kd = 0.2+1.8*rand();

        ic2 = sysdef.ic_fun(p);   % none of V9.1's 4 systems are multi_fp

        if sysdef.is_sde
            % NULL still uses the SAME Euler-Maruyama process-noise
            % integration, just with lat/ch/eq forced to "no latent, no
            % coupling" inside v9_1_sde_integrate (see helper below) --
            % deliberately, so "is this trajectory noise-driven" cannot
            % become a shortcut for "does it have a latent."
            [ok_int, t_ode, S2] = v9_1_sde_integrate( ...
                [ic2; 0], dt, n_steps_sde, p, sysdef, -1, -1, 0, ...
                hill_rep, hill_act, hill_deg);
            if ~ok_int
                skipped=skipped+1; continue;
            end
        else
            t_start_null = tic;
            try
                odefn_null = @(t,s,p) v9_watchdog_wrap( ...
                    sysdef.odefn_base(t,s,p), t_start_null, MAX_ATTEMPT_SECONDS);
                [t_ode,S2] = ode15s(@(t,s) odefn_null(t,s,p), t_span, ic2, opts_ode);
            catch
                skipped=skipped+1; continue;
            end
        end

        if any(~isfinite(S2(:))) || size(S2,1) < N+201
            skipped=skipped+1; continue;
        end

        t_uni = (0:dt:t_end)';
        xa = interp1(t_ode,S2(:,1),t_uni,'linear');
        ya = interp1(t_ode,S2(:,2),t_uni,'linear');
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

        savepack.params_saved    = p;
        savepack.structure_label = NULL_LAT;
        savepack.coupling_label  = NULL_COUP;
        savepack.coupling_eq     = -1;
        savepack.system_name     = sysdef.name;
        savepack.latent_name     = 'NULL';

        fname = fullfile(save_dir, sprintf('example_%04d.mat', saved+1));
        save(fname, '-struct', 'savepack');

        saved = saved+1;
        if mod(saved,50)==0
            fprintf('  %d/%d (skip=%d)\n', saved, n_per_config, skipped);
        end
    end
    end
    fprintf('[%s|NULL] saved=%d skipped=%d\n', sysdef.name, saved, skipped);
    total_saved = total_saved + saved; total_skipped = total_skipped + skipped;
end

fprintf('\n=== V9.1 PRODUCTION GENERATION COMPLETE ===\n');
fprintf('Total saved: %d   Total skipped: %d\n', total_saved, total_skipped);
fprintf('Run V9_1_annex_full.m next.\n');


%% ================================================================
%  HELPER: coupled ODE (deterministic path — Systems 1, 2, 4)
%  Identical logic to v9_coupled_ode in V9_ProductionGen.m, minus the
%  L_overshoot case (never dispatched here since active_latents excludes
%  5) and minus multi_fp handling (none of V9.1's systems are multi_fp).
%% ================================================================
function dsdt = v9_1_coupled_ode(t, s, p, sysdef, lat, ch, eq, hill_rep, hill_act, hill_deg, t_start, max_seconds)
    if toc(t_start) > max_seconds
        error('V9_1:ODETimeout', 'Attempt exceeded %.0fs wall-clock limit', max_seconds);
    end

    xy = s(1:2); I = s(3);
    has_z = length(s) >= 4;
    if has_z; z = s(4); else; z = 0; end
    dz = 0;

    base = sysdef.odefn_base(t, xy, p);
    dI = v9_1_latent_drift(lat, xy, I, z, p, hill_rep);
    if lat == 4
        dz = p.kz_delay*(xy(1) - z);
    end

    coup_term = v9_1_coupling_term(ch, eq, I, xy, p, hill_rep);

    if eq == 0
        dx = base(1) + coup_term;
        dy = base(2);
    else
        dx = base(1);
        dy = base(2) + coup_term;
    end

    if has_z
        dsdt = [dx; dy; dI; dz];
    else
        dsdt = [dx; dy; dI];
    end
end


%% ================================================================
%  HELPER: latent dI/dt dispatch, shared by both the deterministic
%  (ode15s) and stochastic (Euler-Maruyama) integration paths so the
%  two paths can never silently drift apart. lat==-1 means "no latent"
%  (used by NULL-class SDE generation) -> dI is just decay toward 0.
%% ================================================================
function dI = v9_1_latent_drift(lat, xy, I, z, p, hill_rep)
    switch lat
        case 0  % L_mirna
            dI = p.kp*xy(2) - (p.kd + p.km_lat*xy(1))*I;
        case 1  % L_gk
            I_clamped = max(min(I,p.Imax),0);
            dI = p.kp*xy(1)*(1-I_clamped/p.Imax) - p.kcat*I_clamped/(p.Km_lat+I_clamped+1e-8);
        case 2  % L_yhill
            dI = p.kp*hill_rep(xy(2),p.lat_hill_k0,p.lat_hill_n) - p.kd*I;
        case 3  % L_bistable
            dI = p.kp*xy(1) + p.kfb_bistable*I^2/(p.k0_bistable^2+I^2+1e-8) - p.kd*I;
        case 4  % L_delay
            dI = p.kp*z - p.kd*I;
        otherwise  % -1 or unrecognized: no latent (NULL)
            dI = -I;
    end
end


%% ================================================================
%  HELPER: coupling term dispatch, shared by both integration paths.
%  ch==-1 means "no coupling" (NULL) -> zero term regardless of eq.
%% ================================================================
function coup_term = v9_1_coupling_term(ch, eq, I, xy, p, hill_rep)
    switch ch
        case 0
            coup_term = -p.alpha_c * hill_rep(I, p.coup_k0, p.coup_n);
        case 1
            coup_term = -p.beta_c * I;
        case 2
            if eq == 0
                coup_term = -p.gamma_c * I * xy(1);
            else
                coup_term = -p.gamma_c * I * xy(2);
            end
        otherwise  % -1 or unrecognized: no coupling (NULL)
            coup_term = 0;
    end
end


%% ================================================================
%  HELPER: Euler-Maruyama stochastic integrator for is_sde systems
%  (System 3, nfkb_noiseinduced). Process noise (p.sde_sigma_x/y) is
%  added to dx, dy at every step; the latent state I (and relay z, if
%  present) stay deterministic — only the OBSERVED base system is
%  noise-driven, isolating that as the property distinguishing this
%  system from the other three, rather than conflating it with "the
%  latent's own dynamics are also stochastic."
%
%  lat==-1 / ch==-1 signal "no latent / no coupling" (NULL-class calls).
%  No watchdog timer here — a fixed-length EM loop is always bounded,
%  unlike an adaptive-step ODE solver that can genuinely get stuck.
%% ================================================================
function [ok, t_ode, S3] = v9_1_sde_integrate(ic_full, dt, n_steps, p, sysdef, lat, ch, eq, hill_rep, hill_act, hill_deg)
    n_dim = length(ic_full);
    S3 = zeros(n_steps+1, n_dim);
    S3(1,:) = ic_full(:)';
    has_z = n_dim >= 4;

    ok = true;
    for k = 1:n_steps
        s_now = S3(k,:)';
        xy = s_now(1:2); I = s_now(3);
        if has_z; z = s_now(4); else; z = 0; end

        base = sysdef.odefn_base(0, xy, p);
        dI = v9_1_latent_drift(lat, xy, I, z, p, hill_rep);
        dz = 0;
        if lat == 4
            dz = p.kz_delay*(xy(1) - z);
        end
        coup_term = v9_1_coupling_term(ch, eq, I, xy, p, hill_rep);

        if eq == 0
            dx = base(1) + coup_term;
            dy = base(2);
        else
            dx = base(1);
            dy = base(2) + coup_term;
        end

        noise_x = p.sde_sigma_x * sqrt(dt) * randn();
        noise_y = p.sde_sigma_y * sqrt(dt) * randn();

        s_next = s_now;
        s_next(1) = s_now(1) + dx*dt + noise_x;
        s_next(2) = s_now(2) + dy*dt + noise_y;
        s_next(3) = s_now(3) + dI*dt;
        if has_z
            s_next(4) = s_now(4) + dz*dt;
        end
        s_next = max(s_next, 0);   % keep concentrations non-negative

        if any(~isfinite(s_next))
            ok = false; return;
        end
        S3(k+1,:) = s_next';
    end

    t_ode = (0:n_steps)'*dt;
end


function dsdt_out = v9_watchdog_wrap(dsdt_in, t_start, max_seconds)
    if toc(t_start) > max_seconds
        error('V9_1:ODETimeout', 'Attempt exceeded %.0fs wall-clock limit', max_seconds);
    end
    dsdt_out = dsdt_in;
end