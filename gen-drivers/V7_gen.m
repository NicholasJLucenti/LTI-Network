clear; close all; clc;
addpath(genpath('.'));
warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');

% ── CONFIG ──────────────────────────────────────────────────────────────
% V7: finalized 6-class latent library, validated via V6_LatentSearch_LDA.py
% (all pairwise CV accuracy >= 0.658). L_prod was dropped after search —
% every pair involving it and {L_overshoot, L_bistable, L_delay} landed
% WEAK, so the greedy maximum-subset search excluded it rather than
% accept a weaker overall library.
DATA_ROOT = 'C:/Users/nickj/LTInetV7 Local Data Drive';
if ~exist(DATA_ROOT,'dir'); mkdir(DATA_ROOT); end

dt = 0.05; N = 300; sg_p=3; sg_f=11; n_per_config = 295; base_seed = 30260;
ridge_lambda=0.05; sparsity_thresh=0.02; n_stridge_iters=10;
min_rel_amp=0.08; min_transient_ratio=0.3;
MAX_ATTEMPT_SECONDS = 8;

NULL_LAT = 6; NULL_COUP = 3;  % null latent index shifted to 6 (0-5 = the 6 real latents)

hill_rep = @(z,k,n)  k.^n ./ (k.^n + z.^n);
hill_act = @(z,k,n)  z.^n ./ (k.^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);

% ── FINALIZED V7 LATENT LIBRARY ──────────────────────────────────────────
%   0 = L_mirna    : dI/dt = kp*y - (kd + km*x)*I
%   1 = L_selfact  : dI/dt = kp*x*Hact(I,k0,n) - kd*I
%   2 = L_gk       : dI/dt = kp*x*(1-I/Imax) - kcat*I/(Km+I)
%   3 = L_overshoot: dI/dt = kp*x - kp2*x^2 - kd*I
%   4 = L_bistable : dI/dt = kp*x + kfb*I^2/(k0^2+I^2) - kd*I
%   5 = L_delay    : dI/dt = kp*z - kd*I,  dz/dt = kz*(x-z)
%   6 = NULL       : uncoupled base dynamics, no latent
active_latents = [0, 1, 2, 3, 4, 5];
latent_names_v7 = {'L_mirna','L_selfact','L_gk','L_overshoot','L_bistable','L_delay'};
coupling_channels = [0, 1, 2];

S = V6_SystemLib();
n_systems = length(S);

fprintf('=== V7 Generation: %d systems x %d latents x %d couplings + null ===\n', ...
    n_systems, length(active_latents), length(coupling_channels));
fprintf('Latent library: %s\n', strjoin(latent_names_v7, ', '));

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

    % ═══════════════════════════════════════════════════════════════
    %  LATENT + COUPLING CLASSES
    % ═══════════════════════════════════════════════════════════════
    for lat = active_latents
        lat_name = latent_names_v7{lat+1};
        needs_extra_state = (lat == 5);  % L_delay needs the relay state z

        for ch = coupling_channels

            save_dir = sprintf('%s/V7Lat%d_Ch%d_Sys%d_NNdata', ...
                DATA_ROOT, lat, ch, sys_idx);
            if ~exist(save_dir,'dir'); mkdir(save_dir); end

            fprintf('\n--- %s | %s | Ch%d ---\n', sysdef.name, lat_name, ch);

            rng(base_seed + sys_idx*10000 + lat*100 + ch);
            saved = 0; skipped = 0; attempt = 0;

            while saved < n_per_config && attempt < n_per_config*40
                attempt = attempt + 1;
                p = sysdef.psamp();

                p.kp = 0.5+2*rand(); p.kd = 0.2+1.8*rand();
                p.km_lat = 0.1+1.9*rand();
                p.kcat = 0.3+1.7*rand();
                p.Imax = 0.8+0.4*rand();

                I_scale_est = (p.kp * 1.0) / p.kd;
                p.Km_lat = (0.15+0.35*rand()) * I_scale_est;
                p.lat_hill_k0_act = (0.3+0.7*rand()) * I_scale_est;
                p.lat_hill_n_act = round(3+4*rand());
                p.kp2_overshoot = (0.1+0.4*rand()) * p.kp;
                p.kfb_bistable = (0.5+1.5*rand()) * p.kp;
                p.k0_bistable = (0.3+0.7*rand()) * I_scale_est;
                p.kz_delay = 0.5+2.5*rand();

                p.alpha_c = 0.3+1.7*rand();
                p.beta_c  = 0.3+1.7*rand();
                p.gamma_c = 0.1+0.9*rand();
                p.coup_k0 = 0.5+2.5*rand(); p.coup_n = round(2+4*rand());

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
                odefn_full = @(t,s,p) v7_coupled_ode(t,s,p,sysdef,lat,ch, ...
                    hill_rep,hill_act,hill_deg,t_start,MAX_ATTEMPT_SECONDS);

                solve_opts = opts_ode;
                if strcmp(sysdef.libid,'toggle') && needs_extra_state
                    solve_opts = opts_ode_4state;
                end

                try
                    [t_ode,S3] = ode15s(@(t,s) odefn_full(t,s,p), t_span, ic_full, solve_opts);
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

                savepack.params_saved    = p;
                savepack.structure_label = lat;
                savepack.coupling_label  = ch;
                savepack.system_name     = sysdef.name;
                savepack.latent_name     = lat_name;

                fname = fullfile(save_dir, sprintf('example_%04d.mat', saved+1));
                save(fname, '-struct', 'savepack');

                saved = saved+1;
                if mod(saved,50)==0
                    fprintf('  %d/%d (skip=%d)\n', saved, n_per_config, skipped);
                end
            end
            fprintf('[%s|%s|Ch%d] saved=%d skipped=%d\n', sysdef.name, lat_name, ch, saved, skipped);
            total_saved = total_saved + saved; total_skipped = total_skipped + skipped;
        end
    end

    % ═══════════════════════════════════════════════════════════════
    %  NULL CLASS (Lat6/Ch3) — uncoupled base dynamics, no latent
    % ═══════════════════════════════════════════════════════════════
    save_dir = sprintf('%s/V7Lat%d_Ch%d_Sys%d_NNdata', DATA_ROOT, NULL_LAT, NULL_COUP, sys_idx);
    if ~exist(save_dir,'dir'); mkdir(save_dir); end
    fprintf('\n--- %s | NULL ---\n', sysdef.name);

    rng(base_seed + sys_idx*10000 + 9999);
    saved = 0; skipped = 0; attempt = 0;

    while saved < n_per_config && attempt < n_per_config*40
        attempt = attempt + 1;
        p = sysdef.psamp();
        p.kp = 0.5+2*rand(); p.kd = 0.2+1.8*rand();

        if sysdef.multi_fp
            if rand() < 0.5
                ic2 = [p.alpha1*0.8; 0.5];
            else
                ic2 = [0.5; p.alpha2*0.8];
            end
        else
            ic2 = sysdef.ic_fun(p);
        end

        t_start_null = tic;
        try
            if sysdef.multi_fp
                forcing_amp = 0.04;
                forcing_freq1 = 0.3 + 0.4*rand();
                forcing_freq2 = 0.5 + 0.6*rand();
                odefn_forced = @(t,s,p) v7_watchdog_wrap( ...
                    sysdef.odefn_base(t,s,p) + forcing_amp * [s(1)*sin(forcing_freq1*t); s(2)*cos(forcing_freq2*t)], ...
                    t_start_null, MAX_ATTEMPT_SECONDS);
                null_opts = opts_ode;
                if strcmp(sysdef.libid,'toggle')
                    null_opts = opts_ode_2state;
                end
                [t_ode,S2] = ode15s(@(t,s) odefn_forced(t,s,p), t_span, ic2, null_opts);
            else
                odefn_null = @(t,s,p) v7_watchdog_wrap( ...
                    sysdef.odefn_base(t,s,p), t_start_null, MAX_ATTEMPT_SECONDS);
                [t_ode,S2] = ode15s(@(t,s) odefn_null(t,s,p), t_span, ic2, opts_ode);
            end
        catch
            skipped=skipped+1; continue;
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
        if sysdef.multi_fp
            if var(xa(N-tail+1:N)) < 1e-8
                skipped=skipped+1; continue;
            end
        else
            if var(xa(N-tail+1:N))<1e-6 || std(diff(xa(1:N)))<1e-4
                skipped=skipped+1; continue;
            end
        end

        xa = xa + p.noise*std(xa)*randn(n_all,1);
        ya = ya + p.noise*std(ya)*randn(n_all,1);
        x_data = xa(1:N); y_data = ya(1:N);

        if sysdef.multi_fp
            if std(x_data) < 1e-6
                skipped=skipped+1; continue;
            end
        else
            if (max(x_data)-min(x_data))/(mean(abs(x_data))+1e-8) < min_rel_amp
                skipped=skipped+1; continue;
            end
            if var(x_data(1:round(N/2))) < min_transient_ratio*var(x_data(round(N/2):N))
                skipped=skipped+1; continue;
            end
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
        savepack.system_name     = sysdef.name;
        savepack.latent_name     = 'NULL';

        fname = fullfile(save_dir, sprintf('example_%04d.mat', saved+1));
        save(fname, '-struct', 'savepack');

        saved = saved+1;
        if mod(saved,50)==0
            fprintf('  %d/%d (skip=%d)\n', saved, n_per_config, skipped);
        end
    end
    fprintf('[%s|NULL] saved=%d skipped=%d\n', sysdef.name, saved, skipped);
    total_saved = total_saved + saved; total_skipped = total_skipped + skipped;
end

fprintf('\n=== V7 GENERATION COMPLETE ===\n');
fprintf('Total saved: %d   Total skipped: %d\n', total_saved, total_skipped);
fprintf('Run a V7-adapted annexation script next.\n');


%% ══════════════════════════════════════════════════════════════════════
%  HELPER: coupled ODE (base system + latent I [+ delay relay z] + coupling)
%% ══════════════════════════════════════════════════════════════════════
function dsdt = v7_coupled_ode(t, s, p, sysdef, lat, ch, hill_rep, hill_act, hill_deg, t_start, max_seconds)
    if toc(t_start) > max_seconds
        error('V7:ODETimeout', 'Attempt exceeded %.0fs wall-clock limit', max_seconds);
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
        case 0  % L_mirna
            dI = p.kp*xy(2) - (p.kd + p.km_lat*xy(1))*I;
        case 1  % L_selfact — positive autoregulation
            I_pos = max(I,0);
            dI = p.kp*xy(1)*hill_act(I_pos,p.lat_hill_k0_act,p.lat_hill_n_act) - p.kd*I;
        case 2  % L_gk — self-limiting dual saturation
            I_clamped = max(min(I,p.Imax),0);
            dI = p.kp*xy(1)*(1-I_clamped/p.Imax) - p.kcat*I_clamped/(p.Km_lat+I_clamped+1e-8);
        case 3  % L_overshoot
            dI = p.kp*xy(1) - p.kp2_overshoot*xy(1)^2 - p.kd*I;
        case 4  % L_bistable — I-driven positive feedback
            dI = p.kp*xy(1) + p.kfb_bistable*I^2/(p.k0_bistable^2+I^2+1e-8) - p.kd*I;
        case 5  % L_delay — 2-stage relay
            dz = p.kz_delay*(xy(1) - z);
            dI = p.kp*z - p.kd*I;
        otherwise
            dI = -I;
    end

    switch ch
        case 0
            coup_term = -p.alpha_c * hill_rep(I, p.coup_k0, p.coup_n);
        case 1
            coup_term = -p.beta_c * I;
        case 2
            coup_term = -p.gamma_c * I * xy(1);
        otherwise
            coup_term = 0;
    end

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


function dsdt_out = v7_watchdog_wrap(dsdt_in, t_start, max_seconds)
    if toc(t_start) > max_seconds
        error('V7:ODETimeout', 'Attempt exceeded %.0fs wall-clock limit', max_seconds);
    end
    dsdt_out = dsdt_in;
end