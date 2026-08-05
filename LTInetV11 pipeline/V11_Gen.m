clear; close all; clc;
addpath(genpath('.'));
warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');

DATA_ROOT = 'C:/Users/nickj/LTInetV11 Local Data Drive';
if ~exist(DATA_ROOT,'dir'); mkdir(DATA_ROOT); end

dt = 0.05; N = 300; sg_p=3; sg_f=11; n_per_config = 300; base_seed = 40260;
ridge_lambda=0.05; sparsity_thresh=0.02; n_stridge_iters=10;
min_rel_amp=0.08; min_transient_ratio=0.3;
MAX_ATTEMPT_SECONDS = 5;

NULL_LAT = 6; NULL_COUP = 3;

hill_rep = @(z,k,n)  k.^n ./ (k.^n + z.^n);
hill_act = @(z,k,n)  z.^n ./ (k.^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);

active_latents = [0, 1, 2, 3, 4, 5];
latent_names_v11 = {'L_mirna','L_gk','L_yhill','L_bistable','L_delay','L_overshoot'};
coupling_channels = [0, 1, 2];
coupling_equations = [0, 1];

S = V11_SystemLib();
n_systems = length(S);

for sys_idx = 1:n_systems
    sysdef = S(sys_idx);

    if strcmp(sysdef.libid,'toggle')
        opts_ode_3state = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt/2,'NonNegative',[1,2,3]);
        opts_ode_4state = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt/2,'NonNegative',[1,2,3,4]);
        opts_ode_2state = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt/2,'NonNegative',[1,2]);
        opts_ode = opts_ode_3state;
    elseif strcmp(sysdef.libid,'fhn')
        opts_ode = odeset('RelTol',1e-5,'AbsTol',1e-7,'MaxStep',dt);
    elseif strcmp(sysdef.name,'van_der_pol')
        opts_ode = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt);
    else
        opts_ode = odeset('RelTol',1e-5,'AbsTol',1e-7,'MaxStep',dt);
    end

    t_end = (N+200)*dt; t_span = 0:dt:t_end;

    for lat = active_latents
        lat_name = latent_names_v11{lat+1};
        needs_extra_state = (lat == 4);

        for ch = coupling_channels
            for eq = coupling_equations

            save_dir = sprintf('%s/V11Lat%d_Ch%d_Eq%d_Sys%d_NNdata', DATA_ROOT, lat, ch, eq, sys_idx);
            if ~exist(save_dir,'dir'); mkdir(save_dir); end

            existing_count = length(dir(fullfile(save_dir,'example_*.mat')));
            if existing_count >= n_per_config; continue; end

            rng(base_seed + sys_idx*10000 + lat*100 + ch*10 + eq);
            saved = existing_count; skipped = 0; attempt = 0;

            while saved < n_per_config && attempt < (n_per_config-existing_count)*40
                attempt = attempt + 1;
                p = sysdef.psamp();

                p.kp = 0.5+2*rand(); p.kd = 0.2+1.8*rand();
                p.km_lat = 0.1+1.9*rand();
                p.kcat = 0.3+1.7*rand();
                p.Imax = 0.8+0.4*rand();

                I_scale_est = (p.kp * 1.0) / p.kd;
                p.Km_lat = (0.15+0.35*rand()) * I_scale_est;
                p.lat_hill_k0 = 0.5+2.5*rand(); p.lat_hill_n = round(2+4*rand());
                p.kfb_bistable = (0.5+1.5*rand()) * p.kp;
                p.k0_bistable = (0.3+0.7*rand()) * I_scale_est;
                p.kz_delay = 0.5+2.5*rand();
                p.kp2_overshoot = (0.1+0.4*rand()) * p.kp;

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
                odefn_full = @(t,s,p) v11_coupled_ode(t,s,p,sysdef,lat,ch,eq, ...
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

                [ok, savepack] = V11_ProcessFeatures( ...
                    x_data, y_data, xa, ya, N, n_all, dt, sg_p, sg_f, p, ...
                    sysdef, ridge_lambda, sparsity_thresh, n_stridge_iters);
                if ~ok
                    skipped=skipped+1; continue;
                end

                savepack.structure_label = lat;
                savepack.coupling_label  = ch;
                savepack.coupling_eq     = eq;
                savepack.system_name     = sysdef.name;
                savepack.latent_name     = lat_name;
                savepack.params_saved    = p;
                savepack.params_numeric  = strip_handle_fields(p);

                fname = fullfile(save_dir, sprintf('example_%04d.mat', saved+1));
                save(fname, '-struct', 'savepack');
                saved = saved+1;
            end
            end
        end
    end

    save_dir = sprintf('%s/V11Lat%d_Ch%d_SysNull_%d_NNdata', DATA_ROOT, NULL_LAT, NULL_COUP, sys_idx);
    if ~exist(save_dir,'dir'); mkdir(save_dir); end
    existing_count_null = length(dir(fullfile(save_dir,'example_*.mat')));
    saved = existing_count_null; skipped = 0; attempt = 0;

    if existing_count_null < n_per_config

    rng(base_seed + sys_idx*10000 + 9999);

    while saved < n_per_config && attempt < (n_per_config-existing_count_null)*40
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
                odefn_forced = @(t,s,p) v11_watchdog_wrap( ...
                    sysdef.odefn_base(t,s,p) + forcing_amp * [s(1)*sin(forcing_freq1*t); s(2)*cos(forcing_freq2*t)], ...
                    t_start_null, MAX_ATTEMPT_SECONDS);
                null_opts = opts_ode;
                if strcmp(sysdef.libid,'toggle')
                    null_opts = opts_ode_2state;
                end
                [t_ode,S2] = ode15s(@(t,s) odefn_forced(t,s,p), t_span, ic2, null_opts);
            else
                odefn_null = @(t,s,p) v11_watchdog_wrap( ...
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

        [ok, savepack] = V11_ProcessFeatures( ...
            x_data, y_data, xa, ya, N, n_all, dt, sg_p, sg_f, p, ...
            sysdef, ridge_lambda, sparsity_thresh, n_stridge_iters);
        if ~ok
            skipped=skipped+1; continue;
        end

        savepack.params_saved    = p;
        savepack.params_numeric  = strip_handle_fields(p);
        savepack.structure_label = NULL_LAT;
        savepack.coupling_label  = NULL_COUP;
        savepack.coupling_eq     = -1;
        savepack.system_name     = sysdef.name;
        savepack.latent_name     = 'NULL';

        fname = fullfile(save_dir, sprintf('example_%04d.mat', saved+1));
        save(fname, '-struct', 'savepack');
        saved = saved+1;
    end
    end
end


function dsdt = v11_coupled_ode(t, s, p, sysdef, lat, ch, eq, hill_rep, hill_act, hill_deg, t_start, max_seconds)
    if toc(t_start) > max_seconds
        error('V11:ODETimeout', 'Attempt exceeded %.0fs wall-clock limit', max_seconds);
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
        case 0
            dI = p.kp*xy(2) - (p.kd + p.km_lat*xy(1))*I;
        case 1
            I_clamped = max(min(I,p.Imax),0);
            dI = p.kp*xy(1)*(1-I_clamped/p.Imax) - p.kcat*I_clamped/(p.Km_lat+I_clamped+1e-8);
        case 2
            dI = p.kp*hill_rep(xy(2),p.lat_hill_k0,p.lat_hill_n) - p.kd*I;
        case 3
            dI = p.kp*xy(1) + p.kfb_bistable*I^2/(p.k0_bistable^2+I^2+1e-8) - p.kd*I;
        case 4
            dz = p.kz_delay*(xy(1) - z);
            dI = p.kp*z - p.kd*I;
        case 5
            dI = p.kp*xy(1) - p.kp2_overshoot*xy(1)^2 - p.kd*I;
        otherwise
            dI = -I;
    end

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
        otherwise
            coup_term = 0;
    end

    if eq == 0
        dx = base(1) + coup_term;
        dy = base(2);
    else
        dx = base(1);
        dy = base(2) + coup_term;
    end

    if sysdef.multi_fp
        if s(1) <= 0 && dx < 0; dx = 0; end
        if s(2) <= 0 && dy < 0; dy = 0; end
        if s(3) <= 0 && dI < 0; dI = 0; end
    end

    if has_z
        dsdt = [dx; dy; dI; dz];
    else
        dsdt = [dx; dy; dI];
    end
end


function dsdt_out = v11_watchdog_wrap(dsdt_in, t_start, max_seconds)
    if toc(t_start) > max_seconds
        error('V11:ODETimeout', 'Attempt exceeded %.0fs wall-clock limit', max_seconds);
    end
    dsdt_out = dsdt_in;
end


function pn = strip_handle_fields(p)
    HANDLE_FIELDS = {'hill_rep','hill_deg','hill_act'};
    pn = p;
    for h = 1:length(HANDLE_FIELDS)
        fn = HANDLE_FIELDS{h};
        if isfield(pn, fn)
            pn = rmfield(pn, fn);
        end
    end
end
