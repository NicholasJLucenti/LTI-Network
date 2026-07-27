clear; close all; clc;
addpath(genpath('.'));
warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');

% ── CONFIG ──────────────────────────────────────────────────────────────
DATA_ROOT = 'C:/Users/nickj/LTInetV6 Local Data Drive';
if ~exist(DATA_ROOT,'dir'); mkdir(DATA_ROOT); end

dt = 0.05; N = 300; sg_p=3; sg_f=11; n_per_config = 100; base_seed = 20260;
ridge_lambda=0.05; sparsity_thresh=0.02; n_stridge_iters=10;
min_rel_amp=0.08; min_transient_ratio=0.3;
MAX_ATTEMPT_SECONDS = 5;  % kill a single ode15s attempt if it grinds past this

NULL_LAT = 5; NULL_COUP = 3;

hill_rep = @(z,k,n)  k.^n ./ (k.^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);

% ── LATENT / COUPLING DEFINITIONS (unchanged from V5) ────────────────────
% Lat0: dI/dt = kp*x - kd*I
% Lat1: dI/dt = kp*y - (kd + km*x)*I
% Lat2: dI/dt = kp*x*Hrep(x,k0,n) - kd*I
% Lat3: dI/dt = kp*x*(1-I) - kcat*I/(Km+I)      [Imax=1]
% Lat4: dI/dt = kp*Hrep(y,k0,n) - kd*I
%
% Ch0: -alpha_c * Hrep(I,k0_c,n_c)   (Hill repressor)
% Ch1: -beta_c  * I                 (additive)
% Ch2: -gamma_c * I * x             (multiplicative)

active_latents    = [0,1,2,3,4];
coupling_channels = [0,1,2];

S = V6_SystemLib();
n_systems = length(S);

fprintf('=== V6 Generation: %d systems x %d latents x %d couplings + null ===\n', ...
    n_systems, length(active_latents), length(coupling_channels));

total_saved = 0; total_skipped = 0;

for sys_idx = 1:n_systems
    sysdef = S(sys_idx);

    % Common ODE integration settings, tuned per libid stiffness
    if strcmp(sysdef.libid,'toggle')
        % Loosest tolerance of any system — rational Hill-like nonlinearity
        % near s=0 boundary is the stiffest case in the roster. Also caps
        % MaxStep tighter than dt so the solver is forced to take small,
        % checkable steps rather than attempting a large step into a
        % near-singular region and stalling in odenumjac.
        %
        % NOTE: two separate opts_ode variants are needed here because the
        % NonNegative index list must match actual state dimension at each
        % call site — the latent/coupling loop integrates a 3-state system
        % ([x,y,I]) while the null-class loop integrates only the 2-state
        % base system ([x,y], no latent I appended). Using [1,2,3] against
        % a 2-state ODE was the actual cause of null-class generation
        % failing 100% of the time while the isolated diagnostic (correctly
        % using [1,2]) passed — a real code/test mismatch, not a filter or
        % forcing amplitude problem.
        opts_ode_3state = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt/2, ...
                          'NonNegative',[1,2,3]);
        opts_ode_2state = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt/2, ...
                          'NonNegative',[1,2]);
        opts_ode = opts_ode_3state;  % default for latent/coupling loop below
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
        for ch = coupling_channels

            save_dir = sprintf('%s/V6Lat%d_Ch%d_Sys%d_NNdata', ...
                DATA_ROOT, lat, ch, sys_idx);
            if ~exist(save_dir,'dir'); mkdir(save_dir); end

            fprintf('\n--- %s | Lat%d | Ch%d ---\n', sysdef.name, lat, ch);

            rng(base_seed + sys_idx*10000 + lat*100 + ch);
            saved = 0; skipped = 0; attempt = 0;

            while saved < n_per_config && attempt < n_per_config*40
                attempt = attempt + 1;
                p = sysdef.psamp();

                % Sample latent-equation free parameters
                p.kp = 0.5+2*rand(); p.kd = 0.2+1.8*rand();
                p.km_lat = 0.1+1.9*rand();               % Lat1 titration rate
                p.kcat = 0.3+1.7*rand(); p.Km_lat = 0.3+1.7*rand(); % Lat3
                p.lat_hill_k0 = 0.5+2.5*rand(); p.lat_hill_n = round(2+4*rand());

                % Sample coupling free parameters
                p.alpha_c = 0.3+1.7*rand();   % Ch0
                p.beta_c  = 0.3+1.7*rand();   % Ch1
                p.gamma_c = 0.1+0.9*rand();   % Ch2
                p.coup_k0 = 0.5+2.5*rand(); p.coup_n = round(2+4*rand());

                % IC — toggle switch needs randomized basin choice
                if sysdef.multi_fp
                    if rand() < 0.5
                        ic2 = [p.alpha1*0.8; 0.5];   % near x-high basin
                    else
                        ic2 = [0.5; p.alpha2*0.8];   % near y-high basin
                    end
                else
                    ic2 = sysdef.ic_fun(p);
                end
                ic3 = [ic2; 0.3];  % append latent I initial condition

                t_start = tic;
                odefn3 = @(t,s,p) v6_coupled_ode(t,s,p,sysdef,lat,ch,hill_rep,hill_deg,t_start,MAX_ATTEMPT_SECONDS);

                try
                    [t_ode,S3] = ode15s(@(t,s) odefn3(t,s,p), t_span, ic3, opts_ode);
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

                % Save true per-file params (Hill constants + all sampled
                % latent/coupling parameters) so any downstream annexation
                % script recovers real values instead of hardcoded fallbacks
                % — this is the exact bug found in V5_annex_full.m/null gen,
                % now fixed uniformly for BOTH null and non-null classes.
                savepack.params_saved    = p;
                savepack.structure_label = lat;
                savepack.coupling_label  = ch;
                savepack.system_name     = sysdef.name;

                fname = fullfile(save_dir, sprintf('example_%04d.mat', saved+1));
                save(fname, '-struct', 'savepack');

                saved = saved+1;
                if mod(saved,50)==0
                    fprintf('  %d/%d (skip=%d)\n', saved, n_per_config, skipped);
                end
            end
            fprintf('[%s|Lat%d|Ch%d] saved=%d skipped=%d\n', sysdef.name, lat, ch, saved, skipped);
            total_saved = total_saved + saved; total_skipped = total_skipped + skipped;
        end
    end

    % ═══════════════════════════════════════════════════════════════
    %  NULL CLASS (Lat5/Ch3) — uncoupled base dynamics, no latent I
    % ═══════════════════════════════════════════════════════════════
    save_dir = sprintf('%s/V6Lat%d_Ch%d_Sys%d_NNdata', DATA_ROOT, NULL_LAT, NULL_COUP, sys_idx);
    if ~exist(save_dir,'dir'); mkdir(save_dir); end
    fprintf('\n--- %s | NULL ---\n', sysdef.name);

    rng(base_seed + sys_idx*10000 + 9999);
    saved = 0; skipped = 0; attempt = 0;

    while saved < n_per_config && attempt < n_per_config*40
        attempt = attempt + 1;
        p = sysdef.psamp();
        p.lat_hill_k0 = 0.5+2.5*rand(); p.lat_hill_n = round(2+4*rand()); % unused, kept for save consistency

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
                % Bistable systems settle to a hard fixed point with zero
                % intrinsic variance — no oscillation exists to satisfy the
                % informativeness filters below. Add a small persistent
                % forcing term so the trajectory wanders near its basin.
                %
                % IMPORTANT: forcing must be MULTIPLICATIVE (proportional to
                % current state), not additive. Additive forcing
                % (amp*sin(...)) can push the derivative negative right as
                % state approaches the s=0 boundary, which conflicts with
                % the 'NonNegative' ode15s constraint and causes the solver
                % to silently truncate integration early (returns far fewer
                % than N+201 points with no thrown error — this was the
                % actual cause of the 0-saved result: 68% of attempts were
                % truncated mid-integration, not flatlined or genuinely
                % failing). Multiplicative forcing vanishes as s->0, so it
                % never fights the non-negativity constraint.
                forcing_amp = 0.04;
                forcing_freq1 = 0.3 + 0.4*rand();
                forcing_freq2 = 0.5 + 0.6*rand();
                odefn_forced = @(t,s,p) v6_watchdog_wrap( ...
                    sysdef.odefn_base(t,s,p) + forcing_amp * [s(1)*sin(forcing_freq1*t); s(2)*cos(forcing_freq2*t)], ...
                    t_start_null, MAX_ATTEMPT_SECONDS);
                null_opts = opts_ode;
                if strcmp(sysdef.libid,'toggle')
                    null_opts = opts_ode_2state;
                end
                [t_ode,S2] = ode15s(@(t,s) odefn_forced(t,s,p), t_span, ic2, null_opts);
            else
                odefn_null = @(t,s,p) v6_watchdog_wrap( ...
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
            % Bistable near-fixed-point wander: only reject truly dead-flat
            % trajectories (numerical failure), don't require growing
            % transient/oscillatory structure — there isn't any by design.
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
            % Skip min_rel_amp / min_transient_ratio entirely for bistable
            % null — those filters assume oscillatory or decaying-spiral
            % dynamics (checked earlier for Goodwin) and will reject a
            % legitimate settled-with-small-wander trajectory by construction.
            % Minimal sanity check only: make sure it isn't perfectly constant.
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

        % IMPORTANT: save params_saved so annexation recovers TRUE Hill
        % params instead of the V5 hardcoded-fallback bug.
        savepack.params_saved    = p;
        savepack.structure_label = NULL_LAT;
        savepack.coupling_label  = NULL_COUP;
        savepack.system_name     = sysdef.name;

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

fprintf('\n=== V6 GENERATION COMPLETE ===\n');
fprintf('Total saved: %d   Total skipped: %d\n', total_saved, total_skipped);
fprintf('Run V6_annex_full.m next to compute all_new_features (88-dim) if not\n');
fprintf('already folded into this script''s savepack.\n');


%% ══════════════════════════════════════════════════════════════════════
%  HELPER: coupled ODE (base system + latent I + coupling injection)
%% ══════════════════════════════════════════════════════════════════════
function dsdt = v6_coupled_ode(t, s, p, sysdef, lat, ch, hill_rep, hill_deg, t_start, max_seconds)
    % Wall-clock watchdog: ode15s calls this function repeatedly during a
    % solve. If a single attempt has been running longer than max_seconds
    % (almost always a sign of odenumjac grinding on a near-singular
    % Jacobian, the same failure mode diagnosed earlier for the toggle
    % switch), throw an error here — ode15s propagates it up, and the
    % existing try/catch in the generation loop treats it as a normal
    % failed attempt (skipped, move on), rather than stalling indefinitely.
    if toc(t_start) > max_seconds
        error('V6:ODETimeout', 'Attempt exceeded %.0fs wall-clock limit', max_seconds);
    end

    % Clamp state to non-negative before evaluating base dynamics — several
    % systems (toggle switch: x^gamma/y^beta with non-integer exponents;
    % RMA: population densities) are only physically/numerically valid for
    % s>=0. Without this clamp, coupling terms (esp. Ch1/Ch2 subtracting
    % I from dx/dt) can push x negative, producing complex derivatives that
    % make ode15s's internal Jacobian estimation (odenumjac) hang or
    % degrade to extremely small steps.
    if sysdef.multi_fp
        s_eval = max(s, 0);
    else
        s_eval = s;
    end
    xy = s_eval(1:2); I = s_eval(3);
    base = sysdef.odefn_base(t, xy, p);

    % ── Latent equation for dI/dt ──
    switch lat
        case 0
            dI = p.kp*xy(1) - p.kd*I;
        case 1
            dI = p.kp*xy(2) - (p.kd + p.km_lat*xy(1))*I;
        case 2
            dI = p.kp*xy(1)*hill_rep(xy(1), p.lat_hill_k0, p.lat_hill_n) - p.kd*I;
        case 3
            I_clamped = max(min(I,1),0);
            dI = p.kp*xy(1)*(1-I_clamped) - p.kcat*I_clamped/(p.Km_lat+I_clamped+1e-8);
        case 4
            dI = p.kp*hill_rep(xy(2), p.lat_hill_k0, p.lat_hill_n) - p.kd*I;
        otherwise
            dI = -I; % should not happen
    end

    % ── Coupling injection into x-equation ──
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

    % Boundary reflection only applies where the clamp itself applies
    % (multi_fp systems) — do not zero derivatives for systems that were
    % never clamped in the first place.
    if sysdef.multi_fp
        if s(1) <= 0 && dx < 0; dx = 0; end
        dy = base(2);
        if s(2) <= 0 && dy < 0; dy = 0; end
        if s(3) <= 0 && dI < 0; dI = 0; end
    else
        dy = base(2);
    end

    dsdt = [dx; dy; dI];
end


function dsdt_out = v6_watchdog_wrap(dsdt_in, t_start, max_seconds)
    % Lightweight timeout check for the null-class loop, which calls
    % sysdef.odefn_base directly rather than through v6_coupled_ode.
    % Same mechanism: throw once the wall-clock budget is exceeded, so
    % ode15s aborts and the existing catch block treats it as a normal
    % failed attempt instead of stalling.
    if toc(t_start) > max_seconds
        error('V6:ODETimeout', 'Attempt exceeded %.0fs wall-clock limit', max_seconds);
    end
    dsdt_out = dsdt_in;
end