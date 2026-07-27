clear; close all; clc;
addpath(genpath('.'));

warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');

% ── Shared Hill function handles ──────────────────────────────────────────
hill_rep = @(z,k,n)  k^n ./ (k^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);
hill_act = @(x,k,n)  x.^n ./ (k^n + x.^n);

dt            = 0.05;
N             = 300;
eta           = 0.6;
sg_p          = 3;
sg_f          = 11;

base_seed     = 300;
n_per_config  = 600;

% All six base systems
systems = {'goodwin', 'brusselator', 'repressilator', ...
           'van_der_pol', 'fitzhugh_nagumo', 'lotka_volterra'};

% New latent classes 4-7 (saved as 0-indexed: folders Lat4-7, labels 4-7)
% Folders named Lat4-7 (0-indexed, matching structure_label directly)
new_latent_classes = [4, 5, 6, 7];

for sys_idx = 1:length(systems)
    system_name = systems{sys_idx};

    for lc_idx = 1:length(new_latent_classes)
        latent_class = new_latent_classes(lc_idx);

        % Lat4/Goodwin (sys_idx=1, latent_class=4) skipped — generation
        % time too long; WeightedRandomSampler in Python corrects the bias
        if sys_idx == 1 && latent_class == 4
            fprintf('Skipping Lat5/Goodwin combination\n');
            continue
        end

        save_dir = sprintf( ...
            'C:/Users/nickj/MATLAB Drive/LTI Network/V2Lat%d_Sys%d_NNdata', ...
            latent_class, sys_idx);

        if ~exist(save_dir, 'dir'), mkdir(save_dir); end

        fprintf('\n=== V2 | %s | Latent Class %d ===\n', ...
            system_name, latent_class);

        switch system_name
            case 'goodwin'
                param_sampler = @() sample_goodwin(hill_rep, hill_deg, ...
                    hill_act, latent_class);
                ode_fun  = @(t,s,p) goodwin_ode(t,s,p,latent_class);
                ic_fun   = @(p) [1.5; 0.5; 1.0];
                lib_fun  = @(x,y,N,p) standard_lib(x,y,N,p);
                aux_funs = {@aux_x_standard, @aux_y_standard};

            case 'brusselator'
                param_sampler = @() sample_brusselator(hill_rep, hill_deg, ...
                    hill_act, latent_class);
                ode_fun  = @(t,s,p) brusselator_ode(t,s,p,latent_class);
                ic_fun   = @(p) [p.a; p.b/p.a; 0.5];
                lib_fun  = @(x,y,N,p) brusselator_lib(x,y,N,p);
                aux_funs = {@aux_x_bruss, @aux_y_bruss};

            case 'repressilator'
                param_sampler = @() sample_repressilator(hill_rep, hill_deg, ...
                    hill_act, latent_class);
                ode_fun  = @(t,s,p) repressilator_ode(t,s,p,latent_class);
                ic_fun   = @(p) [1.0; 0.5; 0.5];
                lib_fun  = @(x,y,N,p) standard_lib(x,y,N,p);
                aux_funs = {@aux_x_standard, @aux_y_standard};

            case 'van_der_pol'
                param_sampler = @() sample_vanderpol(hill_rep, hill_deg, ...
                    hill_act, latent_class);
                ode_fun  = @(t,s,p) vanderpol_ode(t,s,p,latent_class);
                ic_fun   = @(p) [0.5; 0.0; 0.5];
                lib_fun  = @(x,y,N,p) standard_lib(x,y,N,p);
                aux_funs = {@aux_x_standard, @aux_y_standard};

            case 'fitzhugh_nagumo'
                param_sampler = @() sample_fitzhugh(hill_rep, hill_deg, ...
                    hill_act, latent_class);
                ode_fun  = @(t,s,p) fitzhugh_ode(t,s,p,latent_class);
                ic_fun   = @(p) [0.5; 0.1; 0.5];
                lib_fun  = @(x,y,N,p) fitzhugh_lib(x,y,N,p);
                aux_funs = {@aux_x_fitzhugh, @aux_y_fitzhugh};

            case 'lotka_volterra'
                param_sampler = @() sample_lotkavolterra(hill_rep, hill_deg, ...
                    hill_act, latent_class);
                ode_fun  = @(t,s,p) lotkavolterra_ode(t,s,p,latent_class);
                ic_fun   = @(p) [2.0; 0.5; 0.5];
                lib_fun  = @(x,y,N,p) lotkavolterra_lib(x,y,N,p);
                aux_funs = {@aux_x_lv, @aux_y_lv};
        end

        run_generation(system_name, ode_fun, ic_fun, param_sampler, ...
            lib_fun, aux_funs, latent_class, n_per_config, save_dir, ...
            base_seed + sys_idx*10 + latent_class);
    end
end

fprintf('\nAll V2 Lat4-7 Sys1-6 datasets complete.\n');


%% ── GENERATION LOOP ──────────────────────────────────────────────────────
function run_generation(system_name, ode_fun, ic_fun, param_sampler, ...
        lib_fun, aux_funs, latent_class, n_examples, save_dir, rng_seed)

rng(rng_seed);
existing = length(dir(fullfile(save_dir, '*.mat')));

dt   = 0.05;
N    = 300;
eta  = 0.6;
sg_p = 3;
sg_f = 11;

saved   = 0;
skipped = 0;
attempt = 0;

% Tolerances loosened from original scripts — sufficient for data
% generation and significantly reduces ODE solve time per sample.
% MaxStep = dt prevents the solver overshooting and backtracking on
% stiff trajectories, particularly for Hill repressor latents where
% I changes sharply as x or y crosses the Hill threshold.
if strcmp(system_name, 'van_der_pol')
    opts_ode = odeset('RelTol',1e-4,'AbsTol',1e-6, 'MaxStep',dt);
elseif any(strcmp(system_name, {'fitzhugh_nagumo','lotka_volterra'}))
    opts_ode = odeset('RelTol',1e-4,'AbsTol',1e-6, 'MaxStep',dt);
else
    opts_ode = odeset('RelTol',1e-5,'AbsTol',1e-7, 'MaxStep',dt);
end

opts_ls = optimoptions('lsqlin','Display','off');
t_end   = (N+200)*dt;
t_span  = 0:dt:t_end;
n_fft   = 501;
win     = hann(n_fft);

while saved < n_examples

    attempt = attempt+1;
    p       = param_sampler();

    try
        [t_ode,S] = ode15s(@(t,s) ode_fun(t,s,p), ...
            t_span, ic_fun(p), opts_ode);
    catch
        skipped = skipped+1; continue
    end

    if any(~isfinite(S(:))) || length(t_ode) < N+201
        skipped = skipped+1; continue
    end

    t_uniform  = (0:dt:t_end)';
    x_data_all = interp1(t_ode, S(:,1), t_uniform, 'linear');
    y_data_all = interp1(t_ode, S(:,2), t_uniform, 'linear');
    z_data_all = interp1(t_ode, S(:,3), t_uniform, 'linear');
    t_ode      = t_uniform;
    n_all      = length(t_ode);

    if any(~isfinite(x_data_all)) || any(~isfinite(y_data_all))
        skipped = skipped+1; continue
    end

    % Reject flat or diverging trajectories
    % Hill repressor latents (6,7) produce lower-amplitude I oscillations
    % so the z variance threshold is relaxed to avoid excessive skipping
    tail = round(0.4*n_all);
    if ismember(latent_class, [5, 6])
        z_thresh = 1e-8;
    else
        z_thresh = 1e-6;
    end
    if var(z_data_all(end-tail:end)) < z_thresh || ...
       var(x_data_all(end-tail:end)) < 1e-6
        skipped = skipped+1; continue
    end

    if max(abs(x_data_all)) > 1e4 || max(abs(y_data_all)) > 1e4
        skipped = skipped+1; continue
    end

    % Early cheap rejection: check x oscillation quality before paying
    % the full RI-SINDy cost. Compute derivative and reject near-flat signals.
    dx_quick = diff(x_data_all(1:N));
    if std(dx_quick) < 1e-4
        skipped = skipped+1; continue
    end

    noise      = p.noise;
    x_data_all = x_data_all + noise*std(x_data_all)*randn(n_all,1);
    y_data_all = y_data_all + noise*std(y_data_all)*randn(n_all,1);

    x_data = x_data_all(1:N);
    y_data = y_data_all(1:N);

    lib        = lib_fun(x_data, y_data, N, p);
    Theta_full = lib.Theta_full;
    Theta_poly = lib.Theta_poly;
    col_names  = lib.col_names;
    n_poly     = size(Theta_poly,2);
    n_terms    = size(Theta_full,2);

    colscale_full             = vecnorm(Theta_full,2,1);
    colscale_full(colscale_full==0) = 1;
    ThetaN_full               = Theta_full ./ colscale_full;
    ThetaN_poly               = Theta_poly ./ colscale_full(1:n_poly);

    dxdt = sgolayfilt(gradient(x_data,dt), sg_p, sg_f);
    dydt = sgolayfilt(gradient(y_data,dt), sg_p, sg_f);

    targetScaleX = norm(dxdt,2);
    targetScaleY = norm(dydt,2);
    dSdtN        = [dxdt/targetScaleX, dydt/targetScaleY];

    XiN        = zeros(n_terms,2);
    XiN(1:n_poly,:) = ThetaN_poly \ dSdtN;
    XiN_var    = zeros(n_terms,2);
    XiN_smooth = XiN;
    wx = ones(n_poly,1);
    wy = ones(n_poly,1);

    aux_x_fun = aux_funs{1};
    aux_y_fun = aux_funs{2};

    for iter = 1:25   % reduced from 50 — convergence typically reached by iter 20
        for ind = 1:2
            switch ind
                case 1
                    current_w = wx;
                    [h_val,h_var] = aux_x_fun(XiN_smooth,XiN_var, ...
                        colscale_full,targetScaleX,x_data,y_data,p);
                    Hill_red  = ThetaN_full(:,end-1)*h_val;
                    dSdtN_ej  = dSdtN(:,1) - Hill_red;
                case 2
                    current_w = wy;
                    [h_val,h_var] = aux_y_fun(XiN_smooth,XiN_var, ...
                        colscale_full,targetScaleY,x_data,y_data,p);
                    Hill_red  = ThetaN_full(:,end)*h_val;
                    dSdtN_ej  = dSdtN(:,2) - Hill_red;
            end

            ThetaN_w     = ThetaN_poly ./ current_w';
            biginds      = current_w < 10;
            XiN_poly_vec = XiN(1:n_poly,ind);
            lb = p.lb{ind};  ub = p.ub{ind};

            if any(biginds)
                active = find(biginds);
                XiN_poly_vec(active) = lsqlin( ...
                    ThetaN_w(:,active), dSdtN_ej, ...
                    [],[],[],[], lb(active), ub(active), [], opts_ls);
                XiN_poly_vec(~biginds) = 0;
            end

            XiN(1:n_poly,ind) = XiN_poly_vec;

            hill_idx          = n_poly+ind;
            XiN(hill_idx,ind) = (1-eta)*XiN(hill_idx,ind)+eta*h_val;
            XiN_var(hill_idx,ind) = h_var;

            eps_s = 1e-3; tau0 = 0.05;
            new_w = 1./(tau0^2*(XiN_poly_vec.^2+eps_s));
            if ind==1; wx=0.7*wx+0.3*new_w; current_w=wx;
            else;      wy=0.7*wy+0.3*new_w; current_w=wy;
            end

            resid     = dSdtN_ej - ThetaN_poly*XiN_poly_vec;
            resid_var = max(var(resid),1e-12);
            H         = (ThetaN_poly'*ThetaN_poly)/resid_var + diag(current_w);
            XiN_var(1:n_poly,ind) = diag(inv(H));
        end

        XiN_smooth = 0.7*XiN_smooth + 0.3*XiN;
    end

    Xi      = zeros(n_terms,2);
    Xi(:,1) = (XiN(:,1)*targetScaleX) ./ colscale_full';
    Xi(:,2) = (XiN(:,2)*targetScaleY) ./ colscale_full';
    Xi(abs(Xi)<0.005) = 0;

    dx_obs = sgolayfilt(gradient(x_data_all,dt), sg_p, sg_f);
    dy_obs = sgolayfilt(gradient(y_data_all,dt), sg_p, sg_f);

    % Vectorised model reconstruction — builds full library matrix at once
    % rather than calling lib_row in a scalar loop, ~10-20x faster
    XiX = Xi(:,1); XiY = Xi(:,2);
    Phi_all = lib_matrix(x_data_all, y_data_all, n_all, p, system_name);
    dx_model = [0; Phi_all(1:end-1,:) * XiX];
    dy_model = [0; Phi_all(1:end-1,:) * XiY];

    resid_dx = dx_obs - dx_model;
    resid_dy = dy_obs - dy_model;

    if var(resid_dx) < 0.3*var(resid_dy)
        skipped = skipped+1; continue
    end

    %% ── FFT COMPUTATION ──────────────────────────────────────────────
    rx = resid_dx(1:min(end,n_fft));
    ry = resid_dy(1:min(end,n_fft));

    if length(rx) < n_fft
        rx = [rx; zeros(n_fft - length(rx), 1)];
        ry = [ry; zeros(n_fft - length(ry), 1)];
    end

    rx = (rx - mean(rx)) .* win;
    ry = (ry - mean(ry)) .* win;

    Fx = fft(rx, n_fft);
    Fy = fft(ry, n_fft);

    n_bins = floor(n_fft/2) + 1;
    Fx     = Fx(1:n_bins);
    Fy     = Fy(1:n_bins);

    mag_x = abs(Fx);
    mag_y = abs(Fy);
    mag_x = mag_x / (max(mag_x) + 1e-8);
    mag_y = mag_y / (max(mag_y) + 1e-8);

    phase_diff   = angle(Fx .* conj(Fy));
    fft_features = [mag_x, mag_y, phase_diff]';

    %% ── TERM ACTIVITY MATRIX ─────────────────────────────────────────
    Xi_ternary = sign(Xi).*(abs(Xi)>0.005);
    if size(Xi_ternary,1) < 9
        Xi_ternary = [Xi_ternary; zeros(9-size(Xi_ternary,1),2)];
    end

    %% ── TOPOLOGY ─────────────────────────────────────────────────────
    half      = round(n_all/2);
    x_2nd     = x_data_all(half:end);
    y_2nd     = y_data_all(half:end);
    sustained = (var(x_2nd)>1e-4) && (var(y_2nd)>1e-4);
    loop_err  = norm([x_2nd(end)-x_2nd(1), y_2nd(end)-y_2nd(1)]) / ...
                (norm([x_2nd(1),y_2nd(1)])+1e-8);
    is_closed = loop_err < 0.25;
    seg       = round(length(x_2nd)/3);
    amp_ratio = std(x_2nd(end-seg+1:end)) / max(std(x_2nd(1:seg)),1e-8);

    if     amp_ratio > 1.25; amp_trend = 'growing';
    elseif amp_ratio < 0.75; amp_trend = 'decaying';
    else;                    amp_trend = 'stable';
    end

    if     sustained && is_closed && strcmp(amp_trend,'stable')
        topology = 'LIMIT CYCLE';
    elseif sustained && strcmp(amp_trend,'decaying')
        topology = 'DAMPED OSCILLATION';
    elseif ~sustained
        topology = 'STEADY STATE';
    else
        topology = 'UNDETERMINED';
    end

    % 0-indexed label — folder name stays V2Lat4-7, label matches Python
    structure_label  = latent_class;   % folders and labels both 0-indexed, no offset needed
    params_saved     = rmfield_safe(p, {'hill_rep','hill_deg','hill_act','lb','ub'});
    params_saved.system = system_name;

    fname = fullfile(save_dir, ...
        sprintf('example_%04d.mat', existing+saved+1));

    save(fname, ...
        'fft_features',    ...
        'Xi_ternary',      ...
        'topology',        ...
        'structure_label', ...
        'params_saved',    ...
        'x_data_all', 'y_data_all', ...
        'z_data_all', 't_ode',      ...
        'Xi',         'col_names',  ...
        'resid_dx',   'resid_dy');

    saved = saved+1;

    if mod(saved,10)==0
        fprintf('[%s | Lat%d] Saved %d/%d  (skipped %d, attempts %d)\n', ...
            system_name, latent_class, saved, n_examples, skipped, attempt);
    end
end

fprintf('[%s | Lat%d] Done. Saved %d, skipped %d, attempts %d\n', ...
    system_name, latent_class, saved, skipped, attempt);
end


%% ── ODE FUNCTIONS ────────────────────────────────────────────────────────

function ds = goodwin_ode(~, s, p, latent_class)
    ds    = zeros(3,1);
    ds(1) = p.alpha * p.hill_rep(s(3),p.hill_k0,p.hill_n) - p.d1*s(1);
    ds(2) = p.ks*s(1) - p.Vmax * p.hill_deg(s(2),p.hill_km);
    ds(3) = latent_z(s, p, latent_class);
end

function ds = brusselator_ode(~, s, p, latent_class)
    ds    = zeros(3,1);
    ds(1) = p.a - (p.b+1)*s(1) + s(1)^2*s(2) - ...
            p.hill_rep(s(3),p.hill_k0,p.hill_n)*s(1);
    ds(2) = p.b*s(1) - s(1)^2*s(2);
    ds(3) = latent_z(s, p, latent_class);
end

function ds = repressilator_ode(~, s, p, latent_class)
    ds    = zeros(3,1);
    ds(1) = p.alpha * p.hill_rep(s(3),p.hill_k0,p.hill_n) - ...
            p.delta * p.hill_deg(s(1),p.hill_km);
    ds(2) = p.beta*s(1) - p.gamma*s(2);
    ds(3) = latent_z(s, p, latent_class);
end

function ds = vanderpol_ode(~, s, p, latent_class)
    ds    = zeros(3,1);
    ds(1) = p.mu*(1-s(2)^2)*s(1) - p.omega*s(2) - ...
            p.alpha * p.hill_rep(s(3),p.hill_k0,p.hill_n);
    ds(2) = s(1);
    ds(3) = latent_z(s, p, latent_class);
end

function ds = fitzhugh_ode(~, s, p, latent_class)
    ds    = zeros(3,1);
    ds(1) = s(1) - (s(1)^3)/3 - s(2) + p.I_ext + ...
            p.alpha * p.hill_rep(s(3), p.hill_k0, p.hill_n);
    ds(2) = (s(1) + p.a_fhn - p.b_fhn*s(2)) / p.tau_fhn;
    ds(3) = latent_z(s, p, latent_class);
end

function ds = lotkavolterra_ode(~, s, p, latent_class)
    sat   = p.a_lv * s(1) / (1 + p.h_lv * s(1));
    ds    = zeros(3,1);
    ds(1) = p.r_lv * s(1) * (1 - s(1)/p.K_lv) - sat*s(2) - ...
            p.alpha * p.hill_rep(s(3), p.hill_k0, p.hill_n) * s(1);
    ds(2) = p.e_lv * sat * s(2) - p.d_lv * s(2);
    ds(3) = latent_z(s, p, latent_class);
end

% ── New latent equations ──────────────────────────────────────────────────
%
% Lat4 — Hill repressor driven by y:
%   dI/dt = kp * HillRep(y, k0, n) - kd * I
%   I is suppressed when y is high; rebounds when y falls.
%   Phase-inverted relative to Lat2 (which uses linear y).
%
% Lat5 — Hill repressor driven by x:
%   dI/dt = kp * HillRep(x, k0, n) - kd * I
%   I is suppressed when x is high; antiphase to Lat1 (Hill activation x).
%
% Lat6 — Additive dual-input:
%   dI/dt = kp * x + kq * y - kd * I
%   I integrates both observables; residual carries mixed-phase structure
%   distinguishable from Lat0 (x only) and Lat2 (y only).
%
% Lat7 — Multiplicative (mass-action) coupling:
%   dI/dt = kp * x * y - kd * I
%   Production requires both x and y simultaneously; amplitude modulated
%   by the product xy, peaking at different phases than either alone.

function dI = latent_z(s, p, latent_class)
    switch latent_class
        case 4
            % Hill repressor by y — inverse of Lat2
            dI = p.kp * p.hill_rep(s(2), p.hill_k0, p.hill_n) - p.kd*s(3);
        case 5
            % Hill repressor by x — inverse of Lat1
            dI = p.kp * p.hill_rep(s(1), p.hill_k0, p.hill_n) - p.kd*s(3);
        case 6
            % Additive dual-input
            dI = p.kp*s(1) + p.kq*s(2) - p.kd*s(3);
        case 7
            % Multiplicative mass-action coupling
            dI = p.kp * s(1) * s(2) - p.kd*s(3);
    end
end


%% ── PARAM SAMPLERS ───────────────────────────────────────────────────────
% All samplers are identical to the Lat0-3 versions with one addition:
% Lat6 requires kq (the y-coupling weight) and Lat7 uses kp as the
% bimolecular rate constant. Both are included in all samplers so the
% same struct works for any latent class.

function p = sample_goodwin(hill_rep, hill_deg, hill_act, latent_class)
    p = struct( ...
        'alpha',   5+7*rand(),        'd1',    0.3+0.7*rand(), ...
        'ks',      1+3*rand(),        'Vmax',  3+5*rand(),     ...
        'hill_n',  round(3+3*rand()), ...
        'hill_k0', 1.5+2.5*rand(),    ...
        'hill_km', 0.4+1.6*rand(),    ...
        'hill_ka', 1.5+1.0*rand(),    ...
        'hill_na', round(1+2*rand()), ...
        'kp',      goodwin_kp(latent_class),   ...
        'kq',      0.5+1.5*rand(),              ...   % for Lat6 dual-input
        'kd',      goodwin_kd(latent_class),   ...
        'hill_rep', hill_rep,         'hill_deg', hill_deg, ...
        'hill_act', hill_act,         'noise',  0.01+0.11*rand(), ...
        'lb', {{[0,-inf,-inf,-inf,0,0,0],[0,0,0,0,-inf,-inf,-inf]}}, ...
        'ub', {{[0,0,0,0,0,0,0],[0,inf,inf,inf,0,0,0]}});
end

function p = sample_brusselator(hill_rep, hill_deg, hill_act, latent_class)
    while true
        a = 1+2*rand(); b = 4+6*rand();
        if b > a^2+1; break; end
    end
    p = struct( ...
        'a', a, 'b', b, ...
        'hill_n',  round(2+3*rand()), ...
        'hill_k0', 1.5+2.5*rand(),    ...
        'hill_km', 0.5+1.5*rand(),    ...
        'hill_ka', 0.5+2.0*rand(),    ...
        'hill_na', round(1+2*rand()), ...
        'kp',      hill_rep_kp(latent_class),   ...
        'kq',      0.5+1.5*rand(),              ...
        'kd',      hill_rep_kd(latent_class),   ...
        'hill_rep', hill_rep,  'hill_deg', hill_deg, ...
        'hill_act', hill_act,  'noise',  0.01+0.09*rand(), ...
        'lb', {{repmat(-inf,1,5), repmat(-inf,1,5)}}, ...
        'ub', {{repmat(inf,1,5),  repmat(inf,1,5)}});
end

function p = sample_repressilator(hill_rep, hill_deg, hill_act, latent_class)
    p = struct( ...
        'alpha',  4+6*rand(),         'delta', 1+2*rand(),   ...
        'km',     0.3+1.2*rand(),     ...
        'beta',   1+2*rand(),         'gamma', 0.5+1.5*rand(), ...
        'hill_n', round(2+3*rand()),  ...
        'hill_k0',1+3*rand(),         'hill_km', 0.4+1.6*rand(), ...
        'hill_ka',0.5+2.0*rand(),     'hill_na', round(1+2*rand()), ...
        'kp',     hill_rep_kp(latent_class),    ...
        'kq',     0.5+1.5*rand(),               ...
        'kd',     hill_rep_kd(latent_class),    ...
        'hill_rep', hill_rep,  'hill_deg', hill_deg, ...
        'hill_act', hill_act,  'noise',  0.01+0.09*rand(), ...
        'lb', {{[0,-inf,-inf,-inf,0,0,0],[0,0,0,0,-inf,-inf,-inf]}}, ...
        'ub', {{[0,0,0,0,0,0,0],[0,inf,inf,inf,0,0,0]}});
end

function p = sample_vanderpol(hill_rep, hill_deg, hill_act, latent_class)
    p = struct( ...
        'mu',    0.3+0.7*rand(),     'omega', 0.8+0.4*rand(), ...
        'alpha', 0.5+1.5*rand(),     ...
        'hill_n', round(2+3*rand()), ...
        'hill_k0',1+2*rand(),        'hill_km', 0.5+1.5*rand(), ...
        'hill_ka',0.5+2.0*rand(),    'hill_na', round(1+2*rand()), ...
        'kp',    hill_rep_kp(latent_class),     ...
        'kq',    0.5+1.5*rand(),                ...
        'kd',    hill_rep_kd(latent_class),     ...
        'hill_rep', hill_rep,  'hill_deg', hill_deg, ...
        'hill_act', hill_act,  'noise',  0.01+0.09*rand(), ...
        'lb', {{repmat(-inf,1,7), repmat(-inf,1,7)}}, ...
        'ub', {{repmat(inf,1,7),  repmat(inf,1,7)}});
end

function p = sample_fitzhugh(hill_rep, hill_deg, hill_act, latent_class)
    p = struct( ...
        'a_fhn',   0.5+0.5*rand(),    ...
        'b_fhn',   0.5+0.3*rand(),    ...
        'tau_fhn', 10+10*rand(),      ...
        'I_ext',   0.3+0.6*rand(),    ...
        'alpha',   0.3+0.7*rand(),    ...
        'hill_n',  round(2+3*rand()), ...
        'hill_k0', 1.0+2.0*rand(),    ...
        'hill_km', 0.3+1.2*rand(),    ...
        'hill_ka', 0.5+1.5*rand(),    ...
        'hill_na', round(1+2*rand()), ...
        'kp',      hill_rep_kp(latent_class),   ...
        'kq',      0.5+1.5*rand(),              ...
        'kd',      hill_rep_kd(latent_class),   ...
        'hill_rep', hill_rep,  'hill_deg', hill_deg, ...
        'hill_act', hill_act,  'noise',  0.01+0.09*rand(), ...
        'lb', {{repmat(-inf,1,7), repmat(-inf,1,7)}}, ...
        'ub', {{repmat(inf,1,7),  repmat(inf,1,7)}});
end

function p = sample_lotkavolterra(hill_rep, hill_deg, hill_act, latent_class)
    while true
        r = 0.8+0.8*rand(); K = 4.0+4.0*rand();
        a = 0.5+1.0*rand(); h = 0.1+0.4*rand();
        e = 0.3+0.4*rand(); d = 0.2+0.4*rand();
        if e*a > d*h && e*a > d; break; end
    end
    p = struct( ...
        'r_lv', r,  'K_lv', K,  'a_lv', a,  'h_lv', h, ...
        'e_lv', e,  'd_lv', d,  ...
        'alpha',   0.3+0.7*rand(),    ...
        'hill_n',  round(2+3*rand()), ...
        'hill_k0', 1.0+2.0*rand(),    ...
        'hill_km', 0.3+1.5*rand(),    ...
        'hill_ka', 0.5+2.0*rand(),    ...
        'hill_na', round(1+2*rand()), ...
        'kp',      hill_rep_kp(latent_class),   ...
        'kq',      0.5+1.5*rand(),              ...
        'kd',      hill_rep_kd(latent_class),   ...
        'hill_rep', hill_rep,  'hill_deg', hill_deg, ...
        'hill_act', hill_act,  'noise',  0.01+0.09*rand(), ...
        'lb', {{repmat(-inf,1,8), repmat(-inf,1,8)}}, ...
        'ub', {{repmat(inf,1,8),  repmat(inf,1,8)}});
end


%% ── LIBRARY BUILDERS ─────────────────────────────────────────────────────

function lib = standard_lib(x, y, N, p)
    lib.Theta_full = [ones(N,1), x, x.^2, x.^3, y, y.^2, y.^3, ...
                      p.hill_rep(y,p.hill_k0,p.hill_n), ...
                      p.hill_deg(y,p.hill_km)];
    lib.Theta_poly = lib.Theta_full(:,1:7);
    lib.col_names  = {'1','x','x^2','x^3','y','y^2','y^3', ...
                      'HillRep','HillDeg'};
end

function lib = brusselator_lib(x, y, N, p)
    lib.Theta_full = [ones(N,1), x, x.^2, y, x.^2.*y, ...
                      p.hill_rep(y,p.hill_k0,p.hill_n), ...
                      p.hill_deg(y,p.hill_km)];
    lib.Theta_poly = lib.Theta_full(:,1:5);
    lib.col_names  = {'1','x','x^2','y','x^2y','HillRep','HillDeg'};
end

function lib = fitzhugh_lib(x, y, N, p)
    lib.Theta_full = [ones(N,1), x, x.^2, x.^3, y, y.^2, y.^3, ...
                      p.hill_rep(y, p.hill_k0, p.hill_n), ...
                      p.hill_deg(y, p.hill_km)];
    lib.Theta_poly = lib.Theta_full(:,1:7);
    lib.col_names  = {'1','x','x^2','x^3','y','y^2','y^3', ...
                      'HillRep','HillDeg'};
end

function lib = lotkavolterra_lib(x, y, N, p)
    lib.Theta_full = [ones(N,1), x, x.^2, y, x.*y, x.^2.*y, ...
                      p.hill_rep(y, p.hill_k0, p.hill_n), ...
                      p.hill_deg(y, p.hill_km)];
    lib.Theta_poly = lib.Theta_full(:,1:6);
    lib.col_names  = {'1','x','x^2','y','xy','x^2*y', ...
                      'HillRep','HillDeg'};
end

function row = lib_row(x, y, p, system_name)
    switch system_name
        case 'brusselator'
            row = [1, x, x^2, y, x^2*y, ...
                   p.hill_rep(y,p.hill_k0,p.hill_n), ...
                   p.hill_deg(y,p.hill_km)];
        case 'lotka_volterra'
            row = [1, x, x^2, y, x*y, x^2*y, ...
                   p.hill_rep(y,p.hill_k0,p.hill_n), ...
                   p.hill_deg(y,p.hill_km)];
        otherwise  % goodwin, repressilator, van_der_pol, fitzhugh_nagumo
            row = [1, x, x^2, x^3, y, y^2, y^3, ...
                   p.hill_rep(y,p.hill_k0,p.hill_n), ...
                   p.hill_deg(y,p.hill_km)];
    end
end

% Vectorised version of lib_row — builds the full (n x n_terms) library
% matrix in one shot using column operations instead of a scalar loop.
% Called by the model reconstruction block to replace the for k loop.
function Phi = lib_matrix(x, y, n, p, system_name)
    hr = p.hill_rep(y, p.hill_k0, p.hill_n);
    hd = p.hill_deg(y, p.hill_km);
    switch system_name
        case 'brusselator'
            Phi = [ones(n,1), x, x.^2, y, x.^2.*y, hr, hd];
        case 'lotka_volterra'
            Phi = [ones(n,1), x, x.^2, y, x.*y, x.^2.*y, hr, hd];
        otherwise
            Phi = [ones(n,1), x, x.^2, x.^3, y, y.^2, y.^3, hr, hd];
    end
end


%% ── AUXILIARY FUNCTIONS ──────────────────────────────────────────────────

function [h_val,h_var] = aux_x_standard(XiN_smooth,XiN_var, ...
        colscale_full,targetScaleX,x_data,y_data,p)
    xi  = XiN_smooth([2 3 4],1);
    vxi = XiN_var([2 3 4],1);
    cx  = (xi(1)*targetScaleX)/colscale_full(2);
    cx2 = (xi(2)*targetScaleX)/colscale_full(3);
    cx3 = (xi(3)*targetScaleX)/colscale_full(4);
    avg_drain   = mean(abs(cx*x_data + cx2*x_data.^2 + cx3*x_data.^3));
    h_basis_avg = mean(p.hill_rep(y_data,p.hill_k0,p.hill_n));
    force_phys  = avg_drain / max(h_basis_avg,1e-3);
    h_val       = (force_phys*colscale_full(end-1)) / targetScaleX;
    sv          = (targetScaleX./colscale_full(2:4)') * ...
                  (colscale_full(end-1)/targetScaleX);
    h_var       = sum(vxi.*sv.^2);
end

function [h_val,h_var] = aux_y_standard(XiN_smooth,XiN_var, ...
        colscale_full,targetScaleY,x_data,y_data,p)
    xi  = XiN_smooth([2 3 4],2);
    vxi = XiN_var([2 3 4],2);
    cx  = (xi(1)*targetScaleY)/colscale_full(2);
    cx2 = (xi(2)*targetScaleY)/colscale_full(3);
    cx3 = (xi(3)*targetScaleY)/colscale_full(4);
    avg_prod    = mean(abs(cx*x_data + cx2*x_data.^2 + cx3*x_data.^3));
    h_basis_avg = mean(p.hill_deg(y_data,p.hill_km));
    force_phys  = avg_prod / max(h_basis_avg,1e-3);
    h_val       = -(force_phys*colscale_full(end)) / targetScaleY;
    sv          = (targetScaleY./colscale_full(2:4)') * ...
                  (colscale_full(end)/targetScaleY);
    h_var       = sum(vxi.*sv.^2);
end

function [h_val,h_var] = aux_x_bruss(XiN_smooth,XiN_var, ...
        colscale_full,targetScaleX,x_data,y_data,p)
    xi  = XiN_smooth(2:3,1);
    vxi = XiN_var(2:3,1);
    cx  = (xi(1)*targetScaleX)/colscale_full(2);
    cx2 = (xi(2)*targetScaleX)/colscale_full(3);
    avg_drain   = mean(abs(cx*x_data + cx2*x_data.^2));
    h_basis_avg = mean(p.hill_rep(y_data,p.hill_k0,p.hill_n));
    force_phys  = avg_drain / max(h_basis_avg,1e-3);
    h_val       = (force_phys*colscale_full(end-1)) / targetScaleX;
    sv          = (targetScaleX./colscale_full(2:3)') * ...
                  (colscale_full(end-1)/targetScaleX);
    h_var       = sum(vxi.*sv.^2);
end

function [h_val,h_var] = aux_y_bruss(XiN_smooth,XiN_var, ...
        colscale_full,targetScaleY,x_data,y_data,p)
    xi  = XiN_smooth(2:3,2);
    vxi = XiN_var(2:3,2);
    cx  = (xi(1)*targetScaleY)/colscale_full(2);
    cx2 = (xi(2)*targetScaleY)/colscale_full(3);
    avg_prod    = mean(abs(cx*x_data + cx2*x_data.^2));
    h_basis_avg = mean(p.hill_deg(y_data,p.hill_km));
    force_phys  = avg_prod / max(h_basis_avg,1e-3);
    h_val       = -(force_phys*colscale_full(end)) / targetScaleY;
    sv          = (targetScaleY./colscale_full(2:3)') * ...
                  (colscale_full(end)/targetScaleY);
    h_var       = sum(vxi.*sv.^2);
end

function [h_val,h_var] = aux_x_fitzhugh(XiN_smooth,XiN_var, ...
        colscale_full,targetScaleX,x_data,y_data,p)
    xi  = XiN_smooth([2 3 4],1);
    vxi = XiN_var([2 3 4],1);
    cx  = (xi(1)*targetScaleX)/colscale_full(2);
    cx2 = (xi(2)*targetScaleX)/colscale_full(3);
    cx3 = (xi(3)*targetScaleX)/colscale_full(4);
    avg_drain   = mean(abs(cx*x_data + cx2*x_data.^2 + cx3*x_data.^3));
    h_basis_avg = mean(p.hill_rep(y_data,p.hill_k0,p.hill_n));
    force_phys  = avg_drain / max(h_basis_avg,1e-3);
    h_val       = (force_phys*colscale_full(end-1)) / targetScaleX;
    sv          = (targetScaleX./colscale_full(2:4)') * ...
                  (colscale_full(end-1)/targetScaleX);
    h_var       = sum(vxi.*sv.^2);
end

function [h_val,h_var] = aux_y_fitzhugh(XiN_smooth,XiN_var, ...
        colscale_full,targetScaleY,x_data,y_data,p)
    xi  = XiN_smooth([5 6 7],2);
    vxi = XiN_var([5 6 7],2);
    cy  = (xi(1)*targetScaleY)/colscale_full(5);
    cy2 = (xi(2)*targetScaleY)/colscale_full(6);
    cy3 = (xi(3)*targetScaleY)/colscale_full(7);
    avg_prod    = mean(abs(cy*y_data + cy2*y_data.^2 + cy3*y_data.^3));
    h_basis_avg = mean(p.hill_deg(y_data,p.hill_km));
    force_phys  = avg_prod / max(h_basis_avg,1e-3);
    h_val       = -(force_phys*colscale_full(end)) / targetScaleY;
    sv          = (targetScaleY./colscale_full(5:7)') * ...
                  (colscale_full(end)/targetScaleY);
    h_var       = sum(vxi.*sv.^2);
end

function [h_val,h_var] = aux_x_lv(XiN_smooth,XiN_var, ...
        colscale_full,targetScaleX,x_data,y_data,p)
    xi  = XiN_smooth(2:3,1);
    vxi = XiN_var(2:3,1);
    cx  = (xi(1)*targetScaleX)/colscale_full(2);
    cx2 = (xi(2)*targetScaleX)/colscale_full(3);
    avg_drain   = mean(abs(cx*x_data + cx2*x_data.^2));
    h_basis_avg = mean(p.hill_rep(y_data,p.hill_k0,p.hill_n));
    force_phys  = avg_drain / max(h_basis_avg,1e-3);
    h_val       = (force_phys*colscale_full(end-1)) / targetScaleX;
    sv          = (targetScaleX./colscale_full(2:3)') * ...
                  (colscale_full(end-1)/targetScaleX);
    h_var       = sum(vxi.*sv.^2);
end

function [h_val,h_var] = aux_y_lv(XiN_smooth,XiN_var, ...
        colscale_full,targetScaleY,x_data,y_data,p)
    xi  = XiN_smooth([4 5],2);
    vxi = XiN_var([4 5],2);
    cy  = (xi(1)*targetScaleY)/colscale_full(4);
    cxy = (xi(2)*targetScaleY)/colscale_full(5);
    avg_prod    = mean(abs(cy*y_data + cxy*x_data.*y_data));
    h_basis_avg = mean(p.hill_deg(y_data,p.hill_km));
    force_phys  = avg_prod / max(h_basis_avg,1e-3);
    h_val       = -(force_phys*colscale_full(end)) / targetScaleY;
    sv          = (targetScaleY./colscale_full([4 5])') * ...
                  (colscale_full(end)/targetScaleY);
    h_var       = sum(vxi.*sv.^2);
end



%% ── KP / KD HELPERS FOR HILL REPRESSOR LATENTS ──────────────────────────
% Lat6 (case 6) and Lat7 (case 7) are Hill repressors. Their production
% rate kp is raised and decay rate kd is lowered so that I builds up
% enough amplitude to pass the variance filter and produce informative
% residuals. All other latents use the standard ranges.

function v = goodwin_kp(latent_class)
    if ismember(latent_class, [5, 6])
        v = 2+3*rand();    % raised from 1+2
    else
        v = 1+2*rand();
    end
end

function v = goodwin_kd(latent_class)
    if ismember(latent_class, [5, 6])
        v = 0.2+0.6*rand();  % lowered from 0.5+1.5
    else
        v = 0.5+1.5*rand();
    end
end

function v = hill_rep_kp(latent_class)
    if ismember(latent_class, [5, 6])
        v = 2+3*rand();
    else
        v = 1+2*rand();
    end
end

function v = hill_rep_kd(latent_class)
    if ismember(latent_class, [5, 6])
        v = 0.2+0.6*rand();
    else
        v = 0.5+1.5*rand();
    end
end

%% ── SAFE FIELD REMOVAL ───────────────────────────────────────────────────
function s = rmfield_safe(s, fields)
    for i = 1:length(fields)
        if isfield(s, fields{i})
            s = rmfield(s, fields{i});
        end
    end
end