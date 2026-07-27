clear; close all; clc;
addpath(genpath('.'));

n_examples = 500;
save_dir   = 'C:/Users/nickj/MATLAB Drive/LTI Network/Lat2_Sys1_NNdata';
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

rng(3);

hill_rep  = @(z, k, n)  k^n ./ (k^n + z.^n);
hill_deg  = @(y, km)    y   ./ (km  + y);
hill_act  = @(x, k, n)  x.^n ./ (k^n + x.^n);

dt   = 0.05;
N    = 300;
eta  = 0.6;
sg_p = 3;
sg_f = 11;

alpha_range    = [5.0,  12.0];
d1_range      = [0.3,   1.0];
ks_range      = [1.0,   4.0];
Vmax_range    = [3.0,   8.0];
hill_n_range  = [4,     7  ];
kp_range      = [0.5,   3.0];
kd_range      = [0.5,   2.0];
hill_na_range = [1,     2  ];
hill_ka_range = [0.3,   2.0];
hill_k0_range = [1.5,   4.0];
hill_km_range = [0.4,   2.0];
noise_range   = [0.01,  0.12];

saved   = 0;
skipped = 0;
attempt = 0;

while saved < n_examples
    if mod(attempt, 1000) == 0
        fprintf('Attempt %d | saved %d | skipped %d\n', ...
            attempt, saved, skipped);
    end
    attempt = attempt + 1;

    alpha   = alpha_range(1)   + diff(alpha_range)   * rand();
    d1      = d1_range(1)      + diff(d1_range)      * rand();
    ks      = ks_range(1)      + diff(ks_range)      * rand();
    Vmax    = Vmax_range(1)    + diff(Vmax_range)    * rand();
    kp      = kp_range(1)      + diff(kp_range)      * rand();
    kd      = kd_range(1)      + diff(kd_range)      * rand();
    hill_n  = round(hill_n_range(1)  + diff(hill_n_range)  * rand());
    hill_k0 = hill_k0_range(1) + diff(hill_k0_range) * rand();
    hill_km = hill_km_range(1) + diff(hill_km_range) * rand();
    hill_na = round(hill_na_range(1) + diff(hill_na_range) * rand());
    hill_ka = hill_ka_range(1) + diff(hill_ka_range) * rand();
    noise   = noise_range(1)   + diff(noise_range)   * rand();

    t_end  = (N + 200) * dt;
    t_span = 0 : dt : t_end;

    % Goodwin base — Structure class 1 latent
    % dx/dt =  alpha * H_rep(I) - d1 * x
    % dy/dt =  ks * x - Vmax * H_deg(y)
    % dI/dt =  kp * H_act(x) - kd * I     [saturating driven — class 1]
    ode_rhs = @(t, s) [ ...
        alpha * hill_rep(s(3), hill_k0, hill_n) - d1   * s(1);      ...
        ks    * s(1) - Vmax * hill_deg(s(2), hill_km);               ...
        kp    * hill_act(s(1), hill_ka, hill_na) - kd  * s(3) ];

    opts_ode = odeset('RelTol',1e-9,'AbsTol',1e-11);

    try
        [t_ode, S] = ode45(ode_rhs, t_span, [1.5; 0.5; 1.0], opts_ode);
    catch
        skipped = skipped + 1;
        continue
    end

    if any(~isfinite(S(:)))
        skipped = skipped + 1;
        continue
    end

    n_all      = length(t_ode);
    x_data_all = S(:,1);
    y_data_all = S(:,2);
    z_data_all = S(:,3);

    tail       = round(0.4 * n_all);
    var_z_tail = var(z_data_all(end-tail:end));
    var_x_tail = var(x_data_all(end-tail:end));

    if var_z_tail < 1e-8 || var_x_tail < 1e-8
        skipped = skipped + 1;
        continue
    end

    x_data_all = x_data_all + noise * std(x_data_all) * randn(n_all,1);
    y_data_all = y_data_all + noise * std(y_data_all) * randn(n_all,1);

    x_data = x_data_all(1:N);
    y_data = y_data_all(1:N);

    Theta_full = [ ones(N,1), ...
                   x_data,    x_data.^2, x_data.^3, ...
                   y_data,    y_data.^2, y_data.^3, ...
                   hill_rep(y_data, hill_k0, hill_n), ...
                   hill_deg(y_data, hill_km) ];

    Theta_poly = Theta_full(:, 1:7);

    colscale_full             = vecnorm(Theta_full, 2, 1);
    colscale_full(colscale_full == 0) = 1;
    ThetaN_full               = Theta_full ./ colscale_full;
    colscale_poly             = colscale_full(1:7);
    ThetaN_poly               = Theta_poly  ./ colscale_poly;

    dxdt = sgolayfilt(gradient(x_data, dt), sg_p, sg_f);
    dydt = sgolayfilt(gradient(y_data, dt), sg_p, sg_f);

    targetScaleX = norm(dxdt, 2);
    targetScaleY = norm(dydt, 2);
    dSdtN        = [dxdt / targetScaleX, dydt / targetScaleY];

    XiN         = zeros(9, 2);
    XiN(1:7, :) = ThetaN_poly \ dSdtN;
    XiN_var     = zeros(9, 2);
    XiN_smooth  = XiN;

    wx = ones(7, 1);
    wy = ones(7, 1);

    opts_ls = optimoptions('lsqlin', 'Display', 'off');

    for iter = 1:50
        for ind = 1:2

            switch ind
                case 1
                    current_w = wx;
                    xi_drain  = XiN_smooth([2 3 4], 1);
                    var_drain = XiN_var([2 3 4], 1);
                    [h_val, h_var] = aux_x(xi_drain, var_drain, ...
                        colscale_full, targetScaleX, ...
                        hill_k0, hill_n, x_data, y_data);
                    Hill_reduction = ThetaN_full(:,8) * h_val;
                    dSdtN_ej       = dSdtN(:,1) - Hill_reduction;
                    lb = [0, -inf, -inf, -inf, 0,  0,  0];
                    ub = [0,    0,    0,    0, 0,  0,  0];

                case 2
                    current_w = wy;
                    xi_prod   = XiN_smooth([2 3 4], 2);
                    var_prod  = XiN_var([2 3 4], 2);
                    [h_val, h_var] = aux_y(xi_prod, var_prod, ...
                        colscale_full, targetScaleY, ...
                        hill_km, x_data, y_data);
                    Hill_reduction = ThetaN_full(:,9) * h_val;
                    dSdtN_ej       = dSdtN(:,2) - Hill_reduction;
                    lb = [0,  0,  0,  0, -inf, -inf, -inf];
                    ub = [0, inf, inf, inf,  0,    0,    0];
            end

            ThetaN_w = ThetaN_poly ./ current_w';
            biginds  = current_w < 10;
            XiN_poly = XiN(1:7, ind);

            if any(biginds)
                active = find(biginds);
                XiN_poly(active) = lsqlin( ...
                    ThetaN_w(:, active), dSdtN_ej, ...
                    [], [], [], [], lb(active), ub(active), [], opts_ls);
                XiN_poly(~biginds) = 0;
            end

            XiN(1:7, ind) = XiN_poly;

            hill_idx               = 7 + ind;
            XiN(hill_idx, ind)     = (1-eta)*XiN(hill_idx,ind) + eta*h_val;
            XiN_var(hill_idx, ind) = h_var;

            eps_s = 1e-3; tau0 = 0.05;
            new_w = 1 ./ (tau0^2 * (XiN_poly.^2 + eps_s));
            if ind == 1; wx = 0.7*wx + 0.3*new_w; current_w = wx;
            else;        wy = 0.7*wy + 0.3*new_w; current_w = wy;
            end

            resid     = dSdtN_ej - ThetaN_poly * XiN_poly;
            resid_var = max(var(resid), 1e-12);
            H         = (ThetaN_poly'*ThetaN_poly)/resid_var + diag(current_w);
            XiN_var(1:7, ind) = diag(inv(H));

        end

        XiN_smooth = 0.7*XiN_smooth + 0.3*XiN;

    end

    Xi      = zeros(9, 2);
    Xi(:,1) = (XiN(:,1) * targetScaleX) ./ colscale_full';
    Xi(:,2) = (XiN(:,2) * targetScaleY) ./ colscale_full';
    Xi(abs(Xi) < 0.005) = 0;

    dx_obs = sgolayfilt(gradient(x_data_all, dt), sg_p, sg_f);
    dy_obs = sgolayfilt(gradient(y_data_all, dt), sg_p, sg_f);

    dx_model = zeros(n_all, 1);
    dy_model = zeros(n_all, 1);
    XiX = Xi(:,1);
    XiY = Xi(:,2);

    for k = 2:n_all
        xp = x_data_all(k-1);
        yp = y_data_all(k-1);
        phi = [1, xp, xp^2, xp^3, yp, yp^2, yp^3, ...
               hill_rep(yp, hill_k0, hill_n), ...
               hill_deg(yp, hill_km)];
        dx_model(k) = phi * XiX;
        dy_model(k) = phi * XiY;
    end

    resid_dx = dx_obs - dx_model;
    resid_dy = dy_obs - dy_model;

    % if var(resid_dx) < 0.2 * var(resid_dy)
    %     skipped = skipped + 1;
    %     continue
    % end

    Xi_ternary = sign(Xi) .* (abs(Xi) > 0.005);

    half      = round(n_all/2);
    x_2nd     = x_data_all(half:end);
    y_2nd     = y_data_all(half:end);
    sustained = (var(x_2nd) > 1e-4) && (var(y_2nd) > 1e-4);
    loop_err  = norm([x_2nd(end)-x_2nd(1), y_2nd(end)-y_2nd(1)]) / ...
                (norm([x_2nd(1), y_2nd(1)]) + 1e-8);
    is_closed = loop_err < 0.25;
    seg       = round(length(x_2nd)/3);
    amp_ratio = std(x_2nd(end-seg+1:end)) / max(std(x_2nd(1:seg)), 1e-8);

    if     amp_ratio > 1.25;  amp_trend = 'growing';
    elseif amp_ratio < 0.75;  amp_trend = 'decaying';
    else;                     amp_trend = 'stable';
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

    structure_label = 1;

    params_saved = struct( ...
        'alpha',   alpha,   'd1',      d1,      'ks',    ks,    ...
        'Vmax',    Vmax,    'kp',      kp,      'kd',    kd,    ...
        'hill_n',  hill_n,  'hill_k0', hill_k0, ...
        'hill_km', hill_km, 'hill_na', hill_na, ...
        'hill_ka', hill_ka, 'noise',   noise,   ...
        'system',  'goodwin');

    fname = fullfile(save_dir, sprintf('example_%04d.mat', saved+1));

    save(fname, ...
        'resid_dx',    'resid_dy',   ...
        'Xi_ternary',               ...
        'topology',                 ...
        'structure_label',          ...
        'params_saved',             ...
        'x_data_all',  'y_data_all', ...
        'z_data_all',  't_ode',     ...
        'Xi');

    saved = saved + 1;

    if mod(saved, 50) == 0
        fprintf('Saved %d / %d  (skipped %d, attempts %d)\n', ...
            saved, n_examples, skipped, attempt);
    end

end

fprintf('\nDone. Saved %d examples, skipped %d, total attempts %d\n', ...
    saved, skipped, attempt);

%% ── AUXILIARY FUNCTIONS ──────────────────────────────────────────────────
function [h_val, h_var] = aux_x(xi_drain, var_drain, ...
        colscale_full, targetScaleX, hill_k0, hill_n, x_data, y_data)
    cx  = (xi_drain(1)*targetScaleX) / colscale_full(2);
    cx2 = (xi_drain(2)*targetScaleX) / colscale_full(3);
    cx3 = (xi_drain(3)*targetScaleX) / colscale_full(4);
    avg_drain   = mean(abs(cx*x_data + cx2*x_data.^2 + cx3*x_data.^3));
    h_basis_avg = mean(hill_k0^hill_n ./ (hill_k0^hill_n + y_data.^hill_n));
    force_phys  = avg_drain / max(h_basis_avg, 1e-3);
    h_val       = (force_phys * colscale_full(8)) / targetScaleX;
    sv          = (targetScaleX ./ colscale_full(2:4)') * (colscale_full(8)/targetScaleX);
    h_var       = sum(var_drain .* sv.^2);
end

function [h_val, h_var] = aux_y(xi_prod, var_prod, ...
        colscale_full, targetScaleY, hill_km, x_data, y_data)
    cx  = (xi_prod(1)*targetScaleY) / colscale_full(2);
    cx2 = (xi_prod(2)*targetScaleY) / colscale_full(3);
    cx3 = (xi_prod(3)*targetScaleY) / colscale_full(4);
    avg_prod    = mean(abs(cx*x_data + cx2*x_data.^2 + cx3*x_data.^3));
    h_basis_avg = mean(y_data ./ (hill_km + y_data));
    force_phys  = avg_prod / max(h_basis_avg, 1e-3);
    h_val       = -(force_phys * colscale_full(9)) / targetScaleY;
    sv          = (targetScaleY ./ colscale_full(2:4)') * (colscale_full(9)/targetScaleY);
    h_var       = sum(var_prod .* sv.^2);
end