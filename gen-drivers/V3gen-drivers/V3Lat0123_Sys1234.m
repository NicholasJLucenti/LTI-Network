clear; close all; clc;
addpath(genpath('.'));

warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');

% ── Shared Hill function handles ──────────────────────────────────────────
hill_rep = @(z,k,n)  k^n ./ (k^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);
hill_act = @(x,k,n)  x.^n ./ (k^n + x.^n);

dt           = 0.05;
N            = 300;
eta          = 0.6;
sg_p         = 3;
sg_f         = 11;
base_seed    = 42;
n_per_config = 300;

systems = {'goodwin', 'brusselator', 'repressilator', 'van_der_pol'};

% Lat0: linear x    dI/dt = kp*x - kd*I
% Lat1: Hill act x  dI/dt = kp*H_act(x,ka,na) - kd*I
% Lat2: linear y    dI/dt = kp*y - kd*I
% Lat3: Hill act y  dI/dt = kp*H_act(y,ka,na) - kd*I
latent_classes = [0, 1, 2, 3];

for sys_idx = 1:length(systems)
    system_name = systems{sys_idx};

    for lc_idx = 1:length(latent_classes)
        latent_class = latent_classes(lc_idx);

        save_dir = sprintf( ...
            'C:/Users/nickj/MATLAB Drive/LTI Network/V3Lat%d_Sys%d_NNdata', ...
            latent_class, sys_idx);

        if ~exist(save_dir, 'dir'), mkdir(save_dir); end

        fprintf('\n=== V3 | %s | Latent Class %d ===\n', ...
            system_name, latent_class);

        switch system_name
            case 'goodwin'
                param_sampler = @() sample_goodwin(hill_rep, hill_deg, ...
                    hill_act, latent_class);
                ode_fun = @(t,s,p) goodwin_ode(t,s,p,latent_class);
                ic_fun  = @(p) [1.5; 0.5; 1.0];
                lib_fun = @(x,y,N,p) standard_lib(x,y,N,p);

            case 'brusselator'
                param_sampler = @() sample_brusselator(hill_rep, hill_deg, ...
                    hill_act, latent_class);
                ode_fun = @(t,s,p) brusselator_ode(t,s,p,latent_class);
                ic_fun  = @(p) [p.a; p.b/p.a; 0.5];
                lib_fun = @(x,y,N,p) brusselator_lib(x,y,N,p);

            case 'repressilator'
                param_sampler = @() sample_repressilator(hill_rep, hill_deg, ...
                    hill_act, latent_class);
                ode_fun = @(t,s,p) repressilator_ode(t,s,p,latent_class);
                ic_fun  = @(p) [1.0; 0.5; 0.5];
                lib_fun = @(x,y,N,p) standard_lib(x,y,N,p);

            case 'van_der_pol'
                param_sampler = @() sample_vanderpol(hill_rep, hill_deg, ...
                    hill_act, latent_class);
                ode_fun = @(t,s,p) vanderpol_ode(t,s,p,latent_class);
                ic_fun  = @(p) [0.5; 0.0; 0.5];
                lib_fun = @(x,y,N,p) standard_lib(x,y,N,p);
        end

        run_generation(system_name, ode_fun, ic_fun, param_sampler, ...
            lib_fun, latent_class, n_per_config, save_dir, ...
            base_seed + sys_idx*10 + latent_class);
    end
end

fprintf('\nAll V3 Lat0-3 Sys1-4 datasets complete.\n');


%% ── GENERATION LOOP ──────────────────────────────────────────────────────
function run_generation(system_name, ode_fun, ic_fun, param_sampler, ...
        lib_fun, latent_class, n_examples, save_dir, rng_seed)

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

if strcmp(system_name, 'van_der_pol')
    opts_ode = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt);
else
    opts_ode = odeset('RelTol',1e-5,'AbsTol',1e-7,'MaxStep',dt);
end

opts_ls = optimoptions('lsqlin','Display','off');
t_end   = (N+200)*dt;
t_span  = 0:dt:t_end;

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

    % Reject flat trajectories early before paying SINDy cost
    tail = round(0.4*n_all);
    if var(z_data_all(end-tail:end)) < 1e-6 || ...
       var(x_data_all(end-tail:end)) < 1e-6
        skipped = skipped+1; continue
    end

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

    % Auxiliary functions inline for V3 (standard library only)
    for iter = 1:25
        for ind = 1:2
            switch ind
                case 1
                    current_w = wx;
                    xi_s  = XiN_smooth([2 3 4],1);
                    vxi   = XiN_var([2 3 4],1);
                    cx    = (xi_s(1)*targetScaleX)/colscale_full(2);
                    cx2   = (xi_s(2)*targetScaleX)/colscale_full(3);
                    cx3   = (xi_s(3)*targetScaleX)/colscale_full(4);
                    avg_d = mean(abs(cx*x_data+cx2*x_data.^2+cx3*x_data.^3));
                    hb    = mean(p.hill_rep(y_data,p.hill_k0,p.hill_n));
                    fp    = avg_d / max(hb,1e-3);
                    h_val = (fp*colscale_full(end-1)) / targetScaleX;
                    sv    = (targetScaleX./colscale_full(2:4)') * ...
                            (colscale_full(end-1)/targetScaleX);
                    h_var = sum(vxi.*sv.^2);
                    Hill_red  = ThetaN_full(:,end-1)*h_val;
                    dSdtN_ej  = dSdtN(:,1) - Hill_red;
                    lb = p.lb{1};  ub = p.ub{1};
                case 2
                    current_w = wy;
                    xi_s  = XiN_smooth([2 3 4],2);
                    vxi   = XiN_var([2 3 4],2);
                    cx    = (xi_s(1)*targetScaleY)/colscale_full(2);
                    cx2   = (xi_s(2)*targetScaleY)/colscale_full(3);
                    cx3   = (xi_s(3)*targetScaleY)/colscale_full(4);
                    avg_d = mean(abs(cx*x_data+cx2*x_data.^2+cx3*x_data.^3));
                    hb    = mean(p.hill_deg(y_data,p.hill_km));
                    fp    = avg_d / max(hb,1e-3);
                    h_val = -(fp*colscale_full(end)) / targetScaleY;
                    sv    = (targetScaleY./colscale_full(2:4)') * ...
                            (colscale_full(end)/targetScaleY);
                    h_var = sum(vxi.*sv.^2);
                    Hill_red  = ThetaN_full(:,end)*h_val;
                    dSdtN_ej  = dSdtN(:,2) - Hill_red;
                    lb = p.lb{2};  ub = p.ub{2};
            end

            ThetaN_w     = ThetaN_poly ./ current_w';
            biginds      = current_w < 10;
            XiN_poly_vec = XiN(1:n_poly,ind);

            if any(biginds)
                active = find(biginds);
                XiN_poly_vec(active) = lsqlin( ...
                    ThetaN_w(:,active), dSdtN_ej, ...
                    [],[],[],[], lb(active), ub(active), [], opts_ls);
                XiN_poly_vec(~biginds) = 0;
            end

            XiN(1:n_poly,ind)  = XiN_poly_vec;
            hill_idx           = n_poly+ind;
            XiN(hill_idx,ind)  = (1-eta)*XiN(hill_idx,ind)+eta*h_val;
            XiN_var(hill_idx,ind) = h_var;

            eps_s = 1e-3; tau0 = 0.05;
            new_w = 1./(tau0^2*(XiN_poly_vec.^2+eps_s));
            if ind==1; wx=0.7*wx+0.3*new_w; current_w=wx;
            else;      wy=0.7*wy+0.3*new_w; current_w=wy;
            end

            resid     = dSdtN_ej - ThetaN_poly*XiN_poly_vec;
            resid_var = max(var(resid),1e-12);
            H_mat     = (ThetaN_poly'*ThetaN_poly)/resid_var + diag(current_w);
            XiN_var(1:n_poly,ind) = diag(inv(H_mat));
        end
        XiN_smooth = 0.7*XiN_smooth + 0.3*XiN;
    end

    % Denormalise
    Xi      = zeros(n_terms,2);
    Xi(:,1) = (XiN(:,1)*targetScaleX) ./ colscale_full';
    Xi(:,2) = (XiN(:,2)*targetScaleY) ./ colscale_full';
    Xi(abs(Xi)<0.005) = 0;

    %% ── NULLCLINE FEATURE EXTRACTION ────────────────────────────────────
    % Build a dense grid over the observed (x,y) range
    x_min = min(x_data); x_max = max(x_data);
    y_min = min(y_data); y_max = max(y_data);
    n_grid = 80;
    xg = linspace(x_min, x_max, n_grid);
    yg = linspace(y_min, y_max, n_grid);
    [XG, YG] = meshgrid(xg, yg);

    XiX = Xi(:,1);
    XiY = Xi(:,2);

    % Evaluate fitted dx/dt and dy/dt over the grid (vectorised)
    Phi_grid = lib_matrix_grid(XG(:), YG(:), p, system_name);
    dX_grid  = reshape(Phi_grid * XiX, n_grid, n_grid);
    dY_grid  = reshape(Phi_grid * XiY, n_grid, n_grid);

    % ── Group 3: Fixed point (nullcline intersection) ─────────────────
    % Estimate as the (x,y) point minimising |dx/dt|^2 + |dy/dt|^2
    combined = dX_grid.^2 + dY_grid.^2;
    [~, min_idx] = min(combined(:));
    [r_fp, c_fp] = ind2sub([n_grid, n_grid], min_idx);
    x_fp = xg(c_fp);
    y_fp = yg(r_fp);

    % Amplitude normalisation factor
    amp_x = max(x_data) - min(x_data);
    amp_y = max(y_data) - min(y_data);
    amp_norm = sqrt(amp_x^2 + amp_y^2);

    % Limit cycle centroid
    x_cent = mean(x_data);
    y_cent = mean(y_data);

    fp_x_norm   = (x_fp - x_cent) / (amp_x + 1e-8);
    fp_y_norm   = (y_fp - y_cent) / (amp_y + 1e-8);
    fp_dist     = sqrt((x_fp-x_cent)^2+(y_fp-y_cent)^2) / (amp_norm+1e-8);

    % Crossing angle between nullclines at fixed point
    % Approximate gradient directions from finite differences on the grid
    dx_fp_gx = (dX_grid(r_fp, min(c_fp+1,n_grid)) - ...
                dX_grid(r_fp, max(c_fp-1,1))) / (2*(xg(2)-xg(1))+1e-8);
    dx_fp_gy = (dX_grid(min(r_fp+1,n_grid), c_fp) - ...
                dX_grid(max(r_fp-1,1), c_fp)) / (2*(yg(2)-yg(1))+1e-8);
    dy_fp_gx = (dY_grid(r_fp, min(c_fp+1,n_grid)) - ...
                dY_grid(r_fp, max(c_fp-1,1))) / (2*(xg(2)-xg(1))+1e-8);
    dy_fp_gy = (dY_grid(min(r_fp+1,n_grid), c_fp) - ...
                dY_grid(max(r_fp-1,1), c_fp)) / (2*(yg(2)-yg(1))+1e-8);

    % Nullcline tangents are perpendicular to gradients
    nx1 = [-dx_fp_gy,  dx_fp_gx] / (norm([-dx_fp_gy, dx_fp_gx])+1e-8);
    nx2 = [-dy_fp_gy,  dy_fp_gx] / (norm([-dy_fp_gy, dy_fp_gx])+1e-8);
    cross_angle = acos(min(abs(dot(nx1,nx2)),1));  % [0, pi/2]

    % Group 3 vector: [fp_x_norm, fp_y_norm, fp_dist, cross_angle]
    group3 = [fp_x_norm, fp_y_norm, fp_dist, cross_angle];

    % ── Groups 1 & 2: Nullcline shape moments ─────────────────────────
    % Extract zero-crossing contour points for x and y nullclines
    % using sign-change detection on the grid (avoids contourc complexity)
    [xnull_pts, ynull_x] = extract_nullcline_pts(dX_grid, xg, yg);
    [xnull_y,   ynull_pts] = extract_nullcline_pts(dY_grid, xg, yg);

    % Normalise contour points to unit range
    xnull_pts_n = (xnull_pts - x_cent) / (amp_x+1e-8);
    ynull_x_n   = (ynull_x  - y_cent) / (amp_y+1e-8);
    xnull_y_n   = (xnull_y  - x_cent) / (amp_x+1e-8);
    ynull_pts_n = (ynull_pts - y_cent) / (amp_y+1e-8);

    group1 = nullcline_shape_features(xnull_pts_n, ynull_x_n);   % x-nullcline
    group2 = nullcline_shape_features(xnull_y_n,   ynull_pts_n); % y-nullcline

    % ── Group 4: Orbit-nullcline signed distance ───────────────────────
    % Evaluate fitted equations along the observed trajectory
    Phi_orbit  = lib_matrix_grid(x_data, y_data, p, system_name);
    dx_orbit   = Phi_orbit * XiX;
    dy_orbit   = Phi_orbit * XiY;

    mean_dx_orb = mean(dx_orbit) / (targetScaleX+1e-8);
    std_dx_orb  = std(dx_orbit)  / (targetScaleX+1e-8);
    mean_dy_orb = mean(dy_orbit) / (targetScaleY+1e-8);
    std_dy_orb  = std(dy_orbit)  / (targetScaleY+1e-8);

    % Phase of maximum orbit-nullcline deviation (fraction of cycle)
    [~, max_dev_idx] = max(abs(dx_orbit));
    phase_max_dev    = max_dev_idx / N;

    group4 = [mean_dx_orb, std_dx_orb, mean_dy_orb, std_dy_orb, phase_max_dev];

    % ── Group 6: Nullcline crossing counts ────────────────────────────
    % Count sign changes of dx/dt and dy/dt along the observed orbit
    sgn_dx  = sign(dx_orbit);
    sgn_dy  = sign(dy_orbit);
    cross_x = sum(abs(diff(sgn_dx)) > 0);
    cross_y = sum(abs(diff(sgn_dy)) > 0);

    % Directional asymmetry: +to- vs -to+ crossings
    pos2neg_x = sum(diff(sgn_dx) < 0);
    neg2pos_x = sum(diff(sgn_dx) > 0);
    asym_x    = (pos2neg_x - neg2pos_x) / (cross_x+1e-8);

    pos2neg_y = sum(diff(sgn_dy) < 0);
    neg2pos_y = sum(diff(sgn_dy) > 0);
    asym_y    = (pos2neg_y - neg2pos_y) / (cross_y+1e-8);

    % Normalise crossing counts by number of oscillation cycles
    % (estimate: number of x sign changes / 2)
    n_cycles = max(sum(abs(diff(sign(x_data - mean(x_data)))) > 0)/2, 1);
    cross_x_norm = cross_x / n_cycles;
    cross_y_norm = cross_y / n_cycles;

    group6 = [cross_x_norm, asym_x, cross_y_norm, asym_y];

    % ── Cross-nullcline comparison features (Group 7) ───────────────
    % Directly encodes which nullcline is more curved / asymmetric.
    % This is the key signal that separates x-driven from y-driven
    % Hill activation — Hact-y distorts the y-nullcline more than the
    % x-nullcline, producing a distinctive asymmetry in these diffs.
    %
    % group1 and group2 are both 12-element vectors with identical
    % structure: [cx, cy, pca_angle, aspect, curv_q1..q4,
    %             asym_along, asym_across, arc_len, spread_ratio]
    %
    % Differences that carry x-vs-y directionality:
    %   aspect diff      — which nullcline is more elongated
    %   curvature diffs  — which nullcline has sharper inflections
    %   asym_along diff  — which nullcline is more asymmetric
    %   arc_len diff     — which nullcline spans more of the phase plane
    %   spread_ratio diff — relative spread perpendicular to principal axis

    g1 = group1(:);
    g2 = group2(:);

    % Element-wise difference (x-nullcline minus y-nullcline)
    % Indices: 4=aspect, 5-8=curvature quartiles, 9=asym_along,
    %          10=asym_across, 11=arc_len, 12=spread_ratio
    diff_aspect    = g1(4)  - g2(4);
    diff_curv      = g1(5:8) - g2(5:8);   % 4 values
    diff_asym_a    = g1(9)  - g2(9);
    diff_asym_b    = g1(10) - g2(10);
    diff_arc       = g1(11) - g2(11);
    diff_spread    = g1(12) - g2(12);

    % Ratio features — which nullcline has more curvature overall
    % Clipped to [-5, 5] to prevent explosion when denominator is tiny
    ratio_aspect   = clip_ratio(g1(4),  g2(4));
    ratio_arc      = clip_ratio(g1(11), g2(11));
    ratio_spread   = clip_ratio(g1(12), g2(12));

    group7 = [diff_aspect; diff_curv; diff_asym_a; diff_asym_b; ...
              diff_arc; diff_spread; ...
              ratio_aspect; ratio_arc; ratio_spread];   % 12 values

    % ── Assemble full feature vector ──────────────────────────────────
    % Group1(12) + Group2(12) + Group3(4) + Group4(5) + Group6(4)
    % + Group7(12) = 49 total
    nullcline_features = [group1(:)', group2(:)', group3(:)', ...
                          group4(:)', group6(:)', group7(:)']';

    % Hard clip before saving — prevents float32 overflow in Python
    nullcline_features = max(min(nullcline_features, 50), -50);

    % Catch any NaN/Inf from degenerate fits
    if any(~isfinite(nullcline_features))
        skipped = skipped+1; continue
    end

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
                (norm([x_2nd(1), y_2nd(1)])+1e-8);
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

    structure_label  = latent_class;   % 0-indexed, no offset
    params_saved     = rmfield_safe(p, {'hill_rep','hill_deg','hill_act','lb','ub'});
    params_saved.system = system_name;

    fname = fullfile(save_dir, ...
        sprintf('example_%04d.mat', existing+saved+1));

    save(fname, ...
        'nullcline_features', ...   % (43x1) — replaces fft+residual
        'Xi_ternary',         ...
        'topology',           ...
        'structure_label',    ...
        'params_saved',       ...
        'x_data_all', 'y_data_all', ...
        'z_data_all', 't_ode',      ...
        'Xi',         'col_names');

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
    ds(1) = p.a-(p.b+1)*s(1)+s(1)^2*s(2) - ...
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

function dI = latent_z(s, p, latent_class)
    switch latent_class
        case 0
            dI = p.kp*s(1) - p.kd*s(3);
        case 1
            dI = p.kp * p.hill_act(s(1), p.hill_ka, p.hill_na) - p.kd*s(3);
        case 2
            dI = p.kp*s(2) - p.kd*s(3);
        case 3
            dI = p.kp * p.hill_act(s(2), p.hill_ka, p.hill_na) - p.kd*s(3);
    end
end


%% ── PARAM SAMPLERS ───────────────────────────────────────────────────────
function p = sample_goodwin(hill_rep, hill_deg, hill_act, ~)
    p = struct( ...
        'alpha',   5+7*rand(),        'd1',    0.3+0.7*rand(), ...
        'ks',      1+3*rand(),        'Vmax',  3+5*rand(),     ...
        'hill_n',  round(3+3*rand()), ...
        'hill_k0', 1.5+2.5*rand(),    ...
        'hill_km', 0.4+1.6*rand(),    ...
        'hill_ka', 1.5+1.0*rand(),    ...
        'hill_na', round(1+2*rand()), ...
        'kp',      1+2*rand(),        ...
        'kd',      0.5+1.5*rand(),    ...
        'hill_rep', hill_rep, 'hill_deg', hill_deg, 'hill_act', hill_act, ...
        'noise',   0.01+0.09*rand(),  ...
        'lb', {{[0,-inf,-inf,-inf,0,0,0],[0,0,0,0,-inf,-inf,-inf]}}, ...
        'ub', {{[0,0,0,0,0,0,0],[0,inf,inf,inf,0,0,0]}});
end

function p = sample_brusselator(hill_rep, hill_deg, hill_act, ~)
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
        'kp',      1+2*rand(),        ...
        'kd',      0.5+1.5*rand(),    ...
        'hill_rep', hill_rep, 'hill_deg', hill_deg, 'hill_act', hill_act, ...
        'noise',   0.01+0.09*rand(),  ...
        'lb', {{repmat(-inf,1,5), repmat(-inf,1,5)}}, ...
        'ub', {{repmat(inf,1,5),  repmat(inf,1,5)}});
end

function p = sample_repressilator(hill_rep, hill_deg, hill_act, ~)
    p = struct( ...
        'alpha',  4+6*rand(),         'delta', 1+2*rand(),   ...
        'km',     0.3+1.2*rand(),     ...
        'beta',   1+2*rand(),         'gamma', 0.5+1.5*rand(), ...
        'hill_n', round(2+3*rand()),  ...
        'hill_k0',1+3*rand(),         'hill_km', 0.4+1.6*rand(), ...
        'hill_ka',0.5+2.0*rand(),     'hill_na', round(1+2*rand()), ...
        'kp',     1+2*rand(),         ...
        'kd',     0.5+1.5*rand(),     ...
        'hill_rep', hill_rep, 'hill_deg', hill_deg, 'hill_act', hill_act, ...
        'noise',   0.01+0.09*rand(),  ...
        'lb', {{[0,-inf,-inf,-inf,0,0,0],[0,0,0,0,-inf,-inf,-inf]}}, ...
        'ub', {{[0,0,0,0,0,0,0],[0,inf,inf,inf,0,0,0]}});
end

function p = sample_vanderpol(hill_rep, hill_deg, hill_act, ~)
    p = struct( ...
        'mu',    0.3+0.7*rand(), 'omega', 0.8+0.4*rand(), ...
        'alpha', 0.5+1.5*rand(), ...
        'hill_n', round(2+3*rand()), ...
        'hill_k0',1+2*rand(),  'hill_km', 0.5+1.5*rand(), ...
        'hill_ka',0.5+2.0*rand(), 'hill_na', round(1+2*rand()), ...
        'kp',    1+2*rand(),   ...
        'kd',    0.5+1.5*rand(), ...
        'hill_rep', hill_rep, 'hill_deg', hill_deg, 'hill_act', hill_act, ...
        'noise',   0.01+0.09*rand(), ...
        'lb', {{repmat(-inf,1,7), repmat(-inf,1,7)}}, ...
        'ub', {{repmat(inf,1,7),  repmat(inf,1,7)}});
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

% Vectorised library for grid evaluation — returns (n_pts x n_terms)
function Phi = lib_matrix_grid(x, y, p, system_name)
    x = x(:); y = y(:);
    n = length(x);
    hr = p.hill_rep(y, p.hill_k0, p.hill_n);
    hd = p.hill_deg(y, p.hill_km);
    if strcmp(system_name, 'brusselator')
        Phi = [ones(n,1), x, x.^2, y, x.^2.*y, hr, hd];
    else
        Phi = [ones(n,1), x, x.^2, x.^3, y, y.^2, y.^3, hr, hd];
    end
end


%% ── NULLCLINE HELPER FUNCTIONS ───────────────────────────────────────────

% Extract nullcline point cloud from a grid field via sign-change detection
% Returns x and y coordinates of zero-crossing points
function [xpts, ypts] = extract_nullcline_pts(F, xg, yg)
    xpts = [];
    ypts = [];
    n_grid = length(xg);
    % Horizontal sign changes (between adjacent columns)
    for r = 1:n_grid
        for c = 1:n_grid-1
            if F(r,c)*F(r,c+1) < 0
                % Linear interpolation to find zero crossing
                t    = F(r,c) / (F(r,c) - F(r,c+1));
                xpts = [xpts; xg(c)+t*(xg(c+1)-xg(c))]; %#ok<AGROW>
                ypts = [ypts; yg(r)]; %#ok<AGROW>
            end
        end
    end
    % Vertical sign changes (between adjacent rows)
    for r = 1:n_grid-1
        for c = 1:n_grid
            if F(r,c)*F(r+1,c) < 0
                t    = F(r,c) / (F(r,c) - F(r+1,c));
                xpts = [xpts; xg(c)]; %#ok<AGROW>
                ypts = [ypts; yg(r)+t*(yg(r+1)-yg(r))]; %#ok<AGROW>
            end
        end
    end
    if isempty(xpts)
        xpts = [0]; ypts = [0];  % fallback for degenerate cases
    end
end

% Compute 12 shape features from a nullcline point cloud (Groups 1 & 2)
% Features: centroid(2), PCA angle(1), aspect ratio(1),
%           curvature proxy(4 percentiles), asymmetry(2),
%           arc length proxy(1), spread ratio(1) = 12 total
function feat = nullcline_shape_features(xpts, ypts)
    if length(xpts) < 4
        feat = zeros(12,1);
        return
    end

    % Centroid (already normalised upstream)
    cx = mean(xpts);
    cy = mean(ypts);

    % PCA of point cloud
    pts  = [xpts(:) - cx, ypts(:) - cy];
    C    = (pts'*pts) / max(size(pts,1)-1, 1);
    [V, D] = eig(C);
    evals  = diag(D);
    [evals_s, ord] = sort(evals, 'descend');
    V = V(:, ord);

    % Principal axis angle (angle of first eigenvector)
    pca_angle = atan2(V(2,1), V(1,1));

    % Aspect ratio of bounding ellipse
    aspect = sqrt(evals_s(1)) / (sqrt(evals_s(2)) + 1e-8);
    aspect = min(aspect, 20);   % cap to avoid extreme values

    % Curvature proxy: distance of points from the principal axis line
    proj_along  = pts * V(:,1);   % projection onto major axis
    proj_across = pts * V(:,2);   % projection onto minor axis

    % Divide major axis into 4 quartiles and measure spread in each
    q_bounds = quantile(proj_along, [0 0.25 0.5 0.75 1.0]);
    curv_feat = zeros(4,1);
    for q = 1:4
        mask = proj_along >= q_bounds(q) & proj_along < q_bounds(q+1);
        if sum(mask) > 1
            curv_feat(q) = std(proj_across(mask));
        end
    end

    % Asymmetry: difference in spread between positive and negative halves
    % along each axis
    pos_mask = proj_along > 0;
    neg_mask = proj_along <= 0;
    asym_along  = std(proj_along(pos_mask)+1e-8) / ...
                  (std(proj_along(neg_mask))+1e-8) - 1;
    asym_across = mean(proj_across(pos_mask)) - mean(proj_across(neg_mask));
    asym_along  = max(min(asym_along, 5), -5);   % clip outliers

    % Arc length proxy (mean nearest-neighbour distance × n_pts)
    if length(xpts) > 2
        D_nn  = sqrt(diff(xpts(:)).^2 + diff(ypts(:)).^2);
        arc_len = sum(D_nn);
    else
        arc_len = 0;
    end

    % Spread ratio (std of across vs along projections)
    spread_ratio = std(proj_across) / (std(proj_along)+1e-8);
    spread_ratio = min(spread_ratio, 5);

    feat = [cx; cy; pca_angle; aspect; ...
            curv_feat; ...
            asym_along; asym_across; ...
            arc_len; spread_ratio];
end


%% ── CLIP RATIO HELPER ───────────────────────────────────────────────────
function r = clip_ratio(a, b)
    % Safe ratio clipped to [-5, 5] to prevent explosion
    if abs(b) < 1e-6
        r = 0;
    else
        r = max(min(a/b, 5), -5);
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