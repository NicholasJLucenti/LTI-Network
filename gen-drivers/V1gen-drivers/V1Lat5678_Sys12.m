clear; close all; clc;
addpath(genpath('.'));

warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');

hill_rep = @(z,k,n)  k^n ./ (k^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);

dt   = 0.05;
N    = 300;
eta  = 0.6;
sg_p = 3;
sg_f = 11;

n_per_config = 250;
base_seed    = 100;

systems  = {'goodwin', 'brusselator'};
n_latent = 4;

for sys_idx = 2:2
    system_name = systems{sys_idx};

    for lat_offset = 0:n_latent-1
        latent_class = 5 + lat_offset;

        save_dir = sprintf( ...
            'C:/Users/nickj/MATLAB Drive/LTI Network/Lat%d_Sys%d_NNdata', ...
            latent_class, sys_idx);

        if ~exist(save_dir,'dir'), mkdir(save_dir); end

        fprintf('\n=== %s | Latent Class %d ===\n', system_name, latent_class);

        switch system_name
            case 'goodwin'
                param_sampler = @() sample_goodwin(hill_rep, hill_deg, latent_class);
                ode_fun       = @(t,s,p) goodwin_ode(t,s,p,latent_class);
                ic_fun        = @(p) [1.5; 0.5; 1.0];
                lib_fun       = @(x,y,N,p) standard_lib(x,y,N,p);
                aux_funs      = {@aux_x_standard, @aux_y_standard};

            case 'brusselator'
                param_sampler = @() sample_brusselator(hill_rep, hill_deg, latent_class);
                ode_fun       = @(t,s,p) brusselator_ode(t,s,p,latent_class);
                ic_fun        = @(p) [p.a; p.b/p.a; 0.5];
                lib_fun       = @(x,y,N,p) brusselator_lib(x,y,N,p);
                aux_funs      = {@aux_x_bruss, @aux_y_bruss};
        end

        run_generation(system_name, ode_fun, ic_fun, param_sampler, ...
            lib_fun, aux_funs, latent_class, n_per_config, save_dir, ...
            base_seed + sys_idx*10 + lat_offset);
    end
end

fprintf('\nAll new latent structure datasets complete.\n');

%% ── GENERATION LOOP ──────────────────────────────────────────────────────
function run_generation(system_name, ode_fun, ic_fun, param_sampler, ...
        lib_fun, aux_funs, latent_class, n_examples, save_dir, rng_seed)

rng(rng_seed);
existing = length(dir(fullfile(save_dir,'*.mat')));

dt   = 0.05;
N    = 300;
eta  = 0.6;
sg_p = 3;
sg_f = 11;

saved   = 0;
skipped = 0;
attempt = 0;

opts_ls  = optimoptions('lsqlin','Display','off');
opts_ode = odeset('RelTol',1e-4,'AbsTol',1e-6);

t_end  = (N+200)*dt;
t_span = 0:dt:t_end;

while saved < n_examples

    attempt = attempt+1;
    p       = param_sampler();

    try
        [t_ode,S] = ode15s(@(t,s) ode_fun(t,s,p), t_span, ic_fun(p), opts_ode);
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

    tail = round(0.4*n_all);
    if var(z_data_all(end-tail:end)) < 1e-6 || ...
       var(x_data_all(end-tail:end)) < 1e-6
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

    colscale_full              = vecnorm(Theta_full,2,1);
    colscale_full(colscale_full==0) = 1;
    ThetaN_full                = Theta_full ./ colscale_full;
    ThetaN_poly                = Theta_poly ./ colscale_full(1:n_poly);

    dxdt = sgolayfilt(gradient(x_data,dt), sg_p, sg_f);
    dydt = sgolayfilt(gradient(y_data,dt), sg_p, sg_f);

    targetScaleX = norm(dxdt,2);
    targetScaleY = norm(dydt,2);
    dSdtN        = [dxdt/targetScaleX, dydt/targetScaleY];

    XiN         = zeros(n_terms,2);
    XiN(1:n_poly,:) = ThetaN_poly \ dSdtN;
    XiN_var     = zeros(n_terms,2);
    XiN_smooth  = XiN;
    wx = ones(n_poly,1);
    wy = ones(n_poly,1);

    aux_x_fun = aux_funs{1};
    aux_y_fun = aux_funs{2};

    for iter = 1:50
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

    dx_model = zeros(n_all,1);
    dy_model = zeros(n_all,1);
    XiX = Xi(:,1); XiY = Xi(:,2);

    for k = 2:n_all
        xp  = x_data_all(k-1);
        yp  = y_data_all(k-1);
        phi = lib_row(xp,yp,p,system_name);
        dx_model(k) = phi*XiX;
        dy_model(k) = phi*XiY;
    end

    resid_dx = dx_obs - dx_model;
    resid_dy = dy_obs - dy_model;

    if var(resid_dx) < 0.5*var(resid_dy)
        skipped = skipped+1; continue
    end

    Xi_ternary = sign(Xi).*(abs(Xi)>0.005);
    if size(Xi_ternary,1) < 9
        Xi_ternary = [Xi_ternary; zeros(9-size(Xi_ternary,1),2)];
    end

    half      = round(n_all/2);
    x_2nd     = x_data_all(half:end);
    y_2nd     = y_data_all(half:end);
    sustained = (var(x_2nd)>1e-4) && (var(y_2nd)>1e-4);
    loop_err  = norm([x_2nd(end)-x_2nd(1),y_2nd(end)-y_2nd(1)]) / ...
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

    structure_label  = latent_class;
    params_saved     = p;
    params_saved.system = system_name;

    fname = fullfile(save_dir, ...
        sprintf('example_%04d.mat', existing+saved+1));

    save(fname, ...
        'resid_dx',    'resid_dy',   ...
        'Xi_ternary',               ...
        'topology',                 ...
        'structure_label',          ...
        'params_saved',             ...
        'x_data_all',  'y_data_all', ...
        'z_data_all',  't_ode',     ...
        'Xi',          'col_names');

    saved = saved+1;

    if mod(saved,50)==0
        fprintf('[%s | Lat%d] Saved %d/%d  (skipped %d, attempts %d)\n', ...
            system_name, latent_class, saved, n_examples, skipped, attempt);
    end
end

fprintf('[%s | Lat%d] Done. Saved %d, skipped %d, attempts %d\n', ...
    system_name, latent_class, saved, skipped, attempt);
end

%% ── ODE DEFINITIONS ──────────────────────────────────────────────────────
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

function dI = latent_z(s, p, latent_class)
    switch latent_class
        case 4
            % Michaelis-Menten enzyme complex
            % dI/dt = kf*x*(Etot - I) - kr*I - kcat*I
            dI = p.kf*s(1)*(p.Etot - s(3)) - p.kr*s(3) - p.kcat*s(3);

        case 5
            % Nuclear-cytoplasmic transport
            % dI/dt = kimport*x - kexport*I - kbind*I*y
            dI = p.kimport*s(1) - p.kexport*s(3) - p.kbind*s(3)*s(2);

        case 6
            % microRNA mediated repression
            % dI/dt = ktranscription*x - kbind*I*y - kdecay*I
            dI = p.ktrans*s(1) - p.kbind*s(3)*s(2) - p.kdecay*s(3);

        case 7
            % Positive feedback amplifier
            % dI/dt = ka*x*I + kbasal*x - kd*I
            dI = p.ka*s(1)*s(3) + p.kbasal*s(1) - p.kd*s(3);
    end
end

%% ── PARAM SAMPLERS ───────────────────────────────────────────────────────
function p = sample_goodwin(hill_rep, hill_deg, latent_class)
    p = struct( ...
        'alpha',    5+7*rand(), ...
        'd1',       0.3+0.7*rand(), ...
        'ks',       1+3*rand(), ...
        'Vmax',     3+5*rand(), ...
        'hill_n',   round(3+3*rand()), ...
        'hill_k0',  1.5+2.5*rand(), ...
        'hill_km',  0.4+1.6*rand(), ...
        'hill_rep', hill_rep, ...
        'hill_deg', hill_deg, ...
        'noise',    0.01+0.11*rand(), ...
        'lb', {{[0,-inf,-inf,-inf,0,0,0],[0,0,0,0,-inf,-inf,-inf]}}, ...
        'ub', {{[0,0,0,0,0,0,0],[0,inf,inf,inf,0,0,0]}});

    switch latent_class
        case 4
            p.kf   = 1+3*rand();
            p.kr   = 0.5+1.5*rand();
            p.kcat = 0.3+1.0*rand();
            p.Etot = 0.5+2.0*rand();
        case 5
            p.kimport = 1+3*rand();
            p.kexport = 0.5+2.0*rand();
            p.kbind   = 0.1+0.8*rand();
        case 6
            p.ktrans  = 1+3*rand();
            p.kbind   = 0.2+1.0*rand();
            p.kdecay  = 0.3+1.5*rand();
        case 7
            p.ka     = 0.1+0.5*rand();
            p.kbasal = 0.5+2.0*rand();
            p.kd     = 1+2*rand();
    end
end

function p = sample_brusselator(hill_rep, hill_deg, latent_class)
    while true
        a = 1+2*rand(); b = 4+6*rand();
        if b > a^2+1; break; end
    end
    p = struct( ...
        'a', a, 'b', b, ...
        'hill_n',  round(2+3*rand()), ...
        'hill_k0', 1.5+2.5*rand(), ...
        'hill_km', 0.5+1.5*rand(), ...
        'hill_rep', hill_rep, ...
        'hill_deg', hill_deg, ...
        'noise',   0.01+0.09*rand(), ...
        'lb', {{repmat(-inf,1,5), repmat(-inf,1,5)}}, ...
        'ub', {{repmat(inf,1,5),  repmat(inf,1,5)}});

    switch latent_class
        case 4
            p.kf   = 1+3*rand();
            p.kr   = 0.5+1.5*rand();
            p.kcat = 0.3+1.0*rand();
            p.Etot = 0.5+2.0*rand();
        case 5
            p.kimport = 1+3*rand();
            p.kexport = 0.5+2.0*rand();
            p.kbind   = 0.1+0.8*rand();
        case 6
            p.ktrans  = 1+3*rand();
            p.kbind   = 0.2+1.0*rand();
            p.kdecay  = 0.3+1.5*rand();
        case 7
            p.ka     = 0.1+0.5*rand();
            p.kbasal = 0.5+2.0*rand();
            p.kd     = 1+2*rand();
    end
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

function row = lib_row(x, y, p, system_name)
    if strcmp(system_name,'brusselator')
        row = [1, x, x^2, y, x^2*y, ...
               p.hill_rep(y,p.hill_k0,p.hill_n), ...
               p.hill_deg(y,p.hill_km)];
    else
        row = [1, x, x^2, x^3, y, y^2, y^3, ...
               p.hill_rep(y,p.hill_k0,p.hill_n), ...
               p.hill_deg(y,p.hill_km)];
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