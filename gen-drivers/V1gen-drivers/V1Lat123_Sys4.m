clear; close all; clc;
addpath(genpath('.'));
warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');
hill_rep = @(z,k,n) k^n./(k^n+z.^n);
hill_deg = @(y,km)  y./(km+y);

n_per_class = 250;
base_seed   = 40;

for latent_class = 3:3

    param_sampler = @() struct( ...
        'mu',     0.3+0.7*rand(), ...
        'omega',  0.8+0.4*rand(), ...
        'alpha',  0.5+1.5*rand(), ...
        'kp',     1.5+2.0*rand(), ...
        'kd',     0.3+0.8*rand(), ...
        'kb',     0.1+0.5*rand(), ...
        'hill_n', round(2+3*rand()), ...
        'hill_k0',1+2*rand(), ...
        'hill_km',0.5+1.5*rand(), ...
        'hill_rep', hill_rep, ...
        'hill_deg', hill_deg, ...
        'noise',  0.01+0.09*rand(), ...
        'lb',     {{repmat(-inf,1,7), repmat(-inf,1,7)}}, ...
        'ub',     {{repmat(inf,1,7),  repmat(inf,1,7)}});

    ode_fun = @(t,s,p) vanderpol_ode(t,s,p,latent_class);
    ic      = @(p) [0.5; 0.0; 0.5];
    lib     = @(x,y,N,p) standard_lib(x,y,N,p);
    auxs    = {@aux_x_standard, @aux_y_standard};

    save_dir = sprintf( ...
        'C:/Users/nickj/MATLAB Drive/LTI Network/Lat%d_Sys4_NNdata', ...
        latent_class);

    fprintf('\n=== Van der Pol | Latent Class %d ===\n', latent_class);

    LTInet_DataGen('van_der_pol', ode_fun, ic, ...
        param_sampler, lib, auxs, latent_class, n_per_class, ...
        save_dir, base_seed + latent_class);

end

fprintf('\nAll Van der Pol datasets complete.\n');

%% ── VAN DER POL ODE ──────────────────────────────────────────────────────
function ds = vanderpol_ode(~, s, p, latent_class)
    ds    = zeros(3,1);
    ds(1) = p.mu*(1-s(2)^2)*s(1) - p.omega*s(2) - ...
            p.alpha * p.hill_rep(s(3), p.hill_k0, p.hill_n);
    ds(2) = s(1);
    ds(3) = latent_z(s, p, latent_class);
end

%% ── LATENT EQUATION ──────────────────────────────────────────────────────
function dI = latent_z(s, p, latent_class)
    hill_act = @(x,k,n) x.^n ./ (k^n + x.^n);
    switch latent_class
        case 0
            dI = p.kp*s(1) - p.kd*s(3);
        case 1
            dI = p.kp * hill_act(s(2), p.hill_k0, p.hill_n) - p.kd*s(3);
        case 2
            dI = p.kp*s(1) - p.kb*max(s(1),0)*max(s(3),0) - p.kd*s(3);
    end
end

%% ── STANDARD LIBRARY ─────────────────────────────────────────────────────
function lib = standard_lib(x, y, N, p)
    lib.Theta_full = [ones(N,1), x, x.^2, x.^3, y, y.^2, y.^3, ...
                      p.hill_rep(y, p.hill_k0, p.hill_n), ...
                      p.hill_deg(y, p.hill_km)];
    lib.Theta_poly = lib.Theta_full(:,1:7);
    lib.col_names  = {'1','x','x^2','x^3','y','y^2','y^3', ...
                      'HillRep','HillDeg'};
end

%% ── AUXILIARY FUNCTIONS ──────────────────────────────────────────────────
function [h_val, h_var] = aux_x_standard(XiN_smooth, XiN_var, ...
        colscale_full, targetScaleX, x_data, y_data, p)
    xi   = XiN_smooth([2 3 4], 1);
    vxi  = XiN_var([2 3 4], 1);
    cx   = (xi(1)*targetScaleX) / colscale_full(2);
    cx2  = (xi(2)*targetScaleX) / colscale_full(3);
    cx3  = (xi(3)*targetScaleX) / colscale_full(4);
    avg_drain   = mean(abs(cx*x_data + cx2*x_data.^2 + cx3*x_data.^3));
    h_basis_avg = mean(p.hill_rep(y_data, p.hill_k0, p.hill_n));
    force_phys  = avg_drain / max(h_basis_avg, 1e-3);
    h_val       = (force_phys * colscale_full(end-1)) / targetScaleX;
    sv          = (targetScaleX./colscale_full(2:4)') * ...
                  (colscale_full(end-1)/targetScaleX);
    h_var       = sum(vxi .* sv.^2);
end

function [h_val, h_var] = aux_y_standard(XiN_smooth, XiN_var, ...
        colscale_full, targetScaleY, x_data, y_data, p)
    xi   = XiN_smooth([2 3 4], 2);
    vxi  = XiN_var([2 3 4], 2);
    cx   = (xi(1)*targetScaleY) / colscale_full(2);
    cx2  = (xi(2)*targetScaleY) / colscale_full(3);
    cx3  = (xi(3)*targetScaleY) / colscale_full(4);
    avg_prod    = mean(abs(cx*x_data + cx2*x_data.^2 + cx3*x_data.^3));
    h_basis_avg = mean(p.hill_deg(y_data, p.hill_km));
    force_phys  = avg_prod / max(h_basis_avg, 1e-3);
    h_val       = -(force_phys * colscale_full(end)) / targetScaleY;
    sv          = (targetScaleY./colscale_full(2:4)') * ...
                  (colscale_full(end)/targetScaleY);
    h_var       = sum(vxi .* sv.^2);
end