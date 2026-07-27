function [Theta, col_names] = V6_BuildLib(x, y, N_, p, libid)
% Dispatches library construction by libid. Extends the V5 set
% ('standard', 'brusselator') with three new forms: 'fhn', 'rma', 'toggle'.
switch libid
    case 'brusselator'
        Theta=[ones(N_,1),x,x.^2,y,x.^2.*y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
        col_names={'1','x','x^2','y','x^2y','HillRep','HillDeg'};

    case 'fhn'
        % Matches dv/dt = v - v^3/3 - w + I_ext structurally: 1, v, v^3, w
        % plus Hill terms so latent-coupling library terms remain available.
        Theta=[ones(N_,1),x,x.^3,y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km), ...
            x.*y, x.^2];
        col_names={'1','x','x^3','y','HillRep','HillDeg','xy','x^2'};

    case 'rma'
        % Matches logistic-growth + saturating-predation structure:
        % 1, x, x^2 (logistic), x*y (mass-action), and a saturating term
        % approximated in the polynomial library via x^2*y (STRidge will
        % zero out unneeded terms) plus Hill terms for latent coupling.
        Theta=[ones(N_,1),x,x.^2,y,x.*y,x.^2.*y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
        col_names={'1','x','x^2','y','xy','x^2y','HillRep','HillDeg'};

    case 'toggle'
        % Matches saturating mutual-repression structure. True dynamics are
        % rational (1/(1+y^beta)), not polynomial — library uses Hill-repressor
        % terms directly (n taken from p.hill_n / p.beta-adjacent sampling)
        % rather than a polynomial expansion, since polynomial truncation
        % would badly misfit the rational nullcline shape.
        Theta=[ones(N_,1),x,y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_rep(x,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
        col_names={'1','x','y','HillRep_y','HillRep_x','HillDeg'};

    otherwise % 'standard' — Goodwin/Repressilator/VdP
        Theta=[ones(N_,1),x,x.^2,x.^3,y,y.^2,y.^3,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
        col_names={'1','x','x^2','x^3','y','y^2','y^3','HillRep','HillDeg'};
end
end