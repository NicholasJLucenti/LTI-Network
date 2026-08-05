function [Theta, col_names] = V11_BuildLib(x, y, N_, p, libid)
% Dispatches STRidge candidate-library construction by libid. Unchanged
% from V6_BuildLib.m — still required internally to fit Xi, which the
% nullcline_features block (a kept V11 feature) is derived from.
switch libid
    case 'brusselator'
        Theta=[ones(N_,1),x,x.^2,y,x.^2.*y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
        col_names={'1','x','x^2','y','x^2y','HillRep','HillDeg'};

    case 'fhn'
        Theta=[ones(N_,1),x,x.^3,y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km), ...
            x.*y, x.^2];
        col_names={'1','x','x^3','y','HillRep','HillDeg','xy','x^2'};

    case 'rma'
        Theta=[ones(N_,1),x,x.^2,y,x.*y,x.^2.*y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
        col_names={'1','x','x^2','y','xy','x^2y','HillRep','HillDeg'};

    case 'toggle'
        Theta=[ones(N_,1),x,y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_rep(x,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
        col_names={'1','x','y','HillRep_y','HillRep_x','HillDeg'};

    otherwise % 'standard' — Goodwin/Repressilator/VdP
        Theta=[ones(N_,1),x,x.^2,x.^3,y,y.^2,y.^3,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
        col_names={'1','x','x^2','x^3','y','y^2','y^3','HillRep','HillDeg'};
end
end
