function row = V11_LibRow(x, y, p, libid)
% Unchanged from V6_LibRow.m — needed to build dx_model, which resid_dx
% (feeding xcorr_features, a kept V11 feature block) is derived from.
switch libid
    case 'brusselator'
        row=[1,x,x^2,y,x^2*y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
    case 'fhn'
        row=[1,x,x^3,y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km),x*y,x^2];
    case 'rma'
        row=[1,x,x^2,y,x*y,x^2*y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
    case 'toggle'
        row=[1,x,y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_rep(x,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
    otherwise
        row=[1,x,x^2,x^3,y,y^2,y^3,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
end
end
