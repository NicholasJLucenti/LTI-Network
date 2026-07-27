function row = V6_LibRow(x, y, p, libid)
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