function Phi = V6_LibGrid(x, y, p, libid)
x=x(:);y=y(:);n=length(x);
hr=p.hill_rep(y,p.hill_k0,p.hill_n); hd=p.hill_deg(y,p.hill_km);
switch libid
    case 'brusselator'
        Phi=[ones(n,1),x,x.^2,y,x.^2.*y,hr,hd];
    case 'fhn'
        Phi=[ones(n,1),x,x.^3,y,hr,hd,x.*y,x.^2];
    case 'rma'
        Phi=[ones(n,1),x,x.^2,y,x.*y,x.^2.*y,hr,hd];
    case 'toggle'
        hr_x=p.hill_rep(x,p.hill_k0,p.hill_n);
        Phi=[ones(n,1),x,y,hr,hr_x,hd];
    otherwise
        Phi=[ones(n,1),x,x.^2,x.^3,y,y.^2,y.^3,hr,hd];
end
end