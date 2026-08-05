function S = V11_SystemLib()
hill_rep = @(z,k,n)  k.^n ./ (k.^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);
hill_act = @(x,k,n)  x.^n ./ (k.^n + x.^n);

S = struct('name',{},'psamp',{},'odefn_base',{},'ic_fun',{},'libid',{}, ...
           'lib_ncols',{},'coupling_site',{},'multi_fp',{});

S(1).name = 'goodwin';
S(1).psamp = @() struct('alpha',5+7*rand(),'d1',0.3+0.7*rand(), ...
    'ks',1+3*rand(),'Vmax',3+5*rand(), ...
    'hill_n',round(3+5*rand()),'hill_k0',1.5+2.5*rand(), ...
    'hill_km',0.4+1.6*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
S(1).odefn_base = @(t,s,p) [p.alpha*p.hill_rep(s(2),p.hill_k0,p.hill_n)-p.d1*s(1);
                            p.ks*s(1)-p.Vmax*p.hill_deg(s(2),p.hill_km)];
S(1).ic_fun = @(p) [1.5; 0.5];
S(1).libid = 'standard'; S(1).lib_ncols = 9;
S(1).coupling_site = 'x_eq'; S(1).multi_fp = false;

S(2).name = 'brusselator';
S(2).psamp = @() sample_brussel(hill_rep,hill_deg);
S(2).odefn_base = @(t,s,p) [p.a-(p.b+1)*s(1)+s(1)^2*s(2);
                            p.b*s(1)-s(1)^2*s(2)];
S(2).ic_fun = @(p) [p.a; p.b/p.a];
S(2).libid = 'brusselator'; S(2).lib_ncols = 7;
S(2).coupling_site = 'x_eq'; S(2).multi_fp = false;

S(3).name = 'repressilator';
S(3).psamp = @() struct('alpha',4+6*rand(),'delta',1+2*rand(), ...
    'beta_rep',1+2*rand(),'gamma_rep',0.5+1.5*rand(), ...
    'hill_n',round(2+5*rand()),'hill_k0',1+3*rand(), ...
    'hill_km',0.4+1.6*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
S(3).odefn_base = @(t,s,p) [p.alpha*p.hill_rep(s(2),p.hill_k0,p.hill_n)-p.delta*p.hill_deg(s(1),p.hill_km);
                            p.beta_rep*s(1)-p.gamma_rep*s(2)];
S(3).ic_fun = @(p) [1.0; 0.5];
S(3).libid = 'standard'; S(3).lib_ncols = 9;
S(3).coupling_site = 'x_eq'; S(3).multi_fp = false;

S(4).name = 'van_der_pol';
S(4).psamp = @() struct('mu',0.3+0.7*rand(),'omega',0.8+0.4*rand(), ...
    'hill_k0',1+2*rand(),'hill_n',round(2+3*rand()), ...
    'hill_km',0.5+1.5*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
S(4).odefn_base = @(t,s,p) [p.mu*(1-s(2)^2)*s(1)-p.omega*s(2); s(1)];
S(4).ic_fun = @(p) [0.5; 0.0];
S(4).libid = 'standard'; S(4).lib_ncols = 9;
S(4).coupling_site = 'x_eq'; S(4).multi_fp = false;

S(5).name = 'fitzhugh_nagumo';
S(5).psamp = @() struct('a',0.6+0.3*rand(),'b',0.7+0.3*rand(), ...
    'tau',9+7*rand(),'I_ext',0.2+0.5*rand(), ...
    'hill_k0',1+2*rand(),'hill_n',round(2+3*rand()), ...
    'hill_km',0.5+1.5*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
S(5).odefn_base = @(t,s,p) [s(1)-s(1)^3/3-s(2)+p.I_ext;
                            (s(1)+p.a-p.b*s(2))/p.tau];
S(5).ic_fun = @(p) [0.1; 0.05];
S(5).libid = 'fhn'; S(5).lib_ncols = 8;
S(5).coupling_site = 'x_eq'; S(5).multi_fp = false;

S(6).name = 'rosenzweig_macarthur';
S(6).psamp = @() struct('r',0.8+0.6*rand(),'K',8+6*rand(), ...
    'a_rate',0.4+0.5*rand(),'h',0.3+0.5*rand(),'e',0.4+0.4*rand(), ...
    'm',0.2+0.3*rand(), ...
    'hill_k0',1+2*rand(),'hill_n',round(2+3*rand()), ...
    'hill_km',0.5+1.5*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
S(6).odefn_base = @(t,s,p) [ ...
    p.r*s(1)*(1-s(1)/p.K) - p.a_rate*s(1)*s(2)/(1+p.a_rate*p.h*s(1)); ...
    p.e*p.a_rate*s(1)*s(2)/(1+p.a_rate*p.h*s(1)) - p.m*s(2)];
S(6).ic_fun = @(p) [p.K*0.5; p.K*0.2];
S(6).libid = 'rma'; S(6).lib_ncols = 8;
S(6).coupling_site = 'x_eq'; S(6).multi_fp = false;

S(7).name = 'toggle_switch';
S(7).psamp = @() struct('alpha1',8+8*rand(),'alpha2',8+8*rand(), ...
    'beta',1.5+1.0*rand(),'gamma',1.5+1.0*rand(), ...
    'hill_k0',1+2*rand(),'hill_n',round(2+3*rand()), ...
    'hill_km',0.5+1.5*rand(), ...
    'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.05*rand());
S(7).odefn_base = @(t,s,p) [p.alpha1/(1+s(2)^p.beta) - s(1);
                            p.alpha2/(1+s(1)^p.gamma) - s(2)];
S(7).ic_fun = @(p) [];
S(7).libid = 'toggle'; S(7).lib_ncols = 6;
S(7).coupling_site = 'x_eq'; S(7).multi_fp = true;
end

function p = sample_brussel(hill_rep,hill_deg)
a = 1; b = 5;
for k=1:1000
    a=1+2*rand(); b=4+6*rand();
    if b>a^2+1; break; end
end
p=struct('a',a,'b',b,'hill_k0',1.5+2.5*rand(),'hill_n',round(2+3*rand()), ...
    'hill_km',0.5+1.5*rand(),'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
end
