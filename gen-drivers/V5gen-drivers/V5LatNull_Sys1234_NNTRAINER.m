clear; close all; clc;
addpath(genpath('.'));
warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');

hill_rep = @(z,k,n)  k^n ./ (k^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);

dt           = 0.05;
N            = 300;
sg_p         = 3;
sg_f         = 11;
n_per_system = 300;
base_seed    = 9999;

ridge_lambda    = 0.05;
sparsity_thresh = 0.02;
n_stridge_iters = 10;
min_rel_amp         = 0.08;
min_transient_ratio = 0.3;

NULL_LAT  = 5;
NULL_COUP = 3;
DATA_ROOT = 'C:/Users/nickj/MATLAB Drive/Compiled Works/LTI Network';

systems = {'goodwin','brusselator','repressilator','van_der_pol'};

for sys_idx = 1:4
    sname    = systems{sys_idx};
    save_dir = sprintf('%s/V5Lat%d_Ch%d_Sys%d_NNdata', ...
        DATA_ROOT, NULL_LAT, NULL_COUP, sys_idx);
    if ~exist(save_dir,'dir'), mkdir(save_dir); end
    fprintf('\n=== NULL | %s ===\n', sname);

    switch sname
        case 'goodwin'
            psamp = @() struct('alpha',5+7*rand(),'d1',0.08+0.22*rand(), ...
                'ks',0.3+1.2*rand(),'Vmax',0.8+2.2*rand(), ...
                'hill_n',round(3+5*rand()),'hill_k0',1.5+2.5*rand(), ...
                'hill_km',0.4+1.6*rand(), ...
                'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
            odefn = @(t,s,p) [p.alpha*hill_rep(s(2),p.hill_k0,p.hill_n)-p.d1*s(1);
                               p.ks*s(1)-p.Vmax*hill_deg(s(2),p.hill_km)];
            ic0   = [1.5; 0.5];
            libid = 'standard';

        case 'brusselator'
            psamp = @() sample_brussel(hill_rep,hill_deg);
            odefn = @(t,s,p) [p.a-(p.b+1)*s(1)+s(1)^2*s(2);
                               p.b*s(1)-s(1)^2*s(2)];
            ic0   = [];
            libid = 'brusselator';

        case 'repressilator'
            psamp = @() struct('alpha',4+6*rand(),'delta',1+2*rand(), ...
                'beta_rep',1+2*rand(),'gamma_rep',0.5+1.5*rand(), ...
                'hill_n',round(2+5*rand()),'hill_k0',1+3*rand(), ...
                'hill_km',0.4+1.6*rand(), ...
                'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
            odefn = @(t,s,p) [p.alpha*hill_rep(s(2),p.hill_k0,p.hill_n)-p.delta*hill_deg(s(1),p.hill_km);
                               p.beta_rep*s(1)-p.gamma_rep*s(2)];
            ic0   = [1.0; 0.5];
            libid = 'standard';

        case 'van_der_pol'
            psamp = @() struct('mu',0.3+0.7*rand(),'omega',0.8+0.4*rand(), ...
                'hill_k0',1+2*rand(),'hill_n',round(2+3*rand()), ...
                'hill_km',0.5+1.5*rand(), ...
                'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
            odefn = @(t,s,p) [p.mu*(1-s(2)^2)*s(1)-p.omega*s(2); s(1)];
            ic0   = [0.5; 0.0];
            libid = 'standard';
    end

    rng(base_seed + sys_idx*1000);
    existing = length(dir(fullfile(save_dir,'*.mat')));

    if strcmp(sname,'van_der_pol')
        opts_ode = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt);
    else
        opts_ode = odeset('RelTol',1e-5,'AbsTol',1e-7,'MaxStep',dt);
    end

    t_end  = (N+200)*dt;
    t_span = 0:dt:t_end;
    saved=0; skipped=0; attempt=0;

    while saved < n_per_system
        attempt=attempt+1;
        p = psamp();
        if isempty(ic0); ic = [p.a; p.b/p.a]; else; ic = ic0; end

        try
            [t_ode,S] = ode15s(@(t,s) odefn(t,s,p), t_span, ic, opts_ode);
        catch; skipped=skipped+1; continue; end

        if any(~isfinite(S(:))) || length(t_ode) < N+201
            skipped=skipped+1; continue; end

        t_uni = (0:dt:t_end)';
        xa = interp1(t_ode,S(:,1),t_uni,'linear');
        ya = interp1(t_ode,S(:,2),t_uni,'linear');
        t_ode = t_uni; n_all = length(t_ode);

        if any(~isfinite(xa))||any(~isfinite(ya))||max(abs(xa))>1e4||max(abs(ya))>1e4
            skipped=skipped+1; continue; end

        tail=round(0.4*N);
        if var(xa(N-tail+1:N))<1e-6||std(diff(xa(1:N)))<1e-4
            skipped=skipped+1; continue; end

        xa = xa + p.noise*std(xa)*randn(n_all,1);
        ya = ya + p.noise*std(ya)*randn(n_all,1);
        x_data=xa(1:N); y_data=ya(1:N);

        if (max(x_data)-min(x_data))/(mean(abs(x_data))+1e-8) < min_rel_amp
            skipped=skipped+1; continue; end
        if var(x_data(1:round(N/2))) < min_transient_ratio*var(x_data(round(N/2):N))
            skipped=skipped+1; continue; end

        [Theta_full, col_names] = build_lib(x_data,y_data,N,p,libid);
        n_terms = size(Theta_full,2);

        colscale = vecnorm(Theta_full,2,1); colscale(colscale==0)=1;
        ThetaN   = Theta_full./colscale;
        dxdt = sgolayfilt(gradient(x_data,dt),sg_p,sg_f);
        dydt = sgolayfilt(gradient(y_data,dt),sg_p,sg_f);
        tSx=norm(dxdt,2); tSy=norm(dydt,2);
        if tSx<1e-8||tSy<1e-8; skipped=skipped+1; continue; end
        dSdtN=[dxdt/tSx, dydt/tSy];

        XiN=zeros(n_terms,2); fail=false;
        for eq=1:2
            active=true(n_terms,1);
            for it=1:n_stridge_iters
                Tha=ThetaN(:,active);
                A=Tha'*Tha+ridge_lambda*eye(sum(active)); b=Tha'*dSdtN(:,eq);
                try; c=A\b; catch; fail=true; break; end
                cf=zeros(n_terms,1); cf(active)=c;
                na=abs(cf)>=sparsity_thresh;
                if isequal(na,active); active=na; break; end
                active=na; if ~any(active); active=true(n_terms,1); break; end
            end
            if fail; break; end
            Tha=ThetaN(:,active); A=Tha'*Tha+ridge_lambda*eye(sum(active)); b=Tha'*dSdtN(:,eq);
            try; c=A\b; catch; fail=true; break; end
            cf=zeros(n_terms,1); cf(active)=c; XiN(:,eq)=cf;
        end
        if fail||any(~isfinite(XiN(:))); skipped=skipped+1; continue; end

        Xi=zeros(n_terms,2);
        Xi(:,1)=(XiN(:,1)*tSx)./colscale'; Xi(:,2)=(XiN(:,2)*tSy)./colscale';
        Xi(abs(Xi)<0.005)=0;
        if all(Xi(:)==0); skipped=skipped+1; continue; end

        [nullcline_features, ok_nc] = compute_nullcline_features(x_data,y_data,N,p,libid,Xi,tSx,tSy);
        if ~ok_nc; skipped=skipped+1; continue; end

        dx_obs=sgolayfilt(gradient(xa,dt),sg_p,sg_f);
        dx_model=zeros(n_all,1);
        for k=2:n_all
            row=lib_row(xa(k-1),ya(k-1),p,libid);
            dx_model(k)=row*Xi(:,1);
        end
        xcorr_features=compute_xcorr(dx_obs-dx_model,xa,ya);
        if any(~isfinite(xcorr_features)); skipped=skipped+1; continue; end

        Xi_ternary=sign(Xi).*(abs(Xi)>0.005);
        if size(Xi_ternary,1)<9
            Xi_ternary=[Xi_ternary;zeros(9-size(Xi_ternary,1),2)];
        end

        half=round(n_all/2); x2=xa(half:end); y2=ya(half:end);
        sust=(var(x2)>1e-4)&&(var(y2)>1e-4);
        lpe=norm([x2(end)-x2(1),y2(end)-y2(1)])/(norm([x2(1),y2(1)])+1e-8);
        isc=lpe<0.25;
        seg=round(length(x2)/3);
        ar=std(x2(end-seg+1:end))/max(std(x2(1:seg)),1e-8);
        if ar>1.25; at='growing'; elseif ar<0.75; at='decaying'; else; at='stable'; end
        if sust&&isc&&strcmp(at,'stable'); topology='LIMIT CYCLE';
        elseif sust&&strcmp(at,'decaying'); topology='DAMPED OSCILLATION';
        elseif ~sust; topology='STEADY STATE';
        else; topology='UNDETERMINED'; end

        structure_label=NULL_LAT; coupling_label=NULL_COUP;
        x_data_all=xa; y_data_all=ya;

        fname=fullfile(save_dir,sprintf('example_%04d.mat',existing+saved+1));
        save(fname,'nullcline_features','xcorr_features','Xi_ternary', ...
            'topology','structure_label','coupling_label', ...
            'x_data_all','y_data_all','t_ode','Xi','col_names');

        saved=saved+1;
        if mod(saved,50)==0
            fprintf('  %d/%d (skip=%d)\n',saved,n_per_system,skipped);
        end
    end
    fprintf('[null|%s] saved=%d skipped=%d attempts=%d\n',sname,saved,skipped,attempt);
end
fprintf('\nNull generation done. Run V5_annex_full.m on null folders next.\n');

%% ── HELPERS ──────────────────────────────────────────────────────────────
function p=sample_brussel(hill_rep,hill_deg)
    max_tries = 1000; a = 1; b = 5;
    for k=1:max_tries
        a=1+2*rand(); b=4+6*rand();
        if b>a^2+1; break; end
    end
    p=struct('a',a,'b',b,'hill_k0',1.5+2.5*rand(),'hill_n',round(2+3*rand()), ...
        'hill_km',0.5+1.5*rand(),'hill_rep',hill_rep,'hill_deg',hill_deg,'noise',0.01+0.07*rand());
end

function [Theta, col_names]=build_lib(x,y,N_,p,libid)
    if strcmp(libid,'brusselator')
        Theta=[ones(N_,1),x,x.^2,y,x.^2.*y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
        col_names={'1','x','x^2','y','x^2y','HillRep','HillDeg'};
    else
        Theta=[ones(N_,1),x,x.^2,x.^3,y,y.^2,y.^3,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
        col_names={'1','x','x^2','x^3','y','y^2','y^3','HillRep','HillDeg'};
    end
end

function row=lib_row(x,y,p,libid)
    if strcmp(libid,'brusselator')
        row=[1,x,x^2,y,x^2*y,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
    else
        row=[1,x,x^2,x^3,y,y^2,y^3,p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
    end
end

function Phi=lib_grid(x,y,p,libid)
    x=x(:);y=y(:);n=length(x);
    hr=p.hill_rep(y,p.hill_k0,p.hill_n); hd=p.hill_deg(y,p.hill_km);
    if strcmp(libid,'brusselator'); Phi=[ones(n,1),x,x.^2,y,x.^2.*y,hr,hd];
    else; Phi=[ones(n,1),x,x.^2,x.^3,y,y.^2,y.^3,hr,hd]; end
end

function [feat,ok]=compute_nullcline_features(x_data,y_data,N,p,libid,Xi,tSx,tSy)
    ok=false; feat=zeros(49,1);
    ng=50; xg=linspace(min(x_data),max(x_data),ng); yg=linspace(min(y_data),max(y_data),ng);
    [XG,YG]=meshgrid(xg,yg);
    Phi=lib_grid(XG(:),YG(:),p,libid);
    dX=reshape(Phi*Xi(:,1),ng,ng); dY=reshape(Phi*Xi(:,2),ng,ng);
    xc=mean(x_data); yc_=mean(y_data);
    ax=max(x_data)-min(x_data); ay=max(y_data)-min(y_data); an=sqrt(ax^2+ay^2);

    Fmag=dX.^2+dY.^2;
    [~,mi]=min(Fmag(:));
    [rf,cf]=ind2sub([ng,ng],mi);
    xfp=xg(cf); yfp=yg(rf);
    fp_xn=(xfp-xc)/(ax+1e-8); fp_yn=(yfp-yc_)/(ay+1e-8);
    fp_d=sqrt((xfp-xc)^2+(yfp-yc_)^2)/(an+1e-8);
    gx=(xg(2)-xg(1)); gy=(yg(2)-yg(1));
    nx1=[-(dX(min(rf+1,ng),cf)-dX(max(rf-1,1),cf))/(2*gy+1e-8), ...
          (dX(rf,min(cf+1,ng))-dX(rf,max(cf-1,1)))/(2*gx+1e-8)];
    nx2=[-(dY(min(rf+1,ng),cf)-dY(max(rf-1,1),cf))/(2*gy+1e-8), ...
          (dY(rf,min(cf+1,ng))-dY(rf,max(cf-1,1)))/(2*gx+1e-8)];
    nx1=nx1/(norm(nx1)+1e-8); nx2=nx2/(norm(nx2)+1e-8);
    ca=acos(min(abs(dot(nx1,nx2)),1));
    g3=[fp_xn,fp_yn,fp_d,ca];

    [xnp,ynx]=extract_nc(dX,xg,yg); [xny,ynp]=extract_nc(dY,xg,yg);
    g1=nc_shape((xnp-xc)/(ax+1e-8),(ynx-yc_)/(ay+1e-8));
    g2=nc_shape((xny-xc)/(ax+1e-8),(ynp-yc_)/(ay+1e-8));

    Phi_o=lib_grid(x_data,y_data,p,libid);
    dxo=Phi_o*Xi(:,1); dyo=Phi_o*Xi(:,2);
    [~,mdi]=max(abs(dxo));
    g4=[mean(dxo)/(tSx+1e-8),std(dxo)/(tSx+1e-8), ...
        mean(dyo)/(tSy+1e-8),std(dyo)/(tSy+1e-8),mdi/N];

    sdx=sign(dxo);sdy=sign(dyo);
    cx=sum(abs(diff(sdx))>0); cy=sum(abs(diff(sdy))>0);
    nc_=max(sum(abs(diff(sign(x_data-mean(x_data))))>0)/2,1);
    g6=[cx/nc_,(sum(diff(sdx)<0)-sum(diff(sdx)>0))/(cx+1e-8), ...
        cy/nc_,(sum(diff(sdy)<0)-sum(diff(sdy)>0))/(cy+1e-8)];

    gg1=g1(:);gg2=g2(:);
    cr=@(a,b) max(min(a/(b+1e-6*(abs(b)<1e-6)),5),-5);
    g7=[gg1(4)-gg2(4);gg1(5:8)-gg2(5:8);gg1(9)-gg2(9);gg1(10)-gg2(10); ...
        gg1(11)-gg2(11);gg1(12)-gg2(12); ...
        cr(gg1(4),gg2(4));cr(gg1(11),gg2(11));cr(gg1(12),gg2(12))];

    feat=max(min([g1(:)',g2(:)',g3(:)',g4(:)',g6(:)',g7(:)']',50),-50);
    ok=all(isfinite(feat)) && numel(feat)==49;
end

function [xp,yp]=extract_nc(F,xg,yg)
    Sl=F(:,1:end-1);Sr=F(:,2:end);hm=Sl.*Sr<0; [r,c]=find(hm);
    if ~isempty(r); t=Sl(hm)./(Sl(hm)-Sr(hm)); xh=xg(c)'+t.*(xg(c+1)'-xg(c)'); yh=yg(r)'; else; xh=[];yh=[]; end
    Su=F(1:end-1,:);Sd=F(2:end,:);vm=Su.*Sd<0; [r,c]=find(vm);
    if ~isempty(r); t=Su(vm)./(Su(vm)-Sd(vm)); xv=xg(c)'; yv=yg(r)'+t.*(yg(r+1)'-yg(r)'); else; xv=[];yv=[]; end
    xp=[xh;xv]; yp=[yh;yv];
    if isempty(xp); xp=0; yp=0; end
end

function f=nc_shape(xp,yp)
    if length(xp)<6; f=zeros(12,1); return; end
    cx=mean(xp);cy=mean(yp); pts=[xp(:)-cx,yp(:)-cy];
    C=(pts'*pts)/max(size(pts,1)-1,1); [V,D]=eig(C); ev=diag(D);
    [evs,od]=sort(ev,'descend'); V=V(:,od);
    pa=atan2(V(2,1),V(1,1)); asp=min(sqrt(evs(1))/(sqrt(evs(2))+1e-8),20);
    pal=pts*V(:,1); pac=pts*V(:,2);
    if std(pal)<1e-8; f=zeros(12,1); return; end
    qb=quantile(pal,[0 .25 .5 .75 1]); cf_=zeros(4,1);
    for q=1:4; m=pal>=qb(q)&pal<qb(q+1); if sum(m)>1; cf_(q)=std(pac(m)); end; end
    pm=pal>0; nm=pal<=0;
    if sum(pm)<2||sum(nm)<2; al=0; ac=0;
    else; ps=std(pal(pm));ns=std(pal(nm)); al=max(min((ps+1e-8)/(ns+1e-8)-1,5),-5); ac=mean(pac(pm))-mean(pac(nm)); end
    if length(xp)>2; arc=min(sum(sqrt(diff(xp(:)).^2+diff(yp(:)).^2)),20); else; arc=0; end
    sr=min(std(pac)/(std(pal)+1e-8),5);
    f=[cx;cy;pa;asp;cf_;al;ac;arc;sr];
end

function xcf=compute_xcorr(rdx,xa,ya)
    ml=50;
    xn=(xa-mean(xa))/(std(xa)+1e-8); yn=(ya-mean(ya))/(std(ya)+1e-8);
    rn=(rdx-mean(rdx))/(std(rdx)+1e-8);
    n=min([length(xn),length(yn),length(rn)]);
    xn=xn(1:n);yn=yn(1:n);rn=rn(1:n);
    [xc,lx]=xcorr(rn,xn,ml,'normalized');
    [pvx,pix]=max(abs(xc)); plx=lx(pix); pwx=sum(abs(xc)>pvx/2);
    lax=(mean(abs(xc(lx>0)))-mean(abs(xc(lx<0))))/(mean(abs(xc(lx>0)))+mean(abs(xc(lx<0)))+1e-8);
    [yc,ly]=xcorr(rn,yn,ml,'normalized');
    [pvy,piy]=max(abs(yc)); ply=ly(piy); pwy=sum(abs(yc)>pvy/2);
    lay=(mean(abs(yc(ly>0)))-mean(abs(yc(ly<0))))/(mean(abs(yc(ly>0)))+mean(abs(yc(ly<0)))+1e-8);
    xcf=[plx;pvx;pwx;lax;xc(lx==0);ply;pvy;pwy;lay;yc(ly==0)];
end