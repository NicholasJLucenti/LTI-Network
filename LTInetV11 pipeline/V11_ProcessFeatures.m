function [ok, pack] = V11_ProcessFeatures(x_data, y_data, xa, ya, N, n_all, dt, ...
    sg_p, sg_f, p, sysdef, ridge_lambda, sparsity_thresh, n_stridge_iters)

    ok = false; pack = struct();
    libid = sysdef.libid;

    [Theta_full, ~] = V11_BuildLib(x_data, y_data, N, p, libid);
    n_terms = size(Theta_full,2);

    colscale = vecnorm(Theta_full,2,1); colscale(colscale==0)=1;
    ThetaN = Theta_full./colscale;
    dxdt = sgolayfilt(gradient(x_data,dt),sg_p,sg_f);
    dydt = sgolayfilt(gradient(y_data,dt),sg_p,sg_f);
    tSx=norm(dxdt,2); tSy=norm(dydt,2);
    if tSx<1e-8||tSy<1e-8; return; end
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
    if fail||any(~isfinite(XiN(:))); return; end

    Xi=zeros(n_terms,2);
    Xi(:,1)=(XiN(:,1)*tSx)./colscale'; Xi(:,2)=(XiN(:,2)*tSy)./colscale';
    Xi(abs(Xi)<0.005)=0;
    if all(Xi(:)==0); return; end

    [nullcline_features, ok_nc, fp_extra] = v11_nullcline_features_internal( ...
        x_data, y_data, N, p, libid, Xi, tSx, tSy, sysdef.multi_fp);
    if ~ok_nc; return; end

    dx_obs=sgolayfilt(gradient(xa,dt),sg_p,sg_f);
    dx_model=zeros(n_all,1);
    for k=2:n_all
        row = V11_LibRow(xa(k-1),ya(k-1),p,libid);
        if length(row)==size(Xi,1)
            dx_model(k)=row*Xi(:,1);
        end
    end
    resid_dx = dx_obs - dx_model;
    xcorr_features = v11_compute_xcorr_internal(resid_dx, xa, ya);
    if any(~isfinite(xcorr_features)); return; end

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

    pack.nullcline_features = nullcline_features;
    pack.xcorr_features     = xcorr_features;
    pack.fp_multi_features  = fp_extra;
    pack.topology            = topology;
    pack.x_data_all          = xa;
    pack.y_data_all          = ya;

    ok = true;
end


function [feat, ok, fp_extra] = v11_nullcline_features_internal(x_data, y_data, N, p, libid, Xi, tSx, tSy, is_multi_fp)
    ok=false; fp_extra=zeros(3,1);
    ng=50; xg=linspace(min(x_data),max(x_data),ng); yg=linspace(min(y_data),max(y_data),ng);
    [XG,YG]=meshgrid(xg,yg);
    Phi=V11_LibGrid(XG(:),YG(:),p,libid);
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

    [xnp,ynx]=extract_nc_internal(dX,xg,yg); [xny,ynp]=extract_nc_internal(dY,xg,yg);
    f1=nc_shape_internal((xnp-xc)/(ax+1e-8),(ynx-yc_)/(ay+1e-8));
    f2=nc_shape_internal((xny-xc)/(ax+1e-8),(ynp-yc_)/(ay+1e-8));
    cx1=f1(1); cy1=f1(2); pa1=f1(3); asp1=f1(4); cf1=f1(5); arc1=f1(11); sr1=f1(12);
    cx2=f2(1); cy2=f2(2); pa2=f2(3); asp2=f2(4); sr2=f2(12);

    Phi_o=V11_LibGrid(x_data,y_data,p,libid);
    dxo=Phi_o*Xi(:,1); dyo=Phi_o*Xi(:,2);
    [~,mdi]=max(abs(dxo));
    mean_dxo_n=mean(dxo)/(tSx+1e-8); std_dxo_n=std(dxo)/(tSx+1e-8);
    mean_dyo_n=mean(dyo)/(tSy+1e-8); std_dyo_n=std(dyo)/(tSy+1e-8);
    mdi_n=mdi/N;

    sdx=sign(dxo);
    x_cross_count=sum(abs(diff(sdx))>0);
    x_selfcross_count=max(sum(abs(diff(sign(x_data-mean(x_data))))>0)/2,1);
    x_null_density=x_cross_count/x_selfcross_count;

    cr=@(a,b) max(min(a/(b+1e-6*(abs(b)<1e-6)),5),-5);
    sr_diff=sr1-sr2; cr_asp=cr(asp1,asp2); cr_sr=cr(sr1,sr2);

    feat=max(min([cx1;cy1;pa1;asp1;cf1;arc1;sr1; cx2;cy2;pa2;asp2; ...
                  fp_xn;fp_yn;fp_d;ca; mean_dxo_n;std_dxo_n;mean_dyo_n;std_dyo_n;mdi_n; ...
                  x_null_density; sr_diff;cr_asp;cr_sr],50),-50);
    ok=all(isfinite(feat)) && numel(feat)==24;

    if is_multi_fp
        fp_extra = V11_MultiFixedPoint(dX, dY, xg, yg);
    end
end

function [xp,yp]=extract_nc_internal(F,xg,yg)
    Sl=F(:,1:end-1);Sr=F(:,2:end);hm=Sl.*Sr<0; [r,c]=find(hm);
    if ~isempty(r); t=Sl(hm)./(Sl(hm)-Sr(hm)); xh=xg(c)'+t.*(xg(c+1)'-xg(c)'); yh=yg(r)'; else; xh=[];yh=[]; end
    Su=F(1:end-1,:);Sd=F(2:end,:);vm=Su.*Sd<0; [r,c]=find(vm);
    if ~isempty(r); t=Su(vm)./(Su(vm)-Sd(vm)); xv=xg(c)'; yv=yg(r)'+t.*(yg(r+1)'-yg(r)'); else; xv=[];yv=[]; end
    xp=[xh;xv]; yp=[yh;yv];
    if isempty(xp); xp=0; yp=0; end
end

function f=nc_shape_internal(xp,yp)
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

function xcf=v11_compute_xcorr_internal(rdx,xa,ya)
    ml=50;
    xn=(xa-mean(xa))/(std(xa)+1e-8); yn=(ya-mean(ya))/(std(ya)+1e-8);
    rn=(rdx-mean(rdx))/(std(rdx)+1e-8);
    n=min([length(xn),length(yn),length(rn)]);
    xn=xn(1:n);yn=yn(1:n);rn=rn(1:n);
    [xc,lx]=xcorr(rn,xn,ml,'normalized');
    pvx=max(abs(xc));
    lax=(mean(abs(xc(lx>0)))-mean(abs(xc(lx<0))))/(mean(abs(xc(lx>0)))+mean(abs(xc(lx<0)))+1e-8);
    xc0=xc(lx==0);
    [yc,ly]=xcorr(rn,yn,ml,'normalized');
    pvy=max(abs(yc));
    yc0=yc(ly==0);
    xcf=[pvx;lax;xc0;pvy;yc0];
end