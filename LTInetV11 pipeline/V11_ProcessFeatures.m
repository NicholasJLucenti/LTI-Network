function [ok, pack] = V11_ProcessFeatures(x_data, y_data, xa, ya, N, n_all, dt, ...
    sg_p, sg_f, p, sysdef, ridge_lambda, sparsity_thresh, n_stridge_iters)
% V11_ProcessFeatures — trimmed from V6_ProcessFeatures.m per the final
% V11 feature cut. Changes from V6:
%
%   REMOVED ENTIRELY (dead in V11, computed nothing downstream uses):
%     - v6_new_features (FNN + TransferEntropy + ResidualSpectrum, 12 dims)
%     - pack.Xi / pack.Xi_ternary (STRidge coefficients — superseded by
%       DEN's den_w_x/w_y upstream of V11, and V11 doesn't even keep DEN's
%       Stage-2 den_xi_w, so raw Xi has no remaining consumer)
%     - pack.col_names, pack.libid, pack.t_ode (metadata about the above,
%       equally unused downstream)
%
%   KEPT BUT STILL COMPUTED IN FULL, THEN SLICED:
%     - nullcline_features: g1-g7 sub-blocks share a PCA/eigendecomposition
%       (nc_shape_internal) that's cheaper to compute whole than to hand-
%       split — computed at full 49-dim, then indexed down to the 24
%       surviving dims with KEEP_NC immediately before saving.
%     - xcorr_features: same logic, computed at full 10-dim, sliced to 5
%       with KEEP_XC.
%   Both KEEP_* lists are given in the SAME 0-indexed order used in the
%   V11 Python training pipeline; +1 is applied here for MATLAB indexing.
%   This is a deliberate choice over hand-rewriting g1-g7: manually
%   decomposing a shared eigendecomposition into "only the surviving
%   terms" saves negligible compute and materially raises the risk of an
%   off-by-one mapping error versus the already-validated raw ordering.
%
%   fp_multi_features now comes from V11_MultiFixedPoint (3 dims, not 10).
%
% Xi itself (both equations) is STILL fitted internally — nullcline_features
% is literally the nullclines of the fitted vector field (dX,dY from
% Xi via V11_LibGrid), and xcorr_features needs resid_dx = dx_obs - dx_model
% (dx_model from Xi via V11_LibRow). Neither kept feature exists without it.

    ok = false; pack = struct();
    libid = sysdef.libid;

    % 0-indexed (Python/V11-trainer convention) -> MATLAB +1
    KEEP_NC = [1,2,3,4,5, 11,12, 13,14,15,16, 25,26,27,28, 29,30,31,32,33, 34, 46,47,49];
    KEEP_XC = [2,4,5,7,10];

    [Theta_full, ~] = V11_BuildLib(x_data, y_data, N, p, libid);
    n_terms = size(Theta_full,2);

    colscale = vecnorm(Theta_full,2,1); colscale(colscale==0)=1;
    ThetaN = Theta_full./colscale;
    dxdt = sgolayfilt(gradient(x_data,dt),sg_p,sg_f);
    dydt = sgolayfilt(gradient(y_data,dt),sg_p,sg_f);
    tSx=norm(dxdt,2); tSy=norm(dydt,2);
    if tSx<1e-8||tSy<1e-8
        pack.fail_reason = 'tSx_or_tSy_too_small'; return;
    end
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
    if fail||any(~isfinite(XiN(:)))
        pack.fail_reason = 'stridge_fail'; return;
    end

    Xi=zeros(n_terms,2);
    Xi(:,1)=(XiN(:,1)*tSx)./colscale'; Xi(:,2)=(XiN(:,2)*tSy)./colscale';
    Xi(abs(Xi)<0.005)=0;
    if all(Xi(:)==0)
        pack.fail_reason = 'all_zero_Xi'; return;
    end

    [nullcline_full, ok_nc, fp_extra] = v11_nullcline_features_internal( ...
        x_data, y_data, N, p, libid, Xi, tSx, tSy, sysdef.multi_fp);
    if ~ok_nc
        pack.fail_reason = 'nullcline_fail'; return;
    end

    dx_obs=sgolayfilt(gradient(xa,dt),sg_p,sg_f);
    dx_model=zeros(n_all,1);
    for k=2:n_all
        row = V11_LibRow(xa(k-1),ya(k-1),p,libid);
        if length(row)==size(Xi,1)
            dx_model(k)=row*Xi(:,1);
        end
    end
    resid_dx = dx_obs - dx_model;
    xcorr_full = v11_compute_xcorr_internal(resid_dx, xa, ya);
    if any(~isfinite(xcorr_full))
        pack.fail_reason = 'xcorr_fail'; return;
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

    pack.nullcline_features = nullcline_full(KEEP_NC);
    pack.xcorr_features     = xcorr_full(KEEP_XC);
    pack.fp_multi_features  = fp_extra;
    pack.topology            = topology;
    pack.x_data_all          = xa;
    pack.y_data_all          = ya;

    ok = true;
end


function [feat, ok, fp_extra] = v11_nullcline_features_internal(x_data, y_data, N, p, libid, Xi, tSx, tSy, is_multi_fp)
    ok=false; feat=zeros(49,1); fp_extra=zeros(3,1);
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
    g3=[fp_xn,fp_yn,fp_d,ca];

    [xnp,ynx]=extract_nc_internal(dX,xg,yg); [xny,ynp]=extract_nc_internal(dY,xg,yg);
    g1=nc_shape_internal((xnp-xc)/(ax+1e-8),(ynx-yc_)/(ay+1e-8));
    g2=nc_shape_internal((xny-xc)/(ax+1e-8),(ynp-yc_)/(ay+1e-8));

    Phi_o=V11_LibGrid(x_data,y_data,p,libid);
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
    [pvx,pix]=max(abs(xc)); plx=lx(pix); pwx=sum(abs(xc)>pvx/2);
    lax=(mean(abs(xc(lx>0)))-mean(abs(xc(lx<0))))/(mean(abs(xc(lx>0)))+mean(abs(xc(lx<0)))+1e-8);
    [yc,ly]=xcorr(rn,yn,ml,'normalized');
    [pvy,piy]=max(abs(yc)); ply=ly(piy); pwy=sum(abs(yc)>pvy/2);
    lay=(mean(abs(yc(ly>0)))-mean(abs(yc(ly<0))))/(mean(abs(yc(ly>0)))+mean(abs(yc(ly<0)))+1e-8);
    xcf=[plx;pvx;pwx;lax;xc(lx==0);ply;pvy;pwy;lay;yc(ly==0)];
end
