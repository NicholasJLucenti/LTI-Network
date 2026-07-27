clear; close all;
addpath(genpath('.'));

warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');

% ── Shared handles ────────────────────────────────────────────────────────
hill_rep = @(z,k,n)  k^n ./ (k^n + z.^n);
hill_deg = @(y,km)   y   ./ (km  + y);
hill_act = @(x,k,n)  x.^n ./ (k^n + x.^n);

dt           = 0.05;
N            = 300;
eta          = 0.6;
sg_p         = 3;
sg_f         = 11;
base_seed    = 888;
n_per_config = 50;

% ── V4 design ─────────────────────────────────────────────────────────────
% Latent equations (what I does internally):
%   Lat0: dI/dt = kp*x - kd*I                         linear mRNA readout
%   Lat1: dI/dt = kp*y - (kd + km*x)*I                miRNA stoichiometric titration
%   Lat2: dI/dt = kp*x*H_rep(x,k0,n) - kd*I           incoherent feedforward
%   Lat3: dI/dt = kp*x*(1-I/Imax) - kcat*I/(Km+I)     Goldbeter-Koshland phospho cycle
%   Lat4: dI/dt = kp*H_rep(y,k0,n) - kd*I             Hill repression by protein y
%
% Coupling channels (how I enters the x-equation):
%   Ch0: -alpha_c * H_rep(I, k0, n)     cooperative transcriptional repression
%   Ch1: -beta_c  * I                   linear additive inhibition
%   Ch2: -gamma_c * I * x               multiplicative mRNA degradation
%
% x-equation per system per channel:
%   Goodwin/Repressilator Ch0: alpha*H_rep(I,k0,n) - d1*x            [I is repressor]
%   Goodwin/Repressilator Ch1: alpha*H_rep(y,k0,n) - d1*x - beta_c*I  [y represses, I adds]
%   Goodwin/Repressilator Ch2: alpha*H_rep(y,k0,n) - d1*x - gamma_c*I*x
%   Brusselator/VdP all ch:   [base dynamics] + coupling_term(I,x,ch)
%
% Each saved file carries:
%   structure_label  = latent class (0-4)
%   coupling_label   = coupling channel (0-2)

active_latents    = [0, 1, 2, 3, 4];
coupling_channels = [0, 1, 2];
systems           = {'goodwin', 'brusselator', 'repressilator', 'van_der_pol'};

% Excluded combinations — [latent, channel, sys_idx]
excluded = [4,0,1; 3,0,1; 0,1,3; 0,2,3; 1,1,3; 2,2,3; 3,1,3; 3,2,3];

for sys_idx = 1:3
    system_name = systems{sys_idx};

    for lc_idx = 1:length(active_latents)
        latent_class = active_latents(lc_idx);

        for ch_idx = 1:length(coupling_channels)
            coupling_ch = coupling_channels(ch_idx);

            if any(excluded(:,1)==latent_class & ...
                   excluded(:,2)==coupling_ch  & ...
                   excluded(:,3)==sys_idx)
                fprintf('Skipping Lat%d Ch%d Sys%d (%s)\n', ...
                    latent_class, coupling_ch, sys_idx, system_name);
                continue
            end

            save_dir = sprintf( ...
                'C:/Users/nickj/MATLAB Drive/LTI Network/V4Lat%d_Ch%d_Sys%d_NNdata', ...
                latent_class, coupling_ch, sys_idx);

            if ~exist(save_dir,'dir'), mkdir(save_dir); end

            fprintf('\n=== V4 | %s | Lat%d | Ch%d ===\n', ...
                system_name, latent_class, coupling_ch);

            switch system_name
                case 'goodwin'
                    param_sampler = @() sample_goodwin(hill_rep,hill_deg, ...
                        hill_act,coupling_ch);
                    ode_fun = @(t,s,p) goodwin_ode(t,s,p,latent_class,coupling_ch);
                    ic_fun  = @(p) [1.5; 0.5; 0.5];
                    lib_fun = @(x,y,N_,p) standard_lib(x,y,N_,p);

                case 'brusselator'
                    param_sampler = @() sample_brusselator(hill_rep,hill_deg, ...
                        hill_act,coupling_ch);
                    ode_fun = @(t,s,p) brusselator_ode(t,s,p,latent_class,coupling_ch);
                    ic_fun  = @(p) [p.a; p.b/p.a; 0.5];
                    lib_fun = @(x,y,N_,p) brusselator_lib(x,y,N_,p);

                case 'repressilator'
                    param_sampler = @() sample_repressilator(hill_rep,hill_deg, ...
                        hill_act,coupling_ch);
                    ode_fun = @(t,s,p) repressilator_ode(t,s,p,latent_class,coupling_ch);
                    ic_fun  = @(p) [1.0; 0.5; 0.5];
                    lib_fun = @(x,y,N_,p) standard_lib(x,y,N_,p);

                case 'van_der_pol'
                    param_sampler = @() sample_vanderpol(hill_rep,hill_deg, ...
                        hill_act,coupling_ch);
                    ode_fun = @(t,s,p) vanderpol_ode(t,s,p,latent_class,coupling_ch);
                    ic_fun  = @(p) [0.5; 0.0; 0.5];
                    lib_fun = @(x,y,N_,p) standard_lib(x,y,N_,p);
            end

            run_generation(system_name, ode_fun, ic_fun, param_sampler, ...
                lib_fun, latent_class, coupling_ch, n_per_config, save_dir, ...
                base_seed + sys_idx*1000 + latent_class*10 + coupling_ch);
        end
    end
end

fprintf('\nAll V4 Lat0-4 × Ch0-2 × Sys1-4 generation complete.\n');


%% ── GENERATION LOOP ──────────────────────────────────────────────────────
function run_generation(system_name, ode_fun, ic_fun, param_sampler, ...
        lib_fun, latent_class, coupling_ch, n_examples, save_dir, rng_seed)

existing = length(dir(fullfile(save_dir,'*.mat')));
rng(rng_seed + existing*13);

dt   = 0.05; N = 300; eta = 0.6; sg_p = 3; sg_f = 11;
saved = 0; skipped = 0; attempt = 0;

if strcmp(system_name,'van_der_pol')
    opts_ode = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt);
else
    opts_ode = odeset('RelTol',1e-5,'AbsTol',1e-7,'MaxStep',dt);
end

opts_ls = optimoptions('lsqlin','Display','off');
t_end   = (N+200)*dt;
t_span  = 0:dt:t_end;

while saved < n_examples
    attempt = attempt+1;
    p = param_sampler();

    try
        [t_ode,S] = ode15s(@(t,s) ode_fun(t,s,p), t_span, ic_fun(p), opts_ode);
    catch; skipped=skipped+1; continue; end

    if any(~isfinite(S(:))) || length(t_ode) < N+201
        skipped=skipped+1; continue; end

    t_uniform  = (0:dt:t_end)';
    x_data_all = interp1(t_ode,S(:,1),t_uniform,'linear');
    y_data_all = interp1(t_ode,S(:,2),t_uniform,'linear');
    z_data_all = interp1(t_ode,S(:,3),t_uniform,'linear');
    t_ode      = t_uniform;
    n_all      = length(t_ode);

    if any(~isfinite(x_data_all))||any(~isfinite(y_data_all))
        skipped=skipped+1; continue; end
    if max(abs(x_data_all))>1e4||max(abs(y_data_all))>1e4||max(abs(z_data_all))>1e4
        skipped=skipped+1; continue; end

    tail = round(0.4*n_all);
    if var(z_data_all(end-tail:end))<1e-6||var(x_data_all(end-tail:end))<1e-6
        skipped=skipped+1; continue; end
    if std(diff(x_data_all(1:N)))<1e-4
        skipped=skipped+1; continue; end

    noise      = p.noise;
    x_data_all = x_data_all + noise*std(x_data_all)*randn(n_all,1);
    y_data_all = y_data_all + noise*std(y_data_all)*randn(n_all,1);

    x_data = x_data_all(1:N);
    y_data = y_data_all(1:N);

    lib        = lib_fun(x_data,y_data,N,p);
    Theta_full = lib.Theta_full;
    Theta_poly = lib.Theta_poly;
    col_names  = lib.col_names;
    n_poly     = size(Theta_poly,2);
    n_terms    = size(Theta_full,2);

    colscale_full = vecnorm(Theta_full,2,1);
    colscale_full(colscale_full==0)=1;
    ThetaN_full   = Theta_full./colscale_full;
    ThetaN_poly   = Theta_poly./colscale_full(1:n_poly);

    dxdt = sgolayfilt(gradient(x_data,dt),sg_p,sg_f);
    dydt = sgolayfilt(gradient(y_data,dt),sg_p,sg_f);
    targetScaleX = norm(dxdt,2); targetScaleY = norm(dydt,2);
    dSdtN = [dxdt/targetScaleX, dydt/targetScaleY];

    XiN=zeros(n_terms,2); XiN(1:n_poly,:)=ThetaN_poly\dSdtN;
    XiN_var=zeros(n_terms,2); XiN_smooth=XiN;
    wx=ones(n_poly,1); wy=ones(n_poly,1);

    for iter=1:25
        for ind=1:2
            switch ind
                case 1
                    current_w=wx;
                    xi_s=XiN_smooth([2 3 4],1); vxi=XiN_var([2 3 4],1);
                    cx_=(xi_s(1)*targetScaleX)/colscale_full(2);
                    cx2=(xi_s(2)*targetScaleX)/colscale_full(3);
                    cx3=(xi_s(3)*targetScaleX)/colscale_full(4);
                    avg_d=mean(abs(cx_*x_data+cx2*x_data.^2+cx3*x_data.^3));
                    hb=mean(p.hill_rep(y_data,p.hill_k0,p.hill_n));
                    fp_=avg_d/max(hb,1e-3);
                    h_val=(fp_*colscale_full(end-1))/targetScaleX;
                    sv=(targetScaleX./colscale_full(2:4)').*(colscale_full(end-1)/targetScaleX);
                    h_var=sum(vxi.*sv.^2);
                    Hill_red=ThetaN_full(:,end-1)*h_val;
                    dSdtN_ej=dSdtN(:,1)-Hill_red;
                    lb=p.lb{1}; ub=p.ub{1};
                case 2
                    current_w=wy;
                    xi_s=XiN_smooth([2 3 4],2); vxi=XiN_var([2 3 4],2);
                    cx_=(xi_s(1)*targetScaleY)/colscale_full(2);
                    cx2=(xi_s(2)*targetScaleY)/colscale_full(3);
                    cx3=(xi_s(3)*targetScaleY)/colscale_full(4);
                    avg_d=mean(abs(cx_*x_data+cx2*x_data.^2+cx3*x_data.^3));
                    hb=mean(p.hill_deg(y_data,p.hill_km));
                    fp_=avg_d/max(hb,1e-3);
                    h_val=-(fp_*colscale_full(end))/targetScaleY;
                    sv=(targetScaleY./colscale_full(2:4)').*(colscale_full(end)/targetScaleY);
                    h_var=sum(vxi.*sv.^2);
                    Hill_red=ThetaN_full(:,end)*h_val;
                    dSdtN_ej=dSdtN(:,2)-Hill_red;
                    lb=p.lb{2}; ub=p.ub{2};
            end

            ThetaN_w=ThetaN_poly./current_w'; biginds=current_w<10;
            XiN_poly_vec=XiN(1:n_poly,ind);
            if any(biginds)
                active=find(biginds);
                XiN_poly_vec(active)=lsqlin(ThetaN_w(:,active),dSdtN_ej,...
                    [],[],[],[],lb(active),ub(active),[],opts_ls);
                XiN_poly_vec(~biginds)=0;
            end
            XiN(1:n_poly,ind)=XiN_poly_vec;
            hill_idx=n_poly+ind;
            XiN(hill_idx,ind)=(1-eta)*XiN(hill_idx,ind)+eta*h_val;
            XiN_var(hill_idx,ind)=h_var;
            eps_s=1e-3; tau0=0.05;
            new_w=1./(tau0^2*(XiN_poly_vec.^2+eps_s));
            if ind==1; wx=0.7*wx+0.3*new_w; current_w=wx;
            else;      wy=0.7*wy+0.3*new_w; current_w=wy; end
            resid=dSdtN_ej-ThetaN_poly*XiN_poly_vec;
            resid_var=max(var(resid),1e-12);
            H_mat=(ThetaN_poly'*ThetaN_poly)/resid_var+diag(current_w);
            XiN_var(1:n_poly,ind)=diag(inv(H_mat));
        end
        XiN_smooth=0.7*XiN_smooth+0.3*XiN;
    end

    Xi=zeros(n_terms,2);
    Xi(:,1)=(XiN(:,1)*targetScaleX)./colscale_full';
    Xi(:,2)=(XiN(:,2)*targetScaleY)./colscale_full';
    Xi(abs(Xi)<0.005)=0;

    %% ── NULLCLINE FEATURES ───────────────────────────────────────────
    x_min=min(x_data); x_max=max(x_data);
    y_min=min(y_data); y_max=max(y_data);
    n_grid=50;
    xg=linspace(x_min,x_max,n_grid); yg=linspace(y_min,y_max,n_grid);
    [XG,YG]=meshgrid(xg,yg);
    XiX=Xi(:,1); XiY=Xi(:,2);
    Phi_grid=lib_matrix_grid(XG(:),YG(:),p,system_name);
    dX_grid=reshape(Phi_grid*XiX,n_grid,n_grid);
    dY_grid=reshape(Phi_grid*XiY,n_grid,n_grid);

    combined=dX_grid.^2+dY_grid.^2;
    [~,min_idx]=min(combined(:));
    [r_fp,c_fp]=ind2sub([n_grid,n_grid],min_idx);
    x_fp=xg(c_fp); y_fp=yg(r_fp);
    amp_x=max(x_data)-min(x_data); amp_y=max(y_data)-min(y_data);
    amp_norm=sqrt(amp_x^2+amp_y^2);
    x_cent=mean(x_data); y_cent=mean(y_data);

    fp_x_norm=(x_fp-x_cent)/(amp_x+1e-8);
    fp_y_norm=(y_fp-y_cent)/(amp_y+1e-8);
    fp_dist=sqrt((x_fp-x_cent)^2+(y_fp-y_cent)^2)/(amp_norm+1e-8);
    dx_fp_gx=(dX_grid(r_fp,min(c_fp+1,n_grid))-dX_grid(r_fp,max(c_fp-1,1)))/(2*(xg(2)-xg(1))+1e-8);
    dx_fp_gy=(dX_grid(min(r_fp+1,n_grid),c_fp)-dX_grid(max(r_fp-1,1),c_fp))/(2*(yg(2)-yg(1))+1e-8);
    dy_fp_gx=(dY_grid(r_fp,min(c_fp+1,n_grid))-dY_grid(r_fp,max(c_fp-1,1)))/(2*(xg(2)-xg(1))+1e-8);
    dy_fp_gy=(dY_grid(min(r_fp+1,n_grid),c_fp)-dY_grid(max(r_fp-1,1),c_fp))/(2*(yg(2)-yg(1))+1e-8);
    nx1=[-dx_fp_gy,dx_fp_gx]/(norm([-dx_fp_gy,dx_fp_gx])+1e-8);
    nx2=[-dy_fp_gy,dy_fp_gx]/(norm([-dy_fp_gy,dy_fp_gx])+1e-8);
    cross_angle=acos(min(abs(dot(nx1,nx2)),1));
    group3=[fp_x_norm,fp_y_norm,fp_dist,cross_angle];

    [xnull_pts,ynull_x] =extract_nullcline_pts(dX_grid,xg,yg);
    [xnull_y,  ynull_pts]=extract_nullcline_pts(dY_grid,xg,yg);
    group1=nullcline_shape_features((xnull_pts-x_cent)/(amp_x+1e-8),(ynull_x-y_cent)/(amp_y+1e-8));
    group2=nullcline_shape_features((xnull_y-x_cent)/(amp_x+1e-8),(ynull_pts-y_cent)/(amp_y+1e-8));

    Phi_orbit=lib_matrix_grid(x_data,y_data,p,system_name);
    dx_orbit=Phi_orbit*XiX; dy_orbit=Phi_orbit*XiY;
    [~,max_dev_idx]=max(abs(dx_orbit));
    group4=[mean(dx_orbit)/(targetScaleX+1e-8), std(dx_orbit)/(targetScaleX+1e-8), ...
            mean(dy_orbit)/(targetScaleY+1e-8), std(dy_orbit)/(targetScaleY+1e-8), ...
            max_dev_idx/N];

    sgn_dx=sign(dx_orbit); sgn_dy=sign(dy_orbit);
    cross_x=sum(abs(diff(sgn_dx))>0); cross_y=sum(abs(diff(sgn_dy))>0);
    pos2neg_x=sum(diff(sgn_dx)<0); neg2pos_x=sum(diff(sgn_dx)>0);
    pos2neg_y=sum(diff(sgn_dy)<0); neg2pos_y=sum(diff(sgn_dy)>0);
    n_cycles=max(sum(abs(diff(sign(x_data-mean(x_data))))>0)/2,1);
    group6=[cross_x/n_cycles,(pos2neg_x-neg2pos_x)/(cross_x+1e-8), ...
            cross_y/n_cycles,(pos2neg_y-neg2pos_y)/(cross_y+1e-8)];

    g1=group1(:); g2=group2(:);
    group7=[g1(4)-g2(4);g1(5:8)-g2(5:8);g1(9)-g2(9);g1(10)-g2(10); ...
            g1(11)-g2(11);g1(12)-g2(12); ...
            clip_ratio(g1(4),g2(4));clip_ratio(g1(11),g2(11));clip_ratio(g1(12),g2(12))];

    nullcline_features=max(min([group1(:)',group2(:)',group3(:)', ...
                                group4(:)',group6(:)',group7(:)']',50),-50);
    if any(~isfinite(nullcline_features)), skipped=skipped+1; continue; end

    %% ── XCORR FEATURES ───────────────────────────────────────────────
    dx_obs=sgolayfilt(gradient(x_data_all,dt),sg_p,sg_f);
    dx_model=zeros(n_all,1);
    for k=2:n_all
        phi=lib_row(x_data_all(k-1),y_data_all(k-1),p,system_name);
        dx_model(k)=phi*XiX;
    end
    xcorr_features=compute_xcorr_features(dx_obs-dx_model,x_data_all,y_data_all);
    if any(~isfinite(xcorr_features)), skipped=skipped+1; continue; end

    %% ── TERM ACTIVITY AND TOPOLOGY ───────────────────────────────────
    Xi_ternary=sign(Xi).*(abs(Xi)>0.005);
    if size(Xi_ternary,1)<9
        Xi_ternary=[Xi_ternary;zeros(9-size(Xi_ternary,1),2)];
    end

    half=round(n_all/2); x_2nd=x_data_all(half:end); y_2nd=y_data_all(half:end);
    sustained=(var(x_2nd)>1e-4)&&(var(y_2nd)>1e-4);
    loop_err=norm([x_2nd(end)-x_2nd(1),y_2nd(end)-y_2nd(1)])/(norm([x_2nd(1),y_2nd(1)])+1e-8);
    is_closed=loop_err<0.25;
    seg=round(length(x_2nd)/3);
    amp_ratio=std(x_2nd(end-seg+1:end))/max(std(x_2nd(1:seg)),1e-8);
    if     amp_ratio>1.25; amp_trend='growing';
    elseif amp_ratio<0.75; amp_trend='decaying';
    else;                  amp_trend='stable'; end
    if     sustained&&is_closed&&strcmp(amp_trend,'stable'); topology='LIMIT CYCLE';
    elseif sustained&&strcmp(amp_trend,'decaying');          topology='DAMPED OSCILLATION';
    elseif ~sustained;                                       topology='STEADY STATE';
    else;                                                    topology='UNDETERMINED'; end

    structure_label = latent_class;
    coupling_label  = coupling_ch;
    params_saved    = rmfield_safe(p,{'hill_rep','hill_deg','hill_act','lb','ub'});
    params_saved.system        = system_name;
    params_saved.coupling_ch   = coupling_ch;

    fname=fullfile(save_dir,sprintf('example_%04d.mat',existing+saved+1));
    save(fname,'nullcline_features','xcorr_features','Xi_ternary', ...
        'topology','structure_label','coupling_label','params_saved', ...
        'x_data_all','y_data_all','z_data_all','t_ode','Xi','col_names');

    saved=saved+1;
    if mod(saved,10)==0
        fprintf('[%s|Lat%d|Ch%d] %d/%d (skip=%d att=%d)\n', ...
            system_name,latent_class,coupling_ch,saved,n_examples,skipped,attempt);
    end
end
fprintf('[%s|Lat%d|Ch%d] Done. saved=%d skip=%d att=%d\n', ...
    system_name,latent_class,coupling_ch,saved,skipped,attempt);
end


%% ── ODE FUNCTIONS ────────────────────────────────────────────────────────
% Coupling term — shared across all systems
% Ch0: -alpha_c * H_rep(I, hill_k0, hill_n)   [Hill repressor of I]
% Ch1: -beta_c  * I                            [linear additive]
% Ch2: -gamma_c * I * x                        [multiplicative degradation]
function c = coupling_term(I, x, p, coupling_ch)
    switch coupling_ch
        case 0; c = -p.alpha_c * p.hill_rep(I, p.hill_k0, p.hill_n);
        case 1; c = -p.beta_c  * I;
        case 2; c = -p.gamma_c * I * x;
    end
end

function ds=goodwin_ode(~,s,p,latent_class,coupling_ch)
    ds=zeros(3,1);
    x=s(1); y=s(2); I=s(3);
    switch coupling_ch
        case 0
            % I is the repressor — classic Goodwin chain topology
            ds(1) = p.alpha*p.hill_rep(I,p.hill_k0,p.hill_n) - p.d1*x;
        case 1
            % y represses x, I adds linear inhibition
            ds(1) = p.alpha*p.hill_rep(y,p.hill_k0,p.hill_n) - p.d1*x - p.beta_c*I;
        case 2
            % y represses x, I promotes multiplicative degradation of x
            ds(1) = p.alpha*p.hill_rep(y,p.hill_k0,p.hill_n) - p.d1*x - p.gamma_c*I*x;
    end
    ds(2) = p.ks*x - p.Vmax*p.hill_deg(y,p.hill_km);
    ds(3) = latent_z(s,p,latent_class);
end

function ds=brusselator_ode(~,s,p,latent_class,coupling_ch)
    ds=zeros(3,1);
    x=s(1); y=s(2); I=s(3);
    ds(1) = p.a-(p.b+1)*x+x^2*y + coupling_term(I,x,p,coupling_ch);
    ds(2) = p.b*x - x^2*y;
    ds(3) = latent_z(s,p,latent_class);
end

function ds=repressilator_ode(~,s,p,latent_class,coupling_ch)
    ds=zeros(3,1);
    x=s(1); y=s(2); I=s(3);
    switch coupling_ch
        case 0
            ds(1) = p.alpha*p.hill_rep(I,p.hill_k0,p.hill_n) - p.delta*p.hill_deg(x,p.hill_km);
        case 1
            ds(1) = p.alpha*p.hill_rep(y,p.hill_k0,p.hill_n) - p.delta*p.hill_deg(x,p.hill_km) - p.beta_c*I;
        case 2
            ds(1) = p.alpha*p.hill_rep(y,p.hill_k0,p.hill_n) - p.delta*p.hill_deg(x,p.hill_km) - p.gamma_c*I*x;
    end
    ds(2) = p.beta_rep*x - p.gamma_rep*y;
    ds(3) = latent_z(s,p,latent_class);
end

function ds=vanderpol_ode(~,s,p,latent_class,coupling_ch)
    ds=zeros(3,1);
    x=s(1); y=s(2); I=s(3);
    ds(1) = p.mu*(1-y^2)*x - p.omega*y + coupling_term(I,x,p,coupling_ch);
    ds(2) = x;
    ds(3) = latent_z(s,p,latent_class);
end

function dI=latent_z(s,p,latent_class)
    x=s(1); y=s(2); I=s(3);
    switch latent_class
        case 0; dI = p.kp*x - p.kd*I;
        case 1; dI = p.kp*y - (p.kd + p.km*x)*I;
        case 2; dI = p.kp*x*p.hill_rep(x,p.hill_k0,p.hill_n) - p.kd*I;
        case 3
            I_c=max(min(I,p.I_max),0);
            dI = p.kp*x*(1-I_c/p.I_max) - p.kcat*I_c/(p.Km+I_c);
        case 4; dI = p.kp*p.hill_rep(y,p.hill_k0,p.hill_n) - p.kd*I;
    end
end


%% ── PARAM SAMPLERS ───────────────────────────────────────────────────────
% All samplers include every parameter field needed across all latents
% and coupling channels. Fields unused by a given combination are ignored.

function p=sample_goodwin(hill_rep,hill_deg,hill_act,~)
    p=struct( ...
        'alpha',   5+7*rand(),        'd1',    0.3+0.7*rand(), ...
        'ks',      1+3*rand(),        'Vmax',  3+5*rand(),     ...
        'hill_n',  round(3+3*rand()), 'hill_k0',1.5+2.5*rand(), ...
        'hill_km', 0.4+1.6*rand(),    ...
        'alpha_c', 1+3*rand(),        ...  % Ch0: Hill coupling strength
        'beta_c',  0.5+2.0*rand(),    ...  % Ch1: linear coupling strength
        'gamma_c', 0.2+0.8*rand(),    ...  % Ch2: multiplicative coupling strength
        'kp',      1+2*rand(),        'kd',    0.5+1.5*rand(), ...
        'km',      0.5+1.5*rand(),    ...  % Lat1: x-dep degradation
        'I_max',   2+4*rand(),        ...  % Lat3: conservation ceiling
        'kcat',    2+5*rand(),        'Km',    0.3+1.5*rand(), ...
        'hill_rep',hill_rep,'hill_deg',hill_deg,'hill_act',hill_act, ...
        'noise',   0.01+0.09*rand(),  ...
        'lb',{{[0,-inf,-inf,-inf,0,0,0],[0,0,0,0,-inf,-inf,-inf]}}, ...
        'ub',{{[0,0,0,0,0,0,0],[0,inf,inf,inf,0,0,0]}});
end

function p=sample_brusselator(hill_rep,hill_deg,hill_act,~)
    while true
        a=1+2*rand(); b=4+6*rand();
        if b>a^2+1; break; end
    end
    p=struct('a',a,'b',b, ...
        'hill_n',  round(2+3*rand()), 'hill_k0',1.5+2.5*rand(), ...
        'hill_km', 0.5+1.5*rand(),    ...
        'alpha_c', 1+3*rand(),        'beta_c', 0.5+2.0*rand(), ...
        'gamma_c', 0.2+0.8*rand(),    ...
        'kp',      1+2*rand(),        'kd',    0.5+1.5*rand(), ...
        'km',      0.5+1.5*rand(),    'I_max', 2+4*rand(), ...
        'kcat',    2+5*rand(),        'Km',    0.3+1.5*rand(), ...
        'hill_rep',hill_rep,'hill_deg',hill_deg,'hill_act',hill_act, ...
        'noise',   0.01+0.09*rand(), ...
        'lb',{{repmat(-inf,1,5),repmat(-inf,1,5)}}, ...
        'ub',{{repmat(inf,1,5), repmat(inf,1,5)}});
end

function p=sample_repressilator(hill_rep,hill_deg,hill_act,~)
    p=struct( ...
        'alpha',    4+6*rand(),  'delta',    1+2*rand(), ...
        'beta_rep', 1+2*rand(),  'gamma_rep',0.5+1.5*rand(), ...
        'hill_n',   round(2+3*rand()), 'hill_k0',1+3*rand(), ...
        'hill_km',  0.4+1.6*rand(),    ...
        'alpha_c',  1+3*rand(),  'beta_c',  0.5+2.0*rand(), ...
        'gamma_c',  0.2+0.8*rand(),    ...
        'kp',       1+2*rand(),  'kd',      0.5+1.5*rand(), ...
        'km',       0.5+1.5*rand(),'I_max', 2+4*rand(), ...
        'kcat',     2+5*rand(),  'Km',      0.3+1.5*rand(), ...
        'hill_rep',hill_rep,'hill_deg',hill_deg,'hill_act',hill_act, ...
        'noise',    0.01+0.09*rand(), ...
        'lb',{{[0,-inf,-inf,-inf,0,0,0],[0,0,0,0,-inf,-inf,-inf]}}, ...
        'ub',{{[0,0,0,0,0,0,0],[0,inf,inf,inf,0,0,0]}});
end

function p=sample_vanderpol(hill_rep,hill_deg,hill_act,~)
    p=struct( ...
        'mu',     0.3+0.7*rand(), 'omega',  0.8+0.4*rand(), ...
        'hill_n', round(2+3*rand()),'hill_k0',1+2*rand(), ...
        'hill_km',0.5+1.5*rand(), ...
        'alpha_c',1+3*rand(),    'beta_c',  0.5+2.0*rand(), ...
        'gamma_c',0.2+0.8*rand(),...
        'kp',     1+2*rand(),    'kd',      0.5+1.5*rand(), ...
        'km',     0.5+1.5*rand(),'I_max',   2+4*rand(), ...
        'kcat',   2+5*rand(),    'Km',      0.3+1.5*rand(), ...
        'hill_rep',hill_rep,'hill_deg',hill_deg,'hill_act',hill_act, ...
        'noise',  0.01+0.09*rand(), ...
        'lb',{{repmat(-inf,1,7),repmat(-inf,1,7)}}, ...
        'ub',{{repmat(inf,1,7), repmat(inf,1,7)}});
end


%% ── LIBRARY BUILDERS ─────────────────────────────────────────────────────
function lib=standard_lib(x,y,N_,p)
    lib.Theta_full=[ones(N_,1),x,x.^2,x.^3,y,y.^2,y.^3, ...
                   p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
    lib.Theta_poly=lib.Theta_full(:,1:7);
    lib.col_names={'1','x','x^2','x^3','y','y^2','y^3','HillRep','HillDeg'};
end

function lib=brusselator_lib(x,y,N_,p)
    lib.Theta_full=[ones(N_,1),x,x.^2,y,x.^2.*y, ...
                   p.hill_rep(y,p.hill_k0,p.hill_n),p.hill_deg(y,p.hill_km)];
    lib.Theta_poly=lib.Theta_full(:,1:5);
    lib.col_names={'1','x','x^2','y','x^2y','HillRep','HillDeg'};
end

function Phi=lib_matrix_grid(x,y,p,system_name)
    x=x(:);y=y(:);n=length(x);
    hr=p.hill_rep(y,p.hill_k0,p.hill_n); hd=p.hill_deg(y,p.hill_km);
    if strcmp(system_name,'brusselator')
        Phi=[ones(n,1),x,x.^2,y,x.^2.*y,hr,hd];
    else
        Phi=[ones(n,1),x,x.^2,x.^3,y,y.^2,y.^3,hr,hd];
    end
end

function row=lib_row(x,y,p,system_name)
    hr=p.hill_rep(y,p.hill_k0,p.hill_n); hd=p.hill_deg(y,p.hill_km);
    if strcmp(system_name,'brusselator')
        row=[1,x,x^2,y,x^2*y,hr,hd];
    else
        row=[1,x,x^2,x^3,y,y^2,y^3,hr,hd];
    end
end


%% ── NULLCLINE AND XCORR HELPERS ──────────────────────────────────────────
function [xpts,ypts]=extract_nullcline_pts(F,xg,yg)
    Sl=F(:,1:end-1);Sr=F(:,2:end);hmask=Sl.*Sr<0;
    [rows,cols]=find(hmask);
    if ~isempty(rows)
        t=Sl(hmask)./(Sl(hmask)-Sr(hmask));
        xh=xg(cols)'+t.*(xg(cols+1)'-xg(cols)');yh=yg(rows)';
    else;xh=[];yh=[];end
    Su=F(1:end-1,:);Sd=F(2:end,:);vmask=Su.*Sd<0;
    [rows,cols]=find(vmask);
    if ~isempty(rows)
        t=Su(vmask)./(Su(vmask)-Sd(vmask));
        xv=xg(cols)';yv=yg(rows)'+t.*(yg(rows+1)'-yg(rows)');
    else;xv=[];yv=[];end
    xpts=[xh;xv];ypts=[yh;yv];
    if isempty(xpts),xpts=0;ypts=0;end
end

function feat=nullcline_shape_features(xpts,ypts)
    if length(xpts)<6,feat=zeros(12,1);return;end
    cx=mean(xpts);cy=mean(ypts);
    pts=[xpts(:)-cx,ypts(:)-cy];
    C=(pts'*pts)/max(size(pts,1)-1,1);
    [V,D]=eig(C);evals=diag(D);
    [evals_s,ord]=sort(evals,'descend');V=V(:,ord);
    pca_angle=atan2(V(2,1),V(1,1));
    aspect=min(sqrt(evals_s(1))/(sqrt(evals_s(2))+1e-8),20);
    proj_along=pts*V(:,1);proj_across=pts*V(:,2);
    if std(proj_along)<1e-8,feat=zeros(12,1);return;end
    q_bounds=quantile(proj_along,[0 0.25 0.5 0.75 1.0]);
    curv_feat=zeros(4,1);
    for q=1:4
        mask=proj_along>=q_bounds(q)&proj_along<q_bounds(q+1);
        if sum(mask)>1,curv_feat(q)=std(proj_across(mask));end
    end
    pos_mask=proj_along>0;neg_mask=proj_along<=0;
    if sum(pos_mask)<2||sum(neg_mask)<2
        asym_along=0;asym_across=0;
    else
        pos_std=std(proj_along(pos_mask));neg_std=std(proj_along(neg_mask));
        asym_along=max(min((pos_std+1e-8)/(neg_std+1e-8)-1,5),-5);
        asym_across=mean(proj_across(pos_mask))-mean(proj_across(neg_mask));
    end
    if length(xpts)>2
        arc_len=min(sum(sqrt(diff(xpts(:)).^2+diff(ypts(:)).^2)),20);
    else;arc_len=0;end
    spread_ratio=min(std(proj_across)/(std(proj_along)+1e-8),5);
    feat=[cx;cy;pca_angle;aspect;curv_feat;asym_along;asym_across;arc_len;spread_ratio];
end

function r=clip_ratio(a,b)
    if abs(b)<1e-6;r=0;else;r=max(min(a/b,5),-5);end
end

function xcf=compute_xcorr_features(resid_dx,x_data_all,y_data_all)
    max_lag=50;
    x_n=(x_data_all-mean(x_data_all))/(std(x_data_all)+1e-8);
    y_n=(y_data_all-mean(y_data_all))/(std(y_data_all)+1e-8);
    r_n=(resid_dx  -mean(resid_dx))  /(std(resid_dx)  +1e-8);
    ml=min([length(x_n),length(y_n),length(r_n)]);
    x_n=x_n(1:ml);y_n=y_n(1:ml);r_n=r_n(1:ml);
    [xc,lx]=xcorr(r_n,x_n,max_lag,'normalized');
    [pv_x,pi_x]=max(abs(xc));pl_x=lx(pi_x);pw_x=sum(abs(xc)>pv_x/2);
    px=mean(abs(xc(lx>0)));nx=mean(abs(xc(lx<0)));
    la_x=(px-nx)/(px+nx+1e-8);cz_x=xc(lx==0);
    [yc,ly]=xcorr(r_n,y_n,max_lag,'normalized');
    [pv_y,pi_y]=max(abs(yc));pl_y=ly(pi_y);pw_y=sum(abs(yc)>pv_y/2);
    py=mean(abs(yc(ly>0)));ny_=mean(abs(yc(ly<0)));
    la_y=(py-ny_)/(py+ny_+1e-8);cz_y=yc(ly==0);
    xcf=[pl_x;pv_x;pw_x;la_x;cz_x;pl_y;pv_y;pw_y;la_y;cz_y];
end

function s=rmfield_safe(s,fields)
    for i=1:length(fields)
        if isfield(s,fields{i}),s=rmfield(s,fields{i});end
    end
end