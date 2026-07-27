clear; close all; clc;

% Rewrites nullcline_features in every existing V3 .mat file using the
% fixed nullcline_shape_features function. Specifically fixes features
% 28-31 (asym_along, asym_across, arc_len, spread_ratio) which blew up
% when the nullcline point cloud was degenerate.
%
% Safe to re-run — recomputes unconditionally and overwrites in place.
% Does not touch any other saved variable.

DATA_ROOT   = 'C:/Users/nickj/MATLAB Drive/LTI Network';

% All active latent folders — edit if your set differs
active_lats = [0, 1, 8, 9, 10, 11, 12];
n_sys       = 4;

folders = {};
for lat = active_lats
    for sys = 1:n_sys
        folder = sprintf('%s/V3Lat%d_Sys%d_NNdata', DATA_ROOT, lat, sys);
        if exist(folder, 'dir')
            folders{end+1} = folder; %#ok<AGROW>
        else
            fprintf('  [MISSING] %s\n', folder);
        end
    end
end

fprintf('Found %d folders to process.\n\n', length(folders));

total_updated = 0;
total_bad     = 0;

for f_idx = 1:length(folders)
    folder = folders{f_idx};
    files  = dir(fullfile(folder, '*.mat'));
    updated = 0;

    for k = 1:length(files)
        fpath = fullfile(folder, files(k).name);

        try
            d = load(fpath, 'x_data_all', 'y_data_all', 'Xi', 'params_saved');

            if ~isfield(d,'x_data_all') || ~isfield(d,'Xi') || ...
               ~isfield(d,'params_saved')
                fprintf('  [MISSING FIELDS] %s\n', fpath);
                total_bad = total_bad+1;
                continue
            end

            x_data_all = d.x_data_all(:);
            y_data_all = d.y_data_all(:);
            Xi         = d.Xi;
            p          = d.params_saved;
            sys_name   = p.system;

            N        = min(300, length(x_data_all));
            x_data   = x_data_all(1:N);
            y_data   = y_data_all(1:N);

            % ── Rebuild grid and evaluate fitted model ────────────────
            x_min = min(x_data); x_max = max(x_data);
            y_min = min(y_data); y_max = max(y_data);
            n_grid = 50;
            xg = linspace(x_min, x_max, n_grid);
            yg = linspace(y_min, y_max, n_grid);
            [XG, YG] = meshgrid(xg, yg);

            XiX = Xi(:,1);
            XiY = Xi(:,2);

            Phi_grid = lib_matrix_grid(XG(:), YG(:), p, sys_name);
            dX_grid  = reshape(Phi_grid * XiX, n_grid, n_grid);
            dY_grid  = reshape(Phi_grid * XiY, n_grid, n_grid);

            % ── Group 3: fixed point ──────────────────────────────────
            combined = dX_grid.^2 + dY_grid.^2;
            [~, min_idx] = min(combined(:));
            [r_fp, c_fp] = ind2sub([n_grid, n_grid], min_idx);
            x_fp = xg(c_fp);
            y_fp = yg(r_fp);

            amp_x    = max(x_data)-min(x_data);
            amp_y    = max(y_data)-min(y_data);
            amp_norm = sqrt(amp_x^2+amp_y^2);
            x_cent   = mean(x_data);
            y_cent   = mean(y_data);

            fp_x_norm = (x_fp-x_cent)/(amp_x+1e-8);
            fp_y_norm = (y_fp-y_cent)/(amp_y+1e-8);
            fp_dist   = sqrt((x_fp-x_cent)^2+(y_fp-y_cent)^2)/(amp_norm+1e-8);

            dx_fp_gx = (dX_grid(r_fp,min(c_fp+1,n_grid))- ...
                        dX_grid(r_fp,max(c_fp-1,1)))/(2*(xg(2)-xg(1))+1e-8);
            dx_fp_gy = (dX_grid(min(r_fp+1,n_grid),c_fp)- ...
                        dX_grid(max(r_fp-1,1),c_fp))/(2*(yg(2)-yg(1))+1e-8);
            dy_fp_gx = (dY_grid(r_fp,min(c_fp+1,n_grid))- ...
                        dY_grid(r_fp,max(c_fp-1,1)))/(2*(xg(2)-xg(1))+1e-8);
            dy_fp_gy = (dY_grid(min(r_fp+1,n_grid),c_fp)- ...
                        dY_grid(max(r_fp-1,1),c_fp))/(2*(yg(2)-yg(1))+1e-8);

            nx1 = [-dx_fp_gy,dx_fp_gx]/(norm([-dx_fp_gy,dx_fp_gx])+1e-8);
            nx2 = [-dy_fp_gy,dy_fp_gx]/(norm([-dy_fp_gy,dy_fp_gx])+1e-8);
            cross_angle = acos(min(abs(dot(nx1,nx2)),1));

            group3 = [fp_x_norm, fp_y_norm, fp_dist, cross_angle];

            % ── Groups 1 & 2: nullcline shape (FIXED function) ────────
            [xnull_pts, ynull_x] = extract_nullcline_pts(dX_grid, xg, yg);
            [xnull_y,   ynull_pts] = extract_nullcline_pts(dY_grid, xg, yg);

            xnull_pts_n = (xnull_pts-x_cent)/(amp_x+1e-8);
            ynull_x_n   = (ynull_x -y_cent)/(amp_y+1e-8);
            xnull_y_n   = (xnull_y -x_cent)/(amp_x+1e-8);
            ynull_pts_n = (ynull_pts-y_cent)/(amp_y+1e-8);

            group1 = nullcline_shape_features_fixed(xnull_pts_n, ynull_x_n);
            group2 = nullcline_shape_features_fixed(xnull_y_n,   ynull_pts_n);

            % ── Group 4: orbit-nullcline distance ─────────────────────
            dt   = 0.05; sg_p = 3; sg_f = 11;
            dxdt = sgolayfilt(gradient(x_data,dt), sg_p, sg_f);
            dydt = sgolayfilt(gradient(y_data,dt), sg_p, sg_f);
            targetScaleX = norm(dxdt,2);
            targetScaleY = norm(dydt,2);

            Phi_orbit  = lib_matrix_grid(x_data, y_data, p, sys_name);
            dx_orbit   = Phi_orbit * XiX;
            dy_orbit   = Phi_orbit * XiY;

            mean_dx_orb = mean(dx_orbit)/(targetScaleX+1e-8);
            std_dx_orb  = std(dx_orbit) /(targetScaleX+1e-8);
            mean_dy_orb = mean(dy_orbit)/(targetScaleY+1e-8);
            std_dy_orb  = std(dy_orbit) /(targetScaleY+1e-8);
            [~,max_dev_idx] = max(abs(dx_orbit));
            phase_max_dev   = max_dev_idx/N;

            group4 = [mean_dx_orb,std_dx_orb,mean_dy_orb,std_dy_orb,phase_max_dev];

            % ── Group 6: crossing counts ──────────────────────────────
            sgn_dx  = sign(dx_orbit); sgn_dy = sign(dy_orbit);
            cross_x = sum(abs(diff(sgn_dx))>0);
            cross_y = sum(abs(diff(sgn_dy))>0);

            pos2neg_x = sum(diff(sgn_dx)<0); neg2pos_x = sum(diff(sgn_dx)>0);
            asym_x    = (pos2neg_x-neg2pos_x)/(cross_x+1e-8);
            pos2neg_y = sum(diff(sgn_dy)<0); neg2pos_y = sum(diff(sgn_dy)>0);
            asym_y    = (pos2neg_y-neg2pos_y)/(cross_y+1e-8);

            n_cycles     = max(sum(abs(diff(sign(x_data-mean(x_data))))>0)/2,1);
            cross_x_norm = cross_x/n_cycles;
            cross_y_norm = cross_y/n_cycles;

            group6 = [cross_x_norm,asym_x,cross_y_norm,asym_y];

            % ── Group 7: cross-nullcline comparison ───────────────────
            g1 = group1(:); g2 = group2(:);
            diff_aspect = g1(4)-g2(4);
            diff_curv   = g1(5:8)-g2(5:8);
            diff_asym_a = g1(9)-g2(9);
            diff_asym_b = g1(10)-g2(10);
            diff_arc    = g1(11)-g2(11);
            diff_spread = g1(12)-g2(12);
            ratio_aspect = clip_ratio(g1(4),g2(4));
            ratio_arc    = clip_ratio(g1(11),g2(11));
            ratio_spread = clip_ratio(g1(12),g2(12));

            group7 = [diff_aspect;diff_curv;diff_asym_a;diff_asym_b; ...
                      diff_arc;diff_spread;ratio_aspect;ratio_arc;ratio_spread];

            % ── Assemble and clip ─────────────────────────────────────
            nullcline_features = [group1(:)',group2(:)',group3(:)', ...
                                  group4(:)',group6(:)',group7(:)']';
            nullcline_features = max(min(nullcline_features,50),-50);

            if any(~isfinite(nullcline_features))
                fprintf('  [STILL BAD] %s\n', fpath);
                total_bad = total_bad+1;
                continue
            end

            % Overwrite nullcline_features — -append replaces if exists
            save(fpath, 'nullcline_features', '-append');
            updated       = updated+1;
            total_updated = total_updated+1;

        catch err
            fprintf('  [ERROR] %s: %s\n', fpath, err.message);
            total_bad = total_bad+1;
        end
    end

    fprintf('[%s]  recomputed=%d\n', ...
        folder(max(1,end-35):end), updated);
end

fprintf('\n=== Complete ===\n');
fprintf('Recomputed: %d files\n', total_updated);
fprintf('Errors:     %d files\n', total_bad);


%% ── FIXED NULLCLINE SHAPE FEATURES ──────────────────────────────────────
% Key fixes vs original:
%   1. Guard raised from 4 to 6 minimum points
%   2. Guard added for degenerate proj_along (std < 1e-8)
%   3. Empty mask protection for asym_along and asym_across
%   4. arc_len capped at 20 to prevent explosion
function feat = nullcline_shape_features_fixed(xpts, ypts)
    if length(xpts) < 6
        feat = zeros(12,1); return
    end

    cx = mean(xpts); cy = mean(ypts);
    pts = [xpts(:)-cx, ypts(:)-cy];
    C   = (pts'*pts)/max(size(pts,1)-1,1);
    [V,D] = eig(C);
    evals = diag(D);
    [evals_s,ord] = sort(evals,'descend');
    V = V(:,ord);

    pca_angle = atan2(V(2,1),V(1,1));
    aspect    = min(sqrt(evals_s(1))/(sqrt(evals_s(2))+1e-8), 20);

    proj_along  = pts*V(:,1);
    proj_across = pts*V(:,2);

    % Degenerate projection guard
    if std(proj_along) < 1e-8
        feat = zeros(12,1); return
    end

    q_bounds  = quantile(proj_along,[0 0.25 0.5 0.75 1.0]);
    curv_feat = zeros(4,1);
    for q = 1:4
        mask = proj_along>=q_bounds(q) & proj_along<q_bounds(q+1);
        if sum(mask)>1, curv_feat(q) = std(proj_across(mask)); end
    end

    pos_mask = proj_along > 0;
    neg_mask = proj_along <= 0;

    % Empty mask guard — prevents std([]) = NaN
    if sum(pos_mask) < 2 || sum(neg_mask) < 2
        asym_along  = 0;
        asym_across = 0;
    else
        pos_std     = std(proj_along(pos_mask));
        neg_std     = std(proj_along(neg_mask));
        asym_along  = max(min((pos_std+1e-8)/(neg_std+1e-8)-1, 5), -5);
        asym_across = mean(proj_across(pos_mask)) - mean(proj_across(neg_mask));
    end

    if length(xpts) > 2
        arc_len = sum(sqrt(diff(xpts(:)).^2 + diff(ypts(:)).^2));
        arc_len = min(arc_len, 20);   % hard cap — prevents explosion
    else
        arc_len = 0;
    end

    spread_ratio = min(std(proj_across)/(std(proj_along)+1e-8), 5);

    feat = [cx; cy; pca_angle; aspect; curv_feat; ...
            asym_along; asym_across; arc_len; spread_ratio];
end


%% ── SHARED HELPERS ───────────────────────────────────────────────────────
function Phi = lib_matrix_grid(x, y, p, system_name)
    % hill_rep and hill_deg are stripped from params_saved before saving
    % so we reconstruct them locally from the saved scalar parameters
    x = x(:); y = y(:); n = length(x);
    hr = p.hill_k0^p.hill_n ./ (p.hill_k0^p.hill_n + y.^p.hill_n);
    hd = y ./ (p.hill_km + y);
    if strcmp(system_name, 'brusselator')
        Phi = [ones(n,1), x, x.^2, y, x.^2.*y, hr, hd];
    else
        Phi = [ones(n,1), x, x.^2, x.^3, y, y.^2, y.^3, hr, hd];
    end
end

function [xpts, ypts] = extract_nullcline_pts(F, xg, yg)
    Sl = F(:,1:end-1); Sr = F(:,2:end);
    hmask = Sl.*Sr < 0;
    [rows,cols] = find(hmask);
    if ~isempty(rows)
        t  = Sl(hmask)./(Sl(hmask)-Sr(hmask));
        xh = xg(cols)'+t.*(xg(cols+1)'-xg(cols)');
        yh = yg(rows)';
    else; xh=[]; yh=[]; end
    Su = F(1:end-1,:); Sd = F(2:end,:);
    vmask = Su.*Sd < 0;
    [rows,cols] = find(vmask);
    if ~isempty(rows)
        t  = Su(vmask)./(Su(vmask)-Sd(vmask));
        xv = xg(cols)';
        yv = yg(rows)'+t.*(yg(rows+1)'-yg(rows)');
    else; xv=[]; yv=[]; end
    xpts = [xh;xv]; ypts = [yh;yv];
    if isempty(xpts), xpts=0; ypts=0; end
end

function r = clip_ratio(a, b)
    if abs(b)<1e-6; r=0;
    else; r=max(min(a/b,5),-5); end
end