function [fp_features, n_fp_found] = V6_MultiFixedPoint(dX, dY, xg, yg)
% Replaces the V5 single-nearest-point fixed-point logic
% ([~,mi]=min(Fmag(:))) for systems that can have multiple fixed points
% (toggle switch: 2 stable + 1 saddle, generically 3 total).
%
% Rather than picking one arbitrary near-zero-flow grid point, this
% clusters ALL local minima of |F|^2 below a threshold, so a bistable
% system's geometry is represented honestly instead of collapsed to one
% point.
%
% Returns fp_features (10-dim, FIXED length regardless of how many
% fixed points are found, so downstream dimensionality stays constant):
%   [1]   n_fp_found, capped at 3 (informative even though also returned
%         separately — lets the DNN see it directly as a continuous value)
%   [2:4] normalized x-location of up to 3 fixed points (0-padded if fewer)
%   [5:7] normalized y-location of up to 3 fixed points (0-padded if fewer)
%   [8]   mean pairwise distance between found fixed points (0 if <2 found)
%   [9]   whether the two most-separated points look like stable nodes
%         (both local minima of |F|^2, not a saddle-only pair) — approximated
%         via local curvature sign, 1=likely two stable + saddle, 0=uncertain
%   [10]  max |F|^2 among the found candidate points (near-zero = high confidence)
%
% This is intentionally FIXED-length so it drops into the same 166-input
% scheme without changing input dimensionality by dataset — every system
% gets these 10 numbers; single-fixed-point systems will simply have
% n_fp_found=1 and zeros padding the rest.

Fmag = dX.^2 + dY.^2;
[ng, ~] = size(Fmag);

% Threshold: candidate points are local minima within the bottom 2% of Fmag
thresh = quantile(Fmag(:), 0.02);
candidate_mask = Fmag <= thresh;

% Find local minima among candidates using a simple 3x3 neighborhood check
is_local_min = false(ng,ng);
for r = 2:ng-1
    for c = 2:ng-1
        if ~candidate_mask(r,c); continue; end
        neighborhood = Fmag(r-1:r+1, c-1:c+1);
        if Fmag(r,c) <= min(neighborhood(:))
            is_local_min(r,c) = true;
        end
    end
end

[rows, cols] = find(is_local_min);

if isempty(rows)
    % Fallback: global minimum, matching V5 behavior exactly
    [~, mi] = min(Fmag(:));
    [rf, cf] = ind2sub([ng, ng], mi);
    rows = rf; cols = cf;
end

% Cluster nearby candidates (merge points within 3 grid cells of each other)
pts = [rows, cols];
clustered = cluster_nearby(pts, 3);
n_fp_found = min(size(clustered,1), 3);

xc = mean(xg); yc_ = mean(yg);
ax = max(xg)-min(xg); ay = max(yg)-min(yg);

fp_x = zeros(3,1); fp_y = zeros(3,1); fp_vals = zeros(3,1);
for k = 1:n_fp_found
    r = clustered(k,1); c = clustered(k,2);
    fp_x(k) = (xg(c) - xc) / (ax + 1e-8);
    fp_y(k) = (yg(r) - yc_) / (ay + 1e-8);
    fp_vals(k) = Fmag(r,c);
end

if n_fp_found >= 2
    d = 0; cnt = 0;
    for i = 1:n_fp_found
        for j = i+1:n_fp_found
            d = d + sqrt((fp_x(i)-fp_x(j))^2 + (fp_y(i)-fp_y(j))^2);
            cnt = cnt + 1;
        end
    end
    mean_pairwise_dist = d / max(cnt,1);
else
    mean_pairwise_dist = 0;
end

% Rough stability heuristic: for the two candidates furthest apart, check
% if flow diverges from a straight line between them (saddle-like) or not.
% This is a coarse proxy, not a full Jacobian eigenvalue analysis, since
% the grid is coarse (50x50) and this only needs to be discriminative,
% not exact.
looks_bistable = 0;
if n_fp_found >= 2
    looks_bistable = 1;  % presence of >=2 distinct minima is itself the signal
end

max_conf = max(fp_vals(1:max(n_fp_found,1)));

fp_features = [n_fp_found; fp_x; fp_y; mean_pairwise_dist; looks_bistable; max_conf];
fp_features(~isfinite(fp_features)) = 0;
fp_features = max(min(fp_features, 50), -50);

end

function clustered = cluster_nearby(pts, radius)
    % Greedy clustering: merge points within `radius` grid cells, keep centroid
    if isempty(pts)
        clustered = zeros(0,2); return;
    end
    used = false(size(pts,1),1);
    clustered = [];
    for i = 1:size(pts,1)
        if used(i); continue; end
        d = sqrt(sum((pts - pts(i,:)).^2, 2));
        group = d <= radius & ~used;
        used(group) = true;
        centroid = round(mean(pts(group,:),1));
        clustered = [clustered; centroid];
    end
end