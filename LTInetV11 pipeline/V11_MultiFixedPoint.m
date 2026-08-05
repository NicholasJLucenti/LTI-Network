function fp_features = V11_MultiFixedPoint(dX, dY, xg, yg)
Fmag = dX.^2 + dY.^2;
[ng, ~] = size(Fmag);

thresh = quantile(Fmag(:), 0.02);
candidate_mask = Fmag <= thresh;

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
    [~, mi] = min(Fmag(:));
    [rf, cf] = ind2sub([ng, ng], mi);
    rows = rf; cols = cf;
end

pts = [rows, cols];
clustered = cluster_nearby(pts, 3);
n_fp_found = min(size(clustered,1), 3);

xc = mean(xg); yc_ = mean(yg);
ax = max(xg)-min(xg); ay = max(yg)-min(yg);

if n_fp_found >= 1
    r = clustered(1,1); c = clustered(1,2);
    fp_x0 = (xg(c) - xc) / (ax + 1e-8);
    fp_y0 = (yg(r) - yc_) / (ay + 1e-8);
else
    fp_x0 = 0; fp_y0 = 0;
end

fp_features = [n_fp_found; fp_x0; fp_y0];
fp_features(~isfinite(fp_features)) = 0;
fp_features = max(min(fp_features, 50), -50);
end

function clustered = cluster_nearby(pts, radius)
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
