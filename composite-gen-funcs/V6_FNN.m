function fnn_features = V6_FNN(x_data, dt)
% False-nearest-neighbors embedding-dimension diagnostic.
% Tests whether the observed 1D series x(t) needs more than 2 delay
% coordinates to unfold the attractor without self-intersections —
% direct evidence for/against a hidden (latent) dimension.
%
% Returns 4 numbers: FNN fraction at embedding dims 1,2,3,4.
% Restricted to dims 1-4 (not deeper) since N~300 makes FNN unreliable
% at higher dims (Kennel et al. 1992; the classic guidance is N>=1000
% for stable estimates at dim>4 — flagged explicitly, not hidden).
%
% rtol/atol thresholds follow the standard Kennel heuristic.

x = x_data(:);
N = length(x);
rtol = 15;               % standard Kennel threshold
atol = 2;
Ra = std(x) + 1e-8;      % attractor size proxy for atol criterion

% Delay time: first minimum of autocorrelation, capped to reasonable range
tau = estimate_delay(x, dt);

max_dim = 4;
fnn_features = zeros(max_dim,1);

for m = 1:max_dim
    n_pts = N - m*tau;
    if n_pts < 20
        fnn_features(m) = NaN; continue;
    end
    % Embed at dimension m
    Em = zeros(n_pts, m);
    for k = 1:m
        Em(:,k) = x((1:n_pts) + (k-1)*tau);
    end
    % Embed at dimension m+1 (for the "next coordinate" test)
    n_pts1 = N - (m+1-1)*tau - tau;
    if n_pts1 < 20
        fnn_features(m) = NaN; continue;
    end
    n_common = min(n_pts, n_pts1);
    Em  = Em(1:n_common,:);
    xm1 = x((1:n_common) + m*tau);   % the extra coordinate at dim m+1

    false_count = 0; valid_count = 0;
    % Subsample for tractability at N~300 (full pairwise is fine, N small)
    D = pdist2_local(Em);
    D(1:n_common+1:end) = inf;
    [dmin, idx] = min(D, [], 2);

    for i = 1:n_common
        j = idx(i);
        if ~isfinite(dmin(i)) || dmin(i) < 1e-10; continue; end
        extra_diff = abs(xm1(i) - xm1(j));
        crit1 = extra_diff / dmin(i);
        crit2 = sqrt(dmin(i)^2 + extra_diff^2) / Ra;
        valid_count = valid_count + 1;
        if crit1 > rtol || crit2 > atol
            false_count = false_count + 1;
        end
    end

    if valid_count > 0
        fnn_features(m) = false_count / valid_count;
    else
        fnn_features(m) = NaN;
    end
end

fnn_features(~isfinite(fnn_features)) = 1.0;  % conservative: treat as "still unfolding"
fnn_features = max(min(fnn_features, 1), 0);

end

function tau = estimate_delay(x, dt)
% First zero-crossing (or first minimum) of autocorrelation, in samples
N = length(x);
x = x - mean(x);
maxlag = min(round(N/4), 60);
ac = zeros(maxlag,1);
v0 = sum(x.^2);
if v0 < 1e-10; tau = 1; return; end
for L = 1:maxlag
    ac(L) = sum(x(1:end-L).*x(L+1:end)) / v0;
end
below = find(ac <= 0, 1, 'first');
if isempty(below)
    [~,below] = min(ac);
end
tau = max(below, 1);
end

function D = pdist2_local(E)
% Pairwise Euclidean distance matrix, no toolbox dependency
n = size(E,1);
sq = sum(E.^2,2);
D2 = sq + sq' - 2*(E*E');
D2(D2<0) = 0;
D = sqrt(D2);
end