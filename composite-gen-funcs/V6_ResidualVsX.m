function feat = V6_ResidualVsX(x_data, resid_dx, resid_dy, y_data)
% V6_ResidualVsX — bins the SINDy residual by the INSTANTANEOUS VALUE of
% x (and y), not by time-lag (xcorr_features) or oscillation phase
% (phase-resolved features in all_new_features). This directly probes
% "how does the residual behave as a function of the current x value",
% which is the one thing none of the existing 166 features ask.
%
% Motivation: C0 (dI/dt = kp*x - kd*I), C2 (dI/dt = kp*x*Hrep(x,k0,n) - kd*I),
% and C4 (dI/dt = kp*Hrep(y,k0,n) - kd*I) all produce broadly similar
% TRAJECTORY SHAPES (diagnosed as near-degenerate under nullcline/xcorr
% features), but their underlying functional form differs specifically in
% how the latent's drive term depends on x: C0 is linear in x, C2 is
% non-monotonic in x (rises then self-represses at high x), C4 doesn't
% depend on x at all (depends on y instead). Binning the residual by
% x-QUANTILE and checking for monotonicity/curvature is a direct proxy
% for recovering that functional dependence, since the residual carries
% whatever structure the base library (build_lib) failed to capture in
% the x-equation fit.
%
% Returns 16 features:
%   [1:5]   mean residual_dx in each of 5 x-value quantile bins (normalized)
%   [6]     monotonicity score: Spearman-like rank correlation between
%           x-quantile-bin index and mean residual in that bin
%           (near +-1 = monotonic i.e. C0/C4-like; near 0 = non-monotonic,
%           i.e. consistent with C2's self-repression hump)
%   [7]     curvature score: second-difference of the 5 binned means
%           (large magnitude = pronounced hump/dip, consistent with C2)
%   [8]     x-value at which mean residual peaks (normalized position,
%           0=low end of x range, 1=high end) — for C2, expect a peak
%           away from the boundary; for monotonic C0/C4, expect peak
%           AT a boundary
%   [9:13]  same 5-bin analysis but for resid_dy binned by y-value
%   [14]    y-monotonicity score (analogous to [6] but for resid_dy vs y)
%   [15]    y-curvature score (analogous to [7] but for resid_dy vs y)
%   [16]    cross term: correlation between resid_dx-vs-x monotonicity
%           and resid_dy-vs-y monotonicity

    n_bins = 5;

    [dx_bins, dx_mono, dx_curv, dx_peak_pos] = bin_and_analyze(x_data, resid_dx, n_bins);
    [dy_bins, dy_mono, dy_curv, ~]           = bin_and_analyze(y_data, resid_dy, n_bins);

    cross_term = dx_mono * dy_mono;  % same-sign monotonicity -> correlated; near 0 or opposite sign -> x-specific structure

    feat = [dx_bins(:); dx_mono; dx_curv; dx_peak_pos; dy_bins(:); dy_mono; dy_curv; cross_term];
    feat(~isfinite(feat)) = 0;
    feat = max(min(feat, 20), -20);
end


function [bin_means, mono_score, curv_score, peak_pos] = bin_and_analyze(driver, resid, n_bins)
    driver = driver(:); resid = resid(:);
    n = min(length(driver), length(resid));
    driver = driver(1:n); resid = resid(1:n);

    r_std = std(resid) + 1e-8;
    resid_n = resid / r_std;

    % Quantile-based bin edges (equal-count bins, robust to skewed x distributions)
    edges = quantile(driver, linspace(0, 1, n_bins+1));
    edges(1) = edges(1) - 1e-8; edges(end) = edges(end) + 1e-8;  % ensure inclusive boundaries

    bin_means = zeros(n_bins, 1);
    bin_centers = zeros(n_bins, 1);
    for b = 1:n_bins
        mask = driver >= edges(b) & driver < edges(b+1);
        if sum(mask) >= 3
            bin_means(b) = mean(resid_n(mask));
        end
        bin_centers(b) = 0.5*(edges(b) + edges(b+1));
    end

    % Monotonicity: correlation between bin INDEX (1..n_bins, proxy for
    % increasing x) and bin_means. +-1 = monotonic, near 0 = non-monotonic.
    idx = (1:n_bins)';
    if std(bin_means) > 1e-8
        mono_score = corr_manual(idx, bin_means);
    else
        mono_score = 0;
    end

    % Curvature: mean absolute second difference of the binned means —
    % large value indicates a hump/dip (non-monotonic structure)
    if n_bins >= 3
        second_diff = diff(bin_means, 2);
        curv_score = mean(abs(second_diff));
    else
        curv_score = 0;
    end

    % Peak position: where (in normalized 0-1 x-range) does the max |residual| occur
    [~, peak_bin] = max(abs(bin_means));
    peak_pos = (peak_bin - 1) / max(n_bins - 1, 1);
end


function r = corr_manual(a, b)
    a = a(:); b = b(:);
    a = a - mean(a); b = b - mean(b);
    denom = sqrt(sum(a.^2) * sum(b.^2));
    if denom < 1e-10
        r = 0;
    else
        r = sum(a.*b) / denom;
    end
end