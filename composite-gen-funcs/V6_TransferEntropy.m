function te_features = V6_TransferEntropy(resid, x_data, y_data)
% Directional, nonlinear transfer entropy between the SINDy residual and
% each observed state, at lag 1. Complements xcorr_features (linear only).
%
% TE(A->B) = sum p(b_t+1, b_t, a_t) * log[ p(b_t+1|b_t,a_t) / p(b_t+1|b_t) ]
%
% Uses a simple equal-frequency (rank-based) binning estimator — robust at
% small N without requiring bandwidth tuning (KDE) or long series.
% N_BINS chosen conservatively (4) given N~300, per the small-sample
% guidance in Lee et al. 2012 (Biomed Eng Online) — TE is estimable down
% to N~50-200 with coarse binning; finer binning needs more data.
%
% Returns 4 numbers: TE(resid->x), TE(x->resid), TE(resid->y), TE(y->resid)

N_BINS = 4;

r = resid(:); x = x_data(:); y = y_data(:);
n = min([length(r), length(x), length(y)]);
r = r(1:n); x = x(1:n); y = y(1:n);

te_features = [
    te_lag1(r, x, N_BINS);   % resid -> x
    te_lag1(x, r, N_BINS);   % x -> resid
    te_lag1(r, y, N_BINS);   % resid -> y
    te_lag1(y, r, N_BINS)];  % y -> resid

te_features(~isfinite(te_features)) = 0;
te_features = max(min(te_features, 10), 0);  % TE is non-negative in theory

end

function te = te_lag1(A, B, n_bins)
% TE(A -> B) at lag 1: does A's past help predict B's future beyond B's own past?
n = length(A);
if n < 30; te = 0; return; end

a_t   = A(1:end-1);
b_t   = B(1:end-1);
b_tp1 = B(2:end);

ba = rank_bin(a_t, n_bins);
bb = rank_bin(b_t, n_bins);
bc = rank_bin(b_tp1, n_bins);

% Joint (b_tp1, b_t, a_t) histogram
joint_idx = sub2ind([n_bins,n_bins,n_bins], bc, bb, ba);
p_joint = accumarray(joint_idx(:), 1, [n_bins^3,1]) / length(bc);

% Marginal (b_t, a_t)
idx_ba = sub2ind([n_bins,n_bins], bb, ba);
p_ba = accumarray(idx_ba(:), 1, [n_bins^2,1]) / length(bb);

% Marginal (b_tp1, b_t)
idx_cb = sub2ind([n_bins,n_bins], bc, bb);
p_cb = accumarray(idx_cb(:), 1, [n_bins^2,1]) / length(bc);

% Marginal (b_t)
p_b = accumarray(bb(:), 1, [n_bins,1]) / length(bb);

te = 0;
for i = 1:n_bins        % b_tp1
    for j = 1:n_bins    % b_t
        for k = 1:n_bins  % a_t
            pj = p_joint(sub2ind([n_bins,n_bins,n_bins], i,j,k));
            if pj < 1e-12; continue; end
            p_ba_v = p_ba(sub2ind([n_bins,n_bins],j,k));
            p_cb_v = p_cb(sub2ind([n_bins,n_bins],i,j));
            p_b_v  = p_b(j);
            if p_ba_v < 1e-12 || p_b_v < 1e-12; continue; end
            num = pj / p_ba_v;
            den = p_cb_v / p_b_v;
            if num < 1e-12 || den < 1e-12; continue; end
            te = te + pj * log(num/den);
        end
    end
end
end

function bins = rank_bin(v, n_bins)
% Equal-frequency binning via rank (robust to outliers, no bandwidth param)
[~, order] = sort(v);
ranks = zeros(size(v));
ranks(order) = 1:length(v);
bins = ceil(ranks / length(v) * n_bins);
bins(bins < 1) = 1; bins(bins > n_bins) = n_bins;
end