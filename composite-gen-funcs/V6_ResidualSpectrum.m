function spec_features = V6_ResidualSpectrum(resid, dt)
% Characterizes the STRidge residual as its OWN time series, unconditioned
% on x/y phase or lag alignment — fills the gap where a latent variable's
% own timescale (e.g. distinct degradation kinetics) would show up as
% periodicity in the residual that phase-binned or cross-correlated
% features would dilute or miss entirely.
%
% Returns 4 numbers:
%   [1] dominant frequency (Hz, from periodogram peak)
%   [2] peak power ratio (peak power / total power — how "tonal" vs broadband)
%   [3] spectral entropy (normalized, 0=pure tone, 1=white noise)
%   [4] autocorrelation decay time (samples to first drop below 1/e)

r = resid(:);
r = r - mean(r);
N = length(r);

if std(r) < 1e-10
    spec_features = [0;0;1;0];
    return;
end

% ── Periodogram (simple DFT-based, no toolbox dependency) ───────────────
Y = fft(r);
P = abs(Y(1:floor(N/2)+1)).^2 / N;
P(2:end-1) = 2*P(2:end-1);
freqs = (0:floor(N/2)) * (1/(N*dt));

P_norm = P / (sum(P) + 1e-12);
[peak_val, peak_idx] = max(P(2:end));  % skip DC (index 1)
peak_idx = peak_idx + 1;
dominant_freq = freqs(peak_idx);
peak_power_ratio = peak_val / (sum(P) + 1e-12);

% ── Spectral entropy (Shannon entropy of normalized power spectrum) ─────
p_valid = P_norm(P_norm > 1e-12);
raw_entropy = -sum(p_valid .* log(p_valid));
max_entropy = log(length(P_norm));
spectral_entropy = raw_entropy / (max_entropy + 1e-12);

% ── Autocorrelation decay time ───────────────────────────────────────────
maxlag = min(round(N/3), 100);
v0 = sum(r.^2);
ac = zeros(maxlag,1);
for L = 1:maxlag
    ac(L) = sum(r(1:end-L).*r(L+1:end)) / v0;
end
decay_idx = find(ac <= exp(-1), 1, 'first');
if isempty(decay_idx)
    decay_time = maxlag * dt;  % never decayed within window — cap at window
else
    decay_time = decay_idx * dt;
end

spec_features = [dominant_freq; peak_power_ratio; spectral_entropy; decay_time];
spec_features(~isfinite(spec_features)) = 0;
spec_features = max(min(spec_features, 100), 0);

end