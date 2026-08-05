clear; close all; clc;

% ================================================================
% V11_TrajAnnex.m
%
% Trimmed from the V10 trajectory-only annex script per the V11 feature
% cut. The underlying RQA/cross-RQA/wavelet/Poincare computations are
% UNCHANGED (each stat is produced by a shared pass — e.g. diag_stats/
% vert_stats compute DET/L_mean/L_max/ENTR/LAM/TT together, and the
% wavelet loop computes all scales for x and y together — so hand-
% removing individual dropped stats would not reduce compute, only add
% index-mapping risk). What changed is the OUTPUT ASSEMBLY: each raw
% block is sliced to its surviving V11 indices before being written to
% all_new_trajectory_features, which is now 37-dim instead of 47.
%
%   amp        : [1:8]         all 8 survive
%   rqa_x      : [2,3,4,5,7]   5 of 7  (cut actual_RR[1], L_mean[6])
%   rqa_y      : [2,3,4,5,6,7] 6 of 7  (cut actual_RR[1] only)
%   crqa       : [2,3,4,5,6,7] 6 of 7  (cut CRR[1] only)
%   wavelet_x  : [1,2,4,6]     4 of 6  (cut scales [3,5])
%   wavelet_y  : [1,2,3,6]     4 of 6  (cut scales [4,5])
%   poincare   : [1,4,5,6]     4 of 6  (cut std[2], skew[3])
%   -> 8+5+6+6+4+4+4 = 37
%
% fp_multi_features (the remaining 3 of the b1c=40 total) is produced at
% generation time by V11_ProcessFeatures.m / V11_MultiFixedPoint.m, NOT
% here — this script only ever touched the trajectory-only block.
%
% Per-stat index order within compute_rqa/compute_crqa is unchanged:
%   [actual_RR; DET; LAM; TT; ENTR; L_mean; DIV]  (1-indexed here)
% ================================================================
DATA_ROOT  = 'C:/Users/nickj/LTInetV11 Local Data Drive';
N_BINS     = 8;
RR_TARGET  = 0.10;
MIN_LINE   = 2;
N_SCALES   = 6;
dt = 0.05;

KEEP_RQA_X = [2,3,4,5,7];
KEEP_RQA_Y = [2,3,4,5,6,7];
KEEP_CRQA  = [2,3,4,5,6,7];
KEEP_WAV_X = [1,2,4,6];
KEEP_WAV_Y = [1,2,3,6];
KEEP_POI   = [1,4,5,6];

folders = dir(DATA_ROOT);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name},{'.','..'}));

v11_folders = {};
for i = 1:length(folders)
    if contains(folders(i).name,'NNdata')
        v11_folders{end+1} = fullfile(DATA_ROOT, folders(i).name);
    end
end

fprintf('Found %d V11 folders\n', length(v11_folders));
total_processed = 0; total_skipped = 0;
t_overall_start = tic;

for fi = 1:length(v11_folders)
    folder_path = v11_folders{fi};
    mat_files   = dir(fullfile(folder_path,'*.mat'));
    fprintf('\n[%d/%d] %s — %d files\n', fi, length(v11_folders), folder_path, length(mat_files));
    t_folder_start = tic;

    for mf = 1:length(mat_files)
        fpath = fullfile(folder_path, mat_files(mf).name);
        try
            mat = load(fpath);
        catch
            total_skipped = total_skipped+1; continue;
        end

        if isfield(mat,'annex_v11_complete')
            total_processed = total_processed+1; continue;
        end

        if ~isfield(mat,'x_data_all') || ~isfield(mat,'y_data_all')
            total_skipped = total_skipped+1; continue;
        end

        x_all = mat.x_data_all(:);
        y_all = mat.y_data_all(:);
        n_all = length(x_all);

        if n_all < 100
            total_skipped = total_skipped+1; continue;
        end

        %% ── INSTANTANEOUS PHASE (HILBERT, X-CHANNEL) ────────────────────
        x_cent = x_all - mean(x_all);
        analytic_x = hilbert(x_cent);
        inst_phase  = mod(angle(analytic_x), 2*pi);
        inst_amp_x  = abs(analytic_x);

        bin_edges = linspace(0, 2*pi, N_BINS+1);

        %% ── FEATURE: INSTANTANEOUS AMPLITUDE ENVELOPE (8, all kept) ─────
        amp_mean_x = zeros(N_BINS,1);
        cycle_amp  = mean(inst_amp_x) + 1e-8;
        for b = 1:N_BINS
            mask = inst_phase >= bin_edges(b) & inst_phase < bin_edges(b+1);
            if sum(mask) >= 3
                amp_mean_x(b) = mean(inst_amp_x(mask)) / cycle_amp;
            end
        end
        amp_features = max(min(amp_mean_x, 5), 0);

        %% ── RQA ON X AND Y (full 7 each, sliced after) ──────────────────
        rqa_x_full = compute_rqa(x_all, RR_TARGET, MIN_LINE);
        rqa_y_full = compute_rqa(y_all, RR_TARGET, MIN_LINE);

        if ~all(isfinite(rqa_x_full)) || ~all(isfinite(rqa_y_full))
            total_skipped = total_skipped+1; continue;
        end

        %% ── CROSS-RQA BETWEEN X AND Y (full 7, sliced after) ────────────
        crqa_full = compute_crqa(x_all, y_all, RR_TARGET, MIN_LINE);
        if ~all(isfinite(crqa_full))
            crqa_full = zeros(7,1);
        end

        %% ── WAVELET ENERGY PROFILE (full 12 = 6x+6y, sliced after) ──────
        wavelet_full = compute_wavelet_energy(x_all, y_all, dt, N_SCALES);
        if ~all(isfinite(wavelet_full))
            wavelet_full = zeros(12,1);
        end
        wavelet_x_full = wavelet_full(1:6);
        wavelet_y_full = wavelet_full(7:12);

        %% ── POINCARE SECTION STATISTICS (full 6, sliced after) ──────────
        poincare_full = compute_poincare(x_all, y_all);
        if ~all(isfinite(poincare_full))
            poincare_full = zeros(6,1);
        end

        %% ── ASSEMBLE V11 TRAJECTORY-ONLY FEATURES (37 total) ────────────
        all_new_trajectory_features = [amp_features; ...
                                        rqa_x_full(KEEP_RQA_X); ...
                                        rqa_y_full(KEEP_RQA_Y); ...
                                        crqa_full(KEEP_CRQA); ...
                                        wavelet_x_full(KEEP_WAV_X); ...
                                        wavelet_y_full(KEEP_WAV_Y); ...
                                        poincare_full(KEEP_POI)];

        if ~all(isfinite(all_new_trajectory_features)) || numel(all_new_trajectory_features) ~= 37
            total_skipped = total_skipped+1; continue;
        end

        annex_v11_complete = true;

        save(fpath, 'all_new_trajectory_features', 'annex_v11_complete', '-append');

        total_processed = total_processed+1;
        if mod(mf, 50) == 0
            elapsed = toc(t_folder_start);
            rate = mf / elapsed;
            fprintf('  %d/%d processed (%.1f files/sec)\n', mf, length(mat_files), rate);
        end
    end
    fprintf('  Folder done in %.1fs\n', toc(t_folder_start));
end

fprintf('\n=== COMPLETE ===\n');
fprintf('Processed: %d\nSkipped:   %d\n', total_processed, total_skipped);
fprintf('Total time: %.1f min\n', toc(t_overall_start)/60);
fprintf('\nNext step: run V11_DEN_Annex.py to add the DEN error-signal block.\n');


%% ── HELPER FUNCTIONS (unchanged — shared, not itself a feature) ──────────

function rqa = compute_rqa(x, rr_target, min_line)
    N = length(x); x = x(:);
    if N > 400
        step = floor(N/300); x = x(1:step:end); N = length(x);
    end
    D       = abs(bsxfun(@minus, x, x'));
    d_vals  = D(triu(true(N),1));
    epsilon = quantile(d_vals, rr_target);
    if epsilon < 1e-10; epsilon = 1e-10; end
    R = double(D <= epsilon);
    R(1:N+1:end) = 0;
    actual_RR = sum(R(:)) / (N*(N-1));

    [DET, L_mean, L_max, ENTR] = diag_stats(R, N, min_line);
    DIV = 1 / max(L_max, 1);
    [LAM, TT] = vert_stats(R, N, min_line);

    rqa = max(min([actual_RR; DET; LAM; min(TT,50); ...
                   min(ENTR,10); min(L_mean,50); DIV], 100), 0);
end


function crqa = compute_crqa(x, y, rr_target, min_line)
    N = min(length(x), length(y));
    x = x(1:N); y = y(1:N);
    if N > 400
        step = floor(N/300);
        x = x(1:step:end); y = y(1:step:end); N = length(x);
    end
    x = (x - mean(x)) / (std(x)+1e-8);
    y = (y - mean(y)) / (std(y)+1e-8);

    D       = abs(bsxfun(@minus, x, y'));
    d_vals  = D(:);
    epsilon = quantile(d_vals, rr_target);
    if epsilon < 1e-10; epsilon = 1e-10; end
    CR = double(D <= epsilon);

    CRR = sum(CR(:)) / numel(CR);
    [CDET, CL_mean, CL_max, CENTR] = diag_stats(CR, N, min_line);
    CDIV = 1 / max(CL_max, 1);
    [CLAM, CTT] = vert_stats(CR, N, min_line);

    crqa = max(min([CRR; CDET; CLAM; min(CTT,50); ...
                    min(CENTR,10); min(CL_mean,50); CDIV], 100), 0);
end


function [DET, L_mean, L_max, ENTR] = diag_stats(R, N, min_line)
    total_pts = sum(R(:));
    DET_count = 0; line_lengths = [];
    for d_offset = -(N-min_line):(N-min_line)
        dv = diag(R, d_offset);
        if length(dv) < min_line; continue; end
        runs = find_runs(dv, min_line);
        if ~isempty(runs)
            line_lengths = [line_lengths, runs];
            DET_count = DET_count + sum(runs);
        end
    end
    if total_pts > 0 && ~isempty(line_lengths)
        DET    = DET_count / total_pts;
        L_mean = mean(line_lengths);
        L_max  = max(line_lengths);
        u      = unique(line_lengths);
        p      = histc(line_lengths,u)/length(line_lengths);
        p      = p(p>0);
        ENTR   = -sum(p.*log(p+1e-12));
    else
        DET=0; L_mean=0; L_max=1; ENTR=0;
    end
end


function [LAM, TT] = vert_stats(R, N, min_line)
    total_pts = sum(R(:));
    LAM_count = 0; vert_lengths = [];
    for col = 1:N
        runs = find_runs(R(:,col), min_line);
        if ~isempty(runs)
            vert_lengths = [vert_lengths, runs];
            LAM_count = LAM_count + sum(runs);
        end
    end
    if total_pts > 0 && ~isempty(vert_lengths)
        LAM = LAM_count / total_pts;
        TT  = mean(vert_lengths);
    else
        LAM=0; TT=0;
    end
end


function runs = find_runs(vec, min_len)
    runs = []; i = 1; n = length(vec);
    while i <= n
        if vec(i) == 1
            j = i;
            while j <= n && vec(j) == 1; j=j+1; end
            if (j-i) >= min_len; runs(end+1) = j-i; end
            i = j;
        else; i=i+1; end
    end
end


function wf = compute_wavelet_energy(x, y, dt, n_scales)
    x_cent = x - mean(x);
    y_cent = y - mean(y);
    N = length(x_cent);

    zc = sum(abs(diff(sign(x_cent))) > 0);
    T_est = max(2*N*dt / max(zc,1), 2*dt);

    scales = logspace(log10(T_est/2), log10(2*T_est), n_scales);

    energy_x = zeros(n_scales,1);
    energy_y = zeros(n_scales,1);
    t_vec    = (0:N-1)*dt;

    for s = 1:n_scales
        sigma = scales(s) / (2*pi);
        for k = 1:N
            psi = exp(-(t_vec - t_vec(k)).^2 / (2*sigma^2)) .* ...
                  cos(2*pi*(t_vec - t_vec(k)) / scales(s));
            psi = psi / (sum(psi.^2)^0.5 + 1e-12);
            energy_x(s) = energy_x(s) + (dot(x_cent, psi))^2;
            energy_y(s) = energy_y(s) + (dot(y_cent, psi))^2;
        end
        energy_x(s) = energy_x(s) / N;
        energy_y(s) = energy_y(s) / N;
    end

    total_x = sum(energy_x) + 1e-12;
    total_y = sum(energy_y) + 1e-12;
    wf = max(min([energy_x/total_x; energy_y/total_y], 5), 0);
end


function pf = compute_poincare(x, y)
    x_cent = x - mean(x);
    crossings = find(x_cent(1:end-1) < 0 & x_cent(2:end) >= 0);

    if length(crossings) < 4
        pf = zeros(6,1); return;
    end

    y_cross = zeros(length(crossings),1);
    for c = 1:length(crossings)
        idx  = crossings(c);
        frac = -x_cent(idx) / (x_cent(idx+1) - x_cent(idx) + 1e-12);
        y_cross(c) = y(idx) + frac*(y(idx+1) - y(idx));
    end

    y_mean = mean(y_cross);
    if abs(y_mean) < 1e-8; y_mean = 1.0; end
    y_norm = y_cross / y_mean;

    if length(crossings) > 1
        periods     = diff(crossings) * 0.05;
        period_mean = mean(periods);
        period_cv   = std(periods) / (period_mean + 1e-8);
    else
        period_mean = 0; period_cv = 0;
    end

    n     = length(y_norm);
    mu    = mean(y_norm);
    sigma = std(y_norm);
    if sigma < 1e-10
        skew = 0; kurt = 0;
    else
        skew = mean(((y_norm - mu)/sigma).^3);
        kurt = mean(((y_norm - mu)/sigma).^4);
    end

    pf = max(min([mean(y_norm); std(y_norm); skew; kurt; ...
                  period_mean; period_cv], 10), -10);
end
