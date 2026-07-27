clear; close all; clc;

% ── CONFIG ────────────────────────────────────────────────────────────────
DATA_ROOT  = 'C:/Users/nickj/LTInetV8 Local Data Drive';
N_BINS     = 8;
RR_TARGET  = 0.10;
MIN_LINE   = 2;
N_SCALES   = 6;
sg_p = 3; sg_f = 11; dt = 0.05;

hill_rep = @(z,k,n) k.^n ./ (k.^n + z.^n);
hill_deg = @(y,km)  y   ./ (km  + y);

folders = dir(DATA_ROOT);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name},{'.','..'}));

v7_folders = {};
for i = 1:length(folders)
    if contains(folders(i).name,'NNdata')
        v7_folders{end+1} = fullfile(DATA_ROOT, folders(i).name);
    end
end

fprintf('Found %d V7 folders\n', length(v7_folders));
total_processed = 0; total_skipped = 0;
t_overall_start = tic;

for fi = 1:length(v7_folders)
    folder_path = v7_folders{fi};
    mat_files   = dir(fullfile(folder_path,'*.mat'));
    fprintf('\n[%d/%d] %s — %d files\n', fi, length(v7_folders), folder_path, length(mat_files));
    t_folder_start = tic;

    for mf = 1:length(mat_files)
        fpath = fullfile(folder_path, mat_files(mf).name);
        try
            mat = load(fpath);
        catch
            total_skipped = total_skipped+1; continue;
        end

        if isfield(mat,'annex_v7_complete')
            total_processed = total_processed+1; continue;
        end

        if ~isfield(mat,'x_data_all') || ~isfield(mat,'y_data_all') || ~isfield(mat,'Xi')
            total_skipped = total_skipped+1; continue;
        end

        x_all = mat.x_data_all(:);
        y_all = mat.y_data_all(:);
        Xi    = mat.Xi;
        n_all = length(x_all);

        if n_all < 100
            total_skipped = total_skipped+1; continue;
        end

        if isfield(mat,'params_saved')
            p = mat.params_saved;
            hill_k0 = p.hill_k0; hill_km = p.hill_km; hill_n = p.hill_n;
        else
            hill_k0 = 2.0; hill_km = 0.8; hill_n = 3;
        end

        if isfield(mat,'libid')
            libid = mat.libid;
        elseif isfield(mat,'col_names')
            col_names = mat.col_names;
            if any(cellfun(@(c) strcmp(c,'x^2y'), col_names)) && any(cellfun(@(c) strcmp(c,'xy'), col_names))
                libid = 'rma';
            elseif any(cellfun(@(c) strcmp(c,'x^2y'), col_names))
                libid = 'brusselator';
            elseif any(cellfun(@(c) strcmp(c,'x^3'), col_names)) && any(cellfun(@(c) strcmp(c,'HillRep_y'), col_names))
                libid = 'toggle';
            elseif any(cellfun(@(c) strcmp(c,'x^3'), col_names)) && numel(col_names)==8
                libid = 'fhn';
            else
                libid = 'standard';
            end
        else
            total_skipped = total_skipped+1; continue;
        end

        XiX = Xi(:,1); XiY = Xi(:,2);

        %% ── SINDY RESIDUALS — VECTORIZED (was the actual bottleneck) ────
        % V6's annex script evaluated the library ONE TIMESTEP AT A TIME
        % in a `for k=2:n_all` loop — ~500 scalar function calls per file,
        % in interpreted MATLAB, across thousands of files. That loop, not
        % the RQA/wavelet blocks, was almost certainly the real slowdown
        % vs. generation (which never does per-timestep scalar library
        % evaluation). Fix: build the full library matrix in one
        % vectorized pass across the whole trajectory, then compute
        % dx_model/dy_model as a single matrix-vector product.
        x_lag = [0; x_all(1:end-1)];
        y_lag = [0; y_all(1:end-1)];

        Theta_full = v7_build_lib_vectorized(x_lag, y_lag, hill_k0, hill_km, hill_n, libid, hill_rep, hill_deg);

        if size(Theta_full,2) == size(Xi,1)
            dx_model = Theta_full * XiX;
            dy_model = Theta_full * XiY;
            dx_model(1) = 0; dy_model(1) = 0;
        else
            total_skipped = total_skipped+1; continue;
        end

        dx_obs = sgolayfilt(gradient(x_all,dt),sg_p,sg_f);
        dy_obs = sgolayfilt(gradient(y_all,dt),sg_p,sg_f);

        resid_dx = dx_obs - dx_model;
        resid_dy = dy_obs - dy_model;

        if ~all(isfinite(resid_dx)) || ~all(isfinite(resid_dy))
            total_skipped = total_skipped+1; continue;
        end

        %% ── INSTANTANEOUS PHASE AND AMPLITUDE (HILBERT) ──────────────────
        x_cent = x_all - mean(x_all);
        y_cent = y_all - mean(y_all);

        analytic_x = hilbert(x_cent);
        inst_phase  = mod(angle(analytic_x), 2*pi);
        inst_amp_x  = abs(analytic_x);

        analytic_y = hilbert(y_cent);
        inst_amp_y = abs(analytic_y);

        bin_edges = linspace(0, 2*pi, N_BINS+1);

        %% ── FEATURE 1: PHASE-RESOLVED X-RESIDUAL (16) ────────────────────
        phase_mean_dx = zeros(N_BINS,1);
        phase_std_dx  = zeros(N_BINS,1);
        r_std_x = std(resid_dx);
        if r_std_x < 1e-8; total_skipped=total_skipped+1; continue; end

        for b = 1:N_BINS
            mask = inst_phase >= bin_edges(b) & inst_phase < bin_edges(b+1);
            if sum(mask) >= 3
                phase_mean_dx(b) = mean(resid_dx(mask)) / r_std_x;
                phase_std_dx(b)  = std(resid_dx(mask))  / r_std_x;
            end
        end
        phase_x_features = max(min([phase_mean_dx; phase_std_dx], 10), -10);

        %% ── FEATURE 2: PHASE-RESOLVED Y-RESIDUAL (16) ────────────────────
        phase_mean_dy = zeros(N_BINS,1);
        phase_std_dy  = zeros(N_BINS,1);
        r_std_y = std(resid_dy);
        if r_std_y < 1e-8; r_std_y = 1.0; end

        for b = 1:N_BINS
            mask = inst_phase >= bin_edges(b) & inst_phase < bin_edges(b+1);
            if sum(mask) >= 3
                phase_mean_dy(b) = mean(resid_dy(mask)) / r_std_y;
                phase_std_dy(b)  = std(resid_dy(mask))  / r_std_y;
            end
        end
        phase_y_features = max(min([phase_mean_dy; phase_std_dy], 10), -10);

        %% ── FEATURE 3: INSTANTANEOUS AMPLITUDE ENVELOPE (8) ─────────────
        amp_mean_x = zeros(N_BINS,1);
        cycle_amp  = mean(inst_amp_x) + 1e-8;
        for b = 1:N_BINS
            mask = inst_phase >= bin_edges(b) & inst_phase < bin_edges(b+1);
            if sum(mask) >= 3
                amp_mean_x(b) = mean(inst_amp_x(mask)) / cycle_amp;
            end
        end
        amp_features = max(min(amp_mean_x, 5), 0);

        %% ── FEATURE 4: RQA ON X (7) AND Y (7) = 14 ──────────────────────
        rqa_x = compute_rqa(x_all, RR_TARGET, MIN_LINE);
        rqa_y = compute_rqa(y_all, RR_TARGET, MIN_LINE);

        if ~all(isfinite(rqa_x)) || ~all(isfinite(rqa_y))
            total_skipped = total_skipped+1; continue;
        end
        rqa_features = [rqa_x; rqa_y];

        %% ── FEATURE 5: CROSS-RQA BETWEEN X AND Y (7) ────────────────────
        crqa_features = compute_crqa(x_all, y_all, RR_TARGET, MIN_LINE);
        if ~all(isfinite(crqa_features))
            crqa_features = zeros(7,1);
        end

        %% ── FEATURE 6: WAVELET ENERGY PROFILE (12) ──────────────────────
        wavelet_features = compute_wavelet_energy(x_all, y_all, dt, N_SCALES);
        if ~all(isfinite(wavelet_features))
            wavelet_features = zeros(12,1);
        end

        %% ── FEATURE 7: POINCARE SECTION STATISTICS (6) ──────────────────
        poincare_features = compute_poincare(x_all, y_all);
        if ~all(isfinite(poincare_features))
            poincare_features = zeros(6,1);
        end

        %% ── FEATURE 8: SINDY COEFFICIENT MAGNITUDES (9) ─────────────────
        xi_x_raw = Xi(:,1);
        xi_norm  = norm(xi_x_raw,2);
        if xi_norm < 1e-10; xi_norm = 1.0; end
        xi_mag = abs(xi_x_raw) / xi_norm;
        if length(xi_mag) < 9
            xi_mag = [xi_mag; zeros(9-length(xi_mag),1)];
        else
            xi_mag = xi_mag(1:9);
        end
        xi_mag_features = max(min(xi_mag, 5), 0);

        %% ── ASSEMBLE ALL NEW FEATURES ────────────────────────────────────
        all_new_features = [phase_x_features; phase_y_features; ...
                            amp_features;     rqa_features;     ...
                            crqa_features;    wavelet_features; ...
                            poincare_features; xi_mag_features];

        if ~all(isfinite(all_new_features))
            total_skipped = total_skipped+1; continue;
        end

        rqa_phase_features = [phase_x_features; rqa_features];

        annex_v7_complete  = true;

        save(fpath, 'all_new_features', 'rqa_phase_features', ...
             'annex_v7_complete', '-append');

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


%% ── HELPER: VECTORIZED library matrix builder ─────────────────────────
% Mirrors V6_BuildLib.m's term structure, evaluated across the FULL
% x_lag/y_lag vectors at once using array operators instead of a
% per-timestep scalar loop.
function Theta = v7_build_lib_vectorized(x, y, hill_k0, hill_km, hill_n, libid, hill_rep, hill_deg)
    n = length(x);
    hr = hill_rep(y, hill_k0, hill_n);
    hd = hill_deg(y, hill_km);
    switch libid
        case 'brusselator'
            Theta = [ones(n,1), x, x.^2, y, x.^2.*y, hr, hd];
        case 'fhn'
            Theta = [ones(n,1), x, x.^3, y, hr, hd, x.*y, x.^2];
        case 'rma'
            Theta = [ones(n,1), x, x.^2, y, x.*y, x.^2.*y, hr, hd];
        case 'toggle'
            hr_x = hill_rep(x, hill_k0, hill_n);
            Theta = [ones(n,1), x, y, hr, hr_x, hd];
        otherwise
            Theta = [ones(n,1), x, x.^2, x.^3, y, y.^2, y.^3, hr, hd];
    end
end


%% ── HELPER FUNCTIONS (unchanged from V6_annex_full.m) ────────────────────

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