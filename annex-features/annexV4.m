clear; close all; clc;

% ── CONFIG ────────────────────────────────────────────────────────────────
DATA_ROOT  = 'C:/Users/nickj/MATLAB Drive/LTI Network';
N_BINS     = 8;     % phase bins for phase-resolved residual
RR_TARGET  = 0.10;  % target recurrence rate for epsilon selection
MIN_LINE   = 2;     % minimum diagonal/vertical line length for RQA

sg_p = 3; sg_f = 11; dt = 0.05;

hill_rep = @(z,k,n) k^n ./ (k^n + z.^n);
hill_deg = @(y,km)  y   ./ (km  + y);

% Find all V4 NNdata folders
folders = dir(DATA_ROOT);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name},{'.','..'}));

v4_folders = {};
for i=1:length(folders)
    if contains(folders(i).name,'NNdata')
        v4_folders{end+1} = fullfile(DATA_ROOT,folders(i).name);
    end
end

fprintf('Found %d V4 folders\n',length(v4_folders));
total_processed = 0; total_skipped = 0;

for fi = 1:length(v4_folders)
    folder_path = v4_folders{fi};
    mat_files   = dir(fullfile(folder_path,'*.mat'));
    fprintf('\n[%d/%d] %s — %d files\n', ...
        fi,length(v4_folders),folder_path,length(mat_files));

    for mf = 1:length(mat_files)
        fpath = fullfile(folder_path,mat_files(mf).name);
        try
            mat = load(fpath);
        catch
            total_skipped=total_skipped+1; continue;
        end

        % Skip if already processed
        if isfield(mat,'rqa_phase_features')
            total_processed=total_processed+1; continue;
        end

        % Require saved observables and SINDy output
        if ~isfield(mat,'x_data_all')||~isfield(mat,'y_data_all')||~isfield(mat,'Xi')
            total_skipped=total_skipped+1; continue;
        end

        x_all = mat.x_data_all(:);
        y_all = mat.y_data_all(:);
        Xi    = mat.Xi;
        n_all = length(x_all);

        if n_all < 100
            total_skipped=total_skipped+1; continue;
        end

        % Recover hill parameters from params_saved for residual computation
        if isfield(mat,'params_saved')
            p = mat.params_saved;
            hill_k0 = p.hill_k0;
            hill_km = p.hill_km;
            hill_n  = p.hill_n;
        else
            hill_k0 = 2.0; hill_km = 0.8; hill_n = 3;
        end

        % Identify library type from col_names
        col_names = mat.col_names;
        is_brussel = length(col_names)==7 && ...
                     any(cellfun(@(c) contains(c,'x^2y')||contains(c,'xy'),col_names));

        %% ── PART 1: PHASE-RESOLVED RESIDUAL STATISTICS ───────────────────
        % Recompute SINDy model prediction for dx/dt using saved Xi
        dx_obs   = sgolayfilt(gradient(x_all,dt),sg_p,sg_f);
        dx_model = zeros(n_all,1);
        XiX = Xi(:,1);

        for k = 2:n_all
            xk = x_all(k-1); yk = y_all(k-1);
            hr = hill_rep(yk, hill_k0, hill_n);
            hd = hill_deg(yk, hill_km);
            if is_brussel
                phi = [1, xk, xk^2, yk, xk*yk, hr, hd];
            else
                phi = [1, xk, xk^2, xk^3, yk, yk^2, yk^3, hr, hd];
            end
            if length(phi)==length(XiX)
                dx_model(k) = phi*XiX;
            end
        end

        resid_dx = dx_obs - dx_model;
        if ~all(isfinite(resid_dx))
            total_skipped=total_skipped+1; continue;
        end

        % Instantaneous phase of x via Hilbert transform
        x_centered = x_all - mean(x_all);
        analytic_x = hilbert(x_centered);
        inst_phase  = angle(analytic_x);   % in [-pi, pi]
        inst_phase  = mod(inst_phase, 2*pi); % map to [0, 2pi]

        % Phase-resolved residual statistics: N_BINS bins
        bin_edges = linspace(0, 2*pi, N_BINS+1);
        phase_mean = zeros(N_BINS,1);
        phase_std  = zeros(N_BINS,1);

        for b = 1:N_BINS
            mask = inst_phase >= bin_edges(b) & inst_phase < bin_edges(b+1);
            if sum(mask) >= 3
                phase_mean(b) = mean(resid_dx(mask));
                phase_std(b)  = std(resid_dx(mask));
            end
        end

        % Normalise by overall residual std to make scale-invariant
        r_std = std(resid_dx);
        if r_std < 1e-8; total_skipped=total_skipped+1; continue; end
        phase_mean = phase_mean / r_std;
        phase_std  = phase_std  / r_std;

        % Clip to prevent extreme values
        phase_mean = max(min(phase_mean, 10), -10);
        phase_std  = max(min(phase_std,  10),  0);

        phase_features = [phase_mean; phase_std];  % 16-dim

        if ~all(isfinite(phase_features))
            total_skipped=total_skipped+1; continue;
        end

        %% ── PART 2: RQA FEATURES ─────────────────────────────────────────
        % Compute RQA for x(t) and y(t) independently
        % Epsilon chosen to achieve RR_TARGET recurrence rate

        rqa_x   = compute_rqa(x_all, RR_TARGET, MIN_LINE);
        rqa_y   = compute_rqa(y_all, RR_TARGET, MIN_LINE);

        if ~all(isfinite(rqa_x)) || ~all(isfinite(rqa_y))
            total_skipped=total_skipped+1; continue;
        end

        % rqa_x and rqa_y are each 7-dim:
        % [RR, DET, LAM, TT, ENTR, L_mean, DIV]
        % Total RQA: 14-dim

        rqa_features = [rqa_x; rqa_y];

        %% ── COMBINED NEW FEATURES ────────────────────────────────────────
        % Total: 16 (phase) + 14 (RQA) = 30-dim
        rqa_phase_features = [phase_features; rqa_features];

        if ~all(isfinite(rqa_phase_features))
            total_skipped=total_skipped+1; continue;
        end

        % Append to existing file without rewriting other fields
        save(fpath, 'rqa_phase_features', '-append');
        total_processed = total_processed+1;

        if mod(mf,50)==0
            fprintf('  %d/%d processed\n',mf,length(mat_files));
        end
    end
end

fprintf('\n=== COMPLETE ===\n');
fprintf('Processed: %d\nSkipped:   %d\n',total_processed,total_skipped);


%% ── RQA COMPUTATION ──────────────────────────────────────────────────────
% Computes 7 RQA scalars from a univariate time series.
% Uses fixed-threshold recurrence matrix targeting RR_TARGET recurrence rate.
% No external toolbox required.
%
% Output: [RR, DET, LAM, TT, ENTR, L_mean, DIV]
%   RR:     recurrence rate — density of recurrent points
%   DET:    determinism — fraction of recurrent points on diagonal lines
%   LAM:    laminarity — fraction of recurrent points on vertical lines
%   TT:     trapping time — mean vertical line length (state dwell time)
%   ENTR:   Shannon entropy of diagonal line lengths
%   L_mean: mean diagonal line length
%   DIV:    divergence = 1/L_max (inverse of longest diagonal)

function rqa = compute_rqa(x, rr_target, min_line)
    N = length(x);
    x = x(:);

    % Downsample if very long to keep computation tractable
    if N > 400
        step = floor(N/300);
        x = x(1:step:end);
        N = length(x);
    end

    % Pairwise distance matrix (Euclidean, 1D)
    D = abs(bsxfun(@minus, x, x'));

    % Set epsilon to achieve target recurrence rate
    d_vals = D(triu(true(N),1));
    epsilon = quantile(d_vals, rr_target);
    if epsilon < 1e-10; epsilon = 1e-10; end

    % Binary recurrence matrix (exclude main diagonal)
    R = double(D <= epsilon);
    R(1:N+1:end) = 0;   % zero diagonal

    actual_RR = sum(R(:)) / (N*(N-1));

    %% Diagonal line statistics (determinism, entropy, L_mean, L_max)
    DET_count = 0; total_diag = sum(R(:));
    line_lengths = [];

    for d_offset = -(N-min_line):(N-min_line)
        diag_vec = diag(R, d_offset);
        if length(diag_vec) < min_line; continue; end
        % Find runs of 1s
        runs = find_runs(diag_vec, min_line);
        if ~isempty(runs)
            line_lengths = [line_lengths, runs];
            DET_count = DET_count + sum(runs);
        end
    end

    if total_diag > 0 && ~isempty(line_lengths)
        DET    = DET_count / total_diag;
        L_mean = mean(line_lengths);
        L_max  = max(line_lengths);
        DIV    = 1 / L_max;
        % Shannon entropy of line length distribution
        u_lens = unique(line_lengths);
        p_lens = histc(line_lengths, u_lens) / length(line_lengths);
        p_lens = p_lens(p_lens>0);
        ENTR   = -sum(p_lens .* log(p_lens+1e-12));
    else
        DET=0; L_mean=0; DIV=1; ENTR=0;
    end

    %% Vertical line statistics (laminarity, trapping time)
    LAM_count = 0;
    vert_lengths = [];

    for col = 1:N
        col_vec = R(:,col);
        runs = find_runs(col_vec, min_line);
        if ~isempty(runs)
            vert_lengths = [vert_lengths, runs];
            LAM_count = LAM_count + sum(runs);
        end
    end

    if total_diag > 0 && ~isempty(vert_lengths)
        LAM = LAM_count / total_diag;
        TT  = mean(vert_lengths);
    else
        LAM=0; TT=0;
    end

    % Clip to reasonable range
    rqa = [actual_RR; DET; LAM; min(TT,50); min(ENTR,10); min(L_mean,50); DIV];
    rqa = max(min(rqa, 100), 0);
end


function runs = find_runs(vec, min_len)
    % Returns lengths of consecutive runs of 1s with length >= min_len
    runs = [];
    n = length(vec);
    i = 1;
    while i <= n
        if vec(i) == 1
            j = i;
            while j <= n && vec(j) == 1
                j = j+1;
            end
            run_len = j - i;
            if run_len >= min_len
                runs(end+1) = run_len;
            end
            i = j;
        else
            i = i+1;
        end
    end
end