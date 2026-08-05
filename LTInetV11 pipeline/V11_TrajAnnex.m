clear; close all; clc;

DATA_ROOT  = 'C:/Users/nickj/LTInetV11 Local Data Drive';
N_BINS     = 8;
RR_TARGET  = 0.10;
MIN_LINE   = 2;
N_SCALES   = 6;
dt = 0.05;

folders = dir(DATA_ROOT);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name},{'.','..'}));

v11_folders = {};
for i = 1:length(folders)
    if contains(folders(i).name,'NNdata')
        v11_folders{end+1} = fullfile(DATA_ROOT, folders(i).name);
    end
end

for fi = 1:length(v11_folders)
    folder_path = v11_folders{fi};
    mat_files   = dir(fullfile(folder_path,'*.mat'));

    for mf = 1:length(mat_files)
        fpath = fullfile(folder_path, mat_files(mf).name);
        try
            mat = load(fpath);
        catch
            continue;
        end

        if isfield(mat,'annex_v11_complete'); continue; end
        if ~isfield(mat,'x_data_all') || ~isfield(mat,'y_data_all'); continue; end

        x_all = mat.x_data_all(:);
        y_all = mat.y_data_all(:);
        n_all = length(x_all);
        if n_all < 100; continue; end

        x_cent = x_all - mean(x_all);
        analytic_x = hilbert(x_cent);
        inst_phase  = mod(angle(analytic_x), 2*pi);
        inst_amp_x  = abs(analytic_x);

        bin_edges = linspace(0, 2*pi, N_BINS+1);

        amp_mean_x = zeros(N_BINS,1);
        cycle_amp  = mean(inst_amp_x) + 1e-8;
        for b = 1:N_BINS
            mask = inst_phase >= bin_edges(b) & inst_phase < bin_edges(b+1);
            if sum(mask) >= 3
                amp_mean_x(b) = mean(inst_amp_x(mask)) / cycle_amp;
            end
        end
        amp_features = max(min(amp_mean_x, 5), 0);

        [DET_x,LAM_x,TT_x,ENTR_x,~,DIV_x] = compute_rqa(x_all, RR_TARGET, MIN_LINE);
        [DET_y,LAM_y,TT_y,ENTR_y,L_mean_y,DIV_y] = compute_rqa(y_all, RR_TARGET, MIN_LINE);
        rqa_x = [DET_x;LAM_x;TT_x;ENTR_x;DIV_x];
        rqa_y = [DET_y;LAM_y;TT_y;ENTR_y;L_mean_y;DIV_y];
        if ~all(isfinite(rqa_x)) || ~all(isfinite(rqa_y)); continue; end

        [CDET,CLAM,CTT,CENTR,CL_mean,CDIV] = compute_crqa(x_all, y_all, RR_TARGET, MIN_LINE);
        crqa = [CDET;CLAM;CTT;CENTR;CL_mean;CDIV];
        if ~all(isfinite(crqa)); crqa = zeros(6,1); end

        [energy_x, energy_y] = compute_wavelet_energy(x_all, y_all, dt, N_SCALES);
        if ~all(isfinite(energy_x)) || ~all(isfinite(energy_y))
            energy_x = zeros(6,1); energy_y = zeros(6,1);
        end
        wavelet_x = energy_x([1,2,4,6]);
        wavelet_y = energy_y([1,2,3,6]);

        [mean_y_norm,~,~,kurt,period_mean,period_cv] = compute_poincare(x_all, y_all);
        poincare = [mean_y_norm;kurt;period_mean;period_cv];
        if ~all(isfinite(poincare)); poincare = zeros(4,1); end

        all_new_trajectory_features = [amp_features; rqa_x; rqa_y; crqa; wavelet_x; wavelet_y; poincare];

        if ~all(isfinite(all_new_trajectory_features)) || numel(all_new_trajectory_features) ~= 37
            continue;
        end

        annex_v11_complete = true;
        save(fpath, 'all_new_trajectory_features', 'annex_v11_complete', '-append');
    end
end


function [DET,LAM,TT,ENTR,L_mean,DIV] = compute_rqa(x, rr_target, min_line)
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

    [DET, L_mean, L_max, ENTR] = diag_stats(R, N, min_line);
    DIV = 1 / max(L_max, 1);
    [LAM, TT] = vert_stats(R, N, min_line);

    DET=min(max(DET,0),100); LAM=min(max(LAM,0),100); TT=min(max(TT,0),50);
    ENTR=min(max(ENTR,0),10); L_mean=min(max(L_mean,0),50); DIV=min(max(DIV,0),100);
end


function [CDET,CLAM,CTT,CENTR,CL_mean,CDIV] = compute_crqa(x, y, rr_target, min_line)
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

    [CDET, CL_mean, CL_max, CENTR] = diag_stats(CR, N, min_line);
    CDIV = 1 / max(CL_max, 1);
    [CLAM, CTT] = vert_stats(CR, N, min_line);

    CDET=min(max(CDET,0),100); CLAM=min(max(CLAM,0),100); CTT=min(max(CTT,0),50);
    CENTR=min(max(CENTR,0),10); CL_mean=min(max(CL_mean,0),50); CDIV=min(max(CDIV,0),100);
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


function [energy_x, energy_y] = compute_wavelet_energy(x, y, dt, n_scales)
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
    energy_x = min(max(energy_x/total_x, 0), 5);
    energy_y = min(max(energy_y/total_y, 0), 5);
end


function [mean_y_norm,std_y_norm,skew,kurt,period_mean,period_cv] = compute_poincare(x, y)
    x_cent = x - mean(x);
    crossings = find(x_cent(1:end-1) < 0 & x_cent(2:end) >= 0);
    if length(crossings) < 4
        mean_y_norm=0;std_y_norm=0;skew=0;kurt=0;period_mean=0;period_cv=0; return;
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

    mu    = mean(y_norm);
    sigma = std(y_norm);
    if sigma < 1e-10
        skew = 0; kurt = 0;
    else
        skew = mean(((y_norm - mu)/sigma).^3);
        kurt = mean(((y_norm - mu)/sigma).^4);
    end

    mean_y_norm=max(min(mean(y_norm),10),-10);
    std_y_norm=max(min(std(y_norm),10),-10);
    skew=max(min(skew,10),-10);
    kurt=max(min(kurt,10),-10);
    period_mean=max(min(period_mean,10),-10);
    period_cv=max(min(period_cv,10),-10);
end