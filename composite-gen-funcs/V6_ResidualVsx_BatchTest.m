clear; close all; clc;

% ── CONFIG ────────────────────────────────────────────────────────────────
DATA_ROOT = 'C:/Users/nickj/LTInetV6 Local Data Drive';
dt = 0.05; sg_p = 3; sg_f = 11;

hill_rep = @(z,k,n) k.^n ./ (k.^n + z.^n);
hill_deg = @(y,km)  y   ./ (km  + y);

folders = dir(DATA_ROOT);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name},{'.','..'}));

v6_folders = {};
for i = 1:length(folders)
    if contains(folders(i).name,'NNdata')
        v6_folders{end+1} = fullfile(DATA_ROOT, folders(i).name);
    end
end

fprintf('Found %d folders. Computing V6_ResidualVsX for C0/C2/C4 files...\n', length(v6_folders));

all_feat = [];
all_label = [];
all_sys = {};
n_processed = 0; n_skipped = 0;

skip_reasons = struct('load_fail',0,'no_structure_label',0,'wrong_class',0, ...
    'missing_fields',0,'no_params_saved',0,'library_eval_fail',0, ...
    'nonfinite_resid',0,'feat_computation_fail',0);
printed_examples = struct('missing_fields',false,'no_params_saved',false,'library_eval_fail',false);

for fi = 1:length(v6_folders)
    folder_path = v6_folders{fi};
    mat_files = dir(fullfile(folder_path,'*.mat'));

    for mf = 1:length(mat_files)
        fpath = fullfile(folder_path, mat_files(mf).name);
        try
            mat = load(fpath);
        catch
            skip_reasons.load_fail = skip_reasons.load_fail + 1;
            n_skipped = n_skipped + 1; continue;
        end

        if ~isfield(mat,'structure_label')
            skip_reasons.no_structure_label = skip_reasons.no_structure_label + 1;
            n_skipped = n_skipped + 1; continue;
        end
        sl = mat.structure_label;
        if ~ismember(sl, [0, 2, 4])
            skip_reasons.wrong_class = skip_reasons.wrong_class + 1;
            continue;
        end

        if ~isfield(mat,'x_data_all') || ~isfield(mat,'y_data_all') || ...
           ~isfield(mat,'Xi') || ~isfield(mat,'libid')
            skip_reasons.missing_fields = skip_reasons.missing_fields + 1;
            if ~printed_examples.missing_fields
                fprintf('  [missing_fields example] fields present: %s\n', strjoin(fieldnames(mat),', '));
                printed_examples.missing_fields = true;
            end
            n_skipped = n_skipped + 1; continue;
        end

        x_all = mat.x_data_all(:);
        y_all = mat.y_data_all(:);
        Xi    = mat.Xi;
        libid = mat.libid;
        n_all = length(x_all);

        if isfield(mat,'params_saved')
            p = mat.params_saved;
        else
            skip_reasons.no_params_saved = skip_reasons.no_params_saved + 1;
            if ~printed_examples.no_params_saved
                fprintf('  [no_params_saved example] file: %s\n', fpath);
                printed_examples.no_params_saved = true;
            end
            n_skipped = n_skipped + 1; continue;
        end

        dx_obs = sgolayfilt(gradient(x_all,dt),sg_p,sg_f);
        dy_obs = sgolayfilt(gradient(y_all,dt),sg_p,sg_f);
        dx_model = zeros(n_all,1); dy_model = zeros(n_all,1);

        try
            for k = 2:n_all
                row = V6_LibRow(x_all(k-1), y_all(k-1), p, libid);
                if length(row) == size(Xi,1)
                    dx_model(k) = row * Xi(:,1);
                    dy_model(k) = row * Xi(:,2);
                end
            end
        catch me
            skip_reasons.library_eval_fail = skip_reasons.library_eval_fail + 1;
            if ~printed_examples.library_eval_fail
                fprintf('  [library_eval_fail example] libid=%s error: %s\n', libid, me.message);
                printed_examples.library_eval_fail = true;
            end
            n_skipped = n_skipped + 1; continue;
        end

        resid_dx = dx_obs - dx_model;
        resid_dy = dy_obs - dy_model;

        if ~all(isfinite(resid_dx)) || ~all(isfinite(resid_dy))
            skip_reasons.nonfinite_resid = skip_reasons.nonfinite_resid + 1;
            n_skipped = n_skipped + 1; continue;
        end

        feat = V6_ResidualVsX(x_all, resid_dx, resid_dy, y_all);
        if ~all(isfinite(feat)) || numel(feat) ~= 16
            skip_reasons.feat_computation_fail = skip_reasons.feat_computation_fail + 1;
            n_skipped = n_skipped + 1; continue;
        end

        sys_name = 'unknown';
        if isfield(mat,'system_name')
            sys_name = mat.system_name;
        end

        all_feat = [all_feat; feat'];
        all_label = [all_label; sl];
        all_sys{end+1} = sys_name;

        n_processed = n_processed + 1;
        if mod(n_processed, 200) == 0
            fprintf('  processed %d...\n', n_processed);
        end
    end
end

fprintf('\nProcessed %d files (%d skipped)\n', n_processed, n_skipped);
fprintf('Class breakdown: C0=%d  C2=%d  C4=%d\n', ...
    sum(all_label==0), sum(all_label==2), sum(all_label==4));

fprintf('\n--- Skip reason breakdown ---\n');
fns = fieldnames(skip_reasons);
for i = 1:numel(fns)
    fprintf('  %-24s %d\n', fns{i}, skip_reasons.(fns{i}));
end

save(fullfile(DATA_ROOT, 'V6_ResidualVsX_test_data.mat'), ...
    'all_feat', 'all_label', 'all_sys');

fprintf('\nSaved to %s\n', fullfile(DATA_ROOT, 'V6_ResidualVsX_test_data.mat'));
fprintf('Run V6_ResidualVsX_LDA_test.py next to check separability.\n');