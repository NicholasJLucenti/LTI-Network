clear; close all; clc;
addpath(genpath('.'));
warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');

S = V6_SystemLib();
sysdef = S(7);  % toggle_switch
assert(strcmp(sysdef.name,'toggle_switch'));

dt = 0.05; N = 300;
t_end = (N+200)*dt; t_span = 0:dt:t_end;
opts_ode = odeset('RelTol',1e-4,'AbsTol',1e-6,'MaxStep',dt/2,'NonNegative',[1,2]);

n_attempts = 100;
rng(20260 + 7*10000 + 9999);

counts = struct('ode_fail',0,'nonfinite_len',0,'range',0,'flatline',0,'std_check',0,'passed',0);
sample_traj = {};

for a = 1:n_attempts
    p = sysdef.psamp();

    if rand() < 0.5
        ic2 = [p.alpha1*0.8; 0.5];
    else
        ic2 = [0.5; p.alpha2*0.8];
    end

    forcing_amp = 0.04;
    forcing_freq1 = 0.3 + 0.4*rand();
    forcing_freq2 = 0.5 + 0.6*rand();
    odefn_forced = @(t,s,p) sysdef.odefn_base(t,s,p) + ...
        forcing_amp * [s(1)*sin(forcing_freq1*t); s(2)*cos(forcing_freq2*t)];

    try
        [t_ode,S2] = ode15s(@(t,s) odefn_forced(t,s,p), t_span, ic2, opts_ode);
    catch me
        counts.ode_fail = counts.ode_fail+1;
        if counts.ode_fail <= 3
            fprintf('  [ode_fail #%d] %s\n', counts.ode_fail, me.message);
        end
        continue;
    end

    if any(~isfinite(S2(:))) || size(S2,1) < N+201
        counts.nonfinite_len = counts.nonfinite_len+1;
        if counts.nonfinite_len <= 3
            fprintf('  [nonfinite_len #%d] size=%d, finite=%d\n', ...
                counts.nonfinite_len, size(S2,1), all(isfinite(S2(:))));
        end
        continue;
    end

    t_uni = (0:dt:t_end)';
    xa = interp1(t_ode,S2(:,1),t_uni,'linear');
    ya = interp1(t_ode,S2(:,2),t_uni,'linear');
    n_all = length(t_uni);

    if any(~isfinite(xa))||any(~isfinite(ya))||max(abs(xa))>1e5||max(abs(ya))>1e5
        counts.range = counts.range+1;
        continue;
    end

    tail = round(0.4*N);
    tail_var = var(xa(N-tail+1:N));
    if tail_var < 1e-8
        counts.flatline = counts.flatline+1;
        if counts.flatline <= 3
            fprintf('  [flatline #%d] tail_var=%.2e\n', counts.flatline, tail_var);
            if length(sample_traj) < 3
                sample_traj{end+1} = xa(1:N);
            end
        end
        continue;
    end

    xa_n = xa + p.noise*std(xa)*randn(n_all,1);
    ya_n = ya + p.noise*std(ya)*randn(n_all,1);
    x_data = xa_n(1:N); y_data = ya_n(1:N);

    if std(x_data) < 1e-6
        counts.std_check = counts.std_check+1;
        if counts.std_check <= 3
            fprintf('  [std_check #%d] std(x_data)=%.2e\n', counts.std_check, std(x_data));
        end
        continue;
    end

    try
        [ok, savepack] = V6_ProcessFeatures( ...
            x_data, y_data, xa_n, ya_n, N, n_all, dt, 3, 11, p, ...
            sysdef, 0.05, 0.02, 10);
    catch me
        if ~isfield(counts,'feature_crash'); counts.feature_crash = 0; end
        counts.feature_crash = counts.feature_crash + 1;
        if counts.feature_crash <= 3
            fprintf('  [feature_crash #%d] %s\n', counts.feature_crash, me.message);
            fprintf('       in: %s (line %d)\n', me.stack(1).name, me.stack(1).line);
        end
        continue;
    end
    if ~ok
        if ~isfield(counts,'feature_fail'); counts.feature_fail = 0; end
        counts.feature_fail = counts.feature_fail + 1;
        if counts.feature_fail <= 5 && isfield(savepack,'fail_reason')
            fprintf('  [feature_fail #%d] reason: %s\n', counts.feature_fail, savepack.fail_reason);
        end
        continue;
    end

    counts.passed = counts.passed+1;
end

fprintf('\n=== toggle_switch NULL diagnostic (n=%d) ===\n', n_attempts);
fns = fieldnames(counts);
for i=1:numel(fns)
    fprintf('  %-16s %4d  (%.1f%%)\n', fns{i}, counts.(fns{i}), 100*counts.(fns{i})/n_attempts);
end

if ~isempty(sample_traj)
    figure;
    for i=1:length(sample_traj)
        subplot(length(sample_traj),1,i);
        plot(sample_traj{i}); title(sprintf('Sample flatlined trajectory %d', i));
    end
end