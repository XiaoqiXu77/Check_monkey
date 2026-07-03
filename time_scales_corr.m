%% Two-region analysis: time-resolved + window bars + post-stim scatter
areas = [1 2];  % 1=MCC, 2=LPFC
win_pre  = 13:20;
win_post = 21:30;

% If you already have these, keep yours
if ~exist('area_names','var') || numel(area_names) < 2
    area_names = {'MCC','LPFC'};
end

% ---------- Storage ----------
% We'll infer nTime from first valid neuron we encounter
nTime = [];
r_time = []; p_time = [];      % nTime x 2
r_pre  = nan(1,2); p_pre  = nan(1,2);
r_post = nan(1,2); p_post = nan(1,2);
tau_all = cell(1,2);
imp_post_all = cell(1,2);

for ai = 1:2
    area = areas(ai);

    % ================================
    % Select neurons by area
    % ================================
    area_raw = {data_by_neuron.area};
    if isnumeric(area_raw{1})
        area_vec = cell2mat(area_raw);
        neurons_idx = find(area_vec == area);
    else
        target = "mcc"; if area == 2, target = "lpfc"; end
        area_str = lower(strtrim(string(area_raw)));
        neurons_idx = find(area_str == target);
    end

    % ================================
    % Tau filter
    % ================================
    tau = nan(numel(neurons_idx), 1);
    for i = 1:numel(neurons_idx)
        tau(i) = data_by_neuron(neurons_idx(i)).tau;
    end
    valid = ~isnan(tau) & tau <= 900;
    tau = tau(valid);
    neurons_idx = neurons_idx(valid);
    nNeurons = numel(neurons_idx);

    tau_all{ai} = tau;

    if nNeurons == 0
        warning('No valid neurons for %s after tau filter.', area_names{area});
        continue;
    end

    % ================================
    % Infer nTime once
    % ================================
    if isempty(nTime)
        FR0 = data_by_neuron(neurons_idx(1)).firingRates_ch; % Ntrial x Ntime
        nTime = size(FR0, 2);
        r_time = nan(nTime, 2);
        p_time = nan(nTime, 2);
    end

    % ================================
    % Time-resolved importance: |slope| at each time
    % ================================
    importance_time = nan(nTime, nNeurons);

    for ni = 1:nNeurons
        idx = neurons_idx(ni);

        FR = data_by_neuron(idx).firingRates_ch;  % Ntrial x Ntime
        y  = data_by_neuron(idx).speed_est(:);    % Ntrial x 1
        % choice = data_by_neuron(idx).choice(:); % if you want to filter check trials

        % Optional: only check trials
        % idx_check = (choice == 0);
        % FR = FR(idx_check, :);
        % y  = y(idx_check);

        if std(y(~isnan(y))) == 0
            continue;
        end

        for t = 1:nTime
            x = FR(:, t);
            good = ~isnan(x) & ~isnan(y);
            if sum(good) < 3 || std(x(good)) == 0
                continue;
            end
            Y = x(good);
            X = [ones(sum(good),1), y(good)];
            if rank(X) < 2
                continue;
            end
            beta = X \ Y;
            importance_time(t, ni) = abs(beta(2));
        end
    end

    % ================================
    % Corr(tau, importance_time(t,:)) over time
    % ================================
    for t = 1:nTime
        imp_t = importance_time(t, :)';
        [r, p] = corr(tau, imp_t, 'Rows', 'complete', 'Type', 'Pearson');
        r_time(t, ai) = r;
        p_time(t, ai) = p;
    end

    % ================================
    % Window-based importance (mean FR in window -> slope)
    % ================================
    imp_pre  = nan(nNeurons, 1);
    imp_post = nan(nNeurons, 1);

    for ni = 1:nNeurons
        idx = neurons_idx(ni);
        FR = data_by_neuron(idx).firingRates_ch;
        y  = data_by_neuron(idx).speed_est(:);

        if std(y(~isnan(y))) == 0
            continue;
        end

        x_pre  = mean(FR(:, win_pre),  2, 'omitnan');
        x_post = mean(FR(:, win_post), 2, 'omitnan');

        % PRE
        good = ~isnan(x_pre) & ~isnan(y);
        if sum(good) >= 3 && std(x_pre(good)) > 0
            Y = x_pre(good);
            X = [ones(sum(good),1), y(good)];
            if rank(X) == 2
                beta = X \ Y;
                imp_pre(ni) = abs(beta(2));
            end
        end

        % POST
        good = ~isnan(x_post) & ~isnan(y);
        if sum(good) >= 3 && std(x_post(good)) > 0
            Y = x_post(good);
            X = [ones(sum(good),1), y(good)];
            if rank(X) == 2
                beta = X \ Y;
                imp_post(ni) = abs(beta(2));
            end
        end
    end

    [r_pre(ai),  p_pre(ai)]  = corr(tau, imp_pre,  'Rows','complete','Type','Pearson');
    [r_post(ai), p_post(ai)] = corr(tau, imp_post, 'Rows','complete','Type','Pearson');

    imp_post_all{ai} = imp_post;

    fprintf('\n%s:\n', area_names{area});
    fprintf('  PRE  (13:20): r=%.3f, p=%.3g\n', r_pre(ai),  p_pre(ai));
    fprintf('  POST (21:30): r=%.3f, p=%.3g\n', r_post(ai), p_post(ai));
end

%% ---------- Plot 1: time trajectory (two regions) ----------
figure; hold on;

% Time axis: use your tmin if available, else 1:nTime
if exist('tmin','var') && numel(tmin) >= nTime
    tAxis = tmin(1:nTime);
else
    tAxis = 1:nTime;
end

plot(tAxis(13:end), r_time(13:end,1), 'LineWidth', 2);
plot(tAxis(13:end), r_time(13:end,2), 'LineWidth', 2);

% Stimulus onset vertical line 
stim_idx = 21;  % stimulus onset time index (adjust if yours differs)

if stim_idx >= 1 && stim_idx <= numel(tAxis)
    xl = tAxis(stim_idx);
    yl = ylim;
    plot([xl xl], yl, 'k--', 'LineWidth', 1.5);
end


xlabel('Time');
ylabel('Corr(\tau, |slope|)');
title('Time-resolved correlation');
legend(area_names{1}, area_names{2}, 'Location', 'best');
grid on; box off;
set(gca, 'fontsize', 18);


%% ---------- Plot 2: grouped bar plot (pre vs post, two regions) ----------
% Use absolute r so stars can always go "above" the bars without looking dumb.
vals = abs([r_pre(1)  r_pre(2);
            r_post(1) r_post(2)]);   % rows: [pre; post], cols: [MCC LPFC]
pvals =     [p_pre(1)  p_pre(2);
            p_post(1) p_post(2)];

figure; hold on;
h = bar(1:2, vals, 'grouped');  % 2 groups (pre/post), 2 bars per group (regions)
set(gca, 'XTick', 1:2, 'XTickLabel', {'pre-choice', 'post-choice'});
ylabel('|Corr(\tau, |slope|)|');
% title('Correlations');
legend(area_names{1}, area_names{2}, 'Location', 'best');
grid on; box off;
set(gca, 'fontsize', 18);

% Stars per bar
yl = ylim; yr = yl(2) - yl(1);
offset = 0.05 * yr;

for k = 1:numel(h)  % k=1..2 regions
    for g = 1:2     % g=1..2 windows
        % Robust x position across MATLAB versions
        if isprop(h(k), 'XEndPoints')
            x = h(k).XEndPoints(g);
        else
            x = h(k).XData(g) + h(k).XOffset;
        end

        s = p2stars(pvals(g,k));
        if ~isempty(s)
            text(x, vals(g,k) + offset, s, ...
                'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
                'FontSize', 22, 'FontWeight', 'bold');
        end
    end
end
ylim([yl(1), max(vals(:)) + 3*offset]);

%% ---------- Plot 3: post-stim scatter (two regions) ----------
figure; hold on;

for ai = 1:2
    tau = tau_all{ai};
    imp_post = imp_post_all{ai};
    good = ~isnan(tau) & ~isnan(imp_post);

    scatter(tau(good), imp_post(good), 35, 'filled', 'MarkerFaceAlpha', 0.7);
end

xlabel('\tau (ms)');
ylabel('|slope| (FR~speed), post-stim (21:30)');
title('Post-stim window scatter: MCC vs LPFC');
legend(area_names{1}, area_names{2}, 'Location', 'best');
grid on; box off;
set(gca, 'fontsize', 18);
