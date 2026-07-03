%% Initiation time at different speeds (per monkey + pooled, mixed-effects)
n = 4;
colors = [0 0.45 0.75; 0.75 0.15 0.15];   % colors for individual monkeys

all_betas  = cell(1, n_monkeys);
all_rt     = cell(1, n_monkeys);   % raw log initiation time per session x speed
all_rt_dm  = cell(1, n_monkeys);   % session-demeaned log initiation time per session x speed

% ---- Accumulators for mixed-effects model (long format), session-demeaned response
mm_LogRTdm = [];   % response: log initiation time demeaned within session
mm_Speed   = [];   % predictor (1..n)
mm_Monkey  = [];   % monkey id (numeric -> categorical)

%% =========================
% FIGURE 1: Individual monkeys (raw log initiation time), MATLAB 2018a: subplot
% ==========================
figure('Color','w');
colors = [0.2, 0.6, 0.8;   % Monkey A (blueish)
          0.8, 0.4, 0.4];

for monkey = 1:n_monkeys
    behav = behav_all{monkey};

    % robust session indexing
    sess_ids   = unique(behav(:,11));
    n_sessions = numel(sess_ids);

    rt_all    = NaN(n_sessions, n);   % raw
    rt_all_dm = NaN(n_sessions, n);   % within-session demeaned
    betas     = NaN(n_sessions, 1);

    for iSess = 1:n_sessions
        sess = sess_ids(iSess);

        % --- Compute raw log initiation time per speed for this session
        rt_speed = NaN(1, n);
        for s = 1:n
            idx = (behav(:,11) == sess) & ...
                  (behav(:,21) == s) & ...
                  (behav(:,5)  ~= 0) & ...
                  (behav(:,25) > 0);

            rt_speed(s) = mean(log(1000 .* behav(idx,25)), 'omitnan');
        end

        rt_all(iSess,:) = rt_speed;

        % --- Session demeaning (only if >=2 valid speeds)
        valid = ~isnan(rt_speed);
        if sum(valid) >= 2
            sess_mean    = mean(rt_speed(valid), 'omitnan');
            rt_speed_dm  = rt_speed - sess_mean;
            rt_all_dm(iSess,:) = rt_speed_dm;

            % ---- Add demeaned observations to mixed model table
            for s = 1:n
                if valid(s)
                    mm_LogRTdm(end+1,1) = rt_speed_dm(s);
                    mm_Speed(end+1,1)   = s;
                    mm_Monkey(end+1,1)  = monkey;
                end
            end

            % --- Regression slope (raw log initiation time vs speed) for your per-session beta summary
            x = (1:n)';
            X = [ones(sum(valid),1) x(valid)];
            b = regress(rt_speed(valid)', X);
            betas(iSess) = b(2);
        end
    end

    all_betas{monkey} = betas;
    all_rt{monkey}    = rt_all;
    all_rt_dm{monkey} = rt_all_dm;

    % monkey name robust to cell array / string array / char array
    mname = monkey_names(monkey);
    if iscell(monkey_names), mname = monkey_names{monkey}; end

    % --- Plot per monkey (raw)
    subplot(1, n_monkeys, monkey); hold on;

    m_rt   = mean(rt_all, 1, 'omitnan');
    sem_rt = std(rt_all, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(rt_all), 1));

    errorbar(1:n, m_rt, sem_rt, ...
        'o-', 'Color', colors(monkey,:), 'LineWidth', 2, ...
        'MarkerFaceColor', colors(monkey,:), 'MarkerSize', 8);

    xlabel('Progress rate', 'FontSize', 15, 'FontWeight', 'bold');
    ylabel('Log Initiation Time (ms)', 'FontSize', 15, 'FontWeight', 'bold');
    title(sprintf('Monkey %s', char(mname)), 'FontSize', 16);
    set(gca, 'FontSize', 14, 'XTick', 1:n, 'Box', 'off', 'YGrid', 'on');
    hold off;

    % ---- Statistical Test on Slopes (per monkey)
    btmp = betas(~isnan(betas));
    if numel(btmp) >= 2
        [~, pval, ~, stats] = ttest(btmp);
        fprintf('Monkey %s: mean beta = %.4f ± %.4f, t(%d)=%.2f, p=%.4f\n', ...
            char(mname), mean(btmp), std(btmp)/sqrt(numel(btmp)), stats.df, stats.tstat, pval);
    else
        fprintf('Monkey %s: not enough valid sessions for t-test (n=%d)\n', char(mname), numel(btmp));
    end
end

%% =========================
% FIGURE 2: Pooled session-demeaned plot (NO LME fit line, NO CI)
% ==========================
rt_pooled_dm = vertcat(all_rt_dm{:});   % pooled session x speed (demeaned)

figure('Color','w'); hold on;

m_rt_dm   = mean(rt_pooled_dm, 1, 'omitnan');
sem_rt_dm = std(rt_pooled_dm, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(rt_pooled_dm), 1));

errorbar(1:n, m_rt_dm, sem_rt_dm, ...
    'o-', 'Color', [0 0 0], 'LineWidth', 3, ...
    'MarkerFaceColor', [0 0 0], 'MarkerSize', 8);

xlabel('Progress rate', 'FontSize', 15, 'FontWeight', 'bold');
ylabel('Demeaned Log Initiation Time (ms)', 'FontSize', 15, 'FontWeight', 'bold');
% title('Pooled (session-demeaned): mean ± SEM', 'FontSize', 16);
set(gca, 'FontSize', 24, 'XTick', 1:n, 'Box', 'off', 'YGrid', 'on');
hold off;

%% =========================
% Mixed-effects model (session-demeaned response, no Session RE, random slope by Monkey)
% ==========================
tbl = table(mm_LogRTdm, mm_Speed, categorical(mm_Monkey), ...
    'VariableNames', {'LogRTdm','Speed','Monkey'});

lme = fitlme(tbl, 'LogRTdm ~ Speed + (Speed|Monkey)');

coefTbl   = lme.Coefficients;
ix        = strcmp(coefTbl.Name, 'Speed');
betaSpeed = coefTbl.Estimate(ix);
seSpeed   = coefTbl.SE(ix);
pSpeed    = coefTbl.pValue(ix);

fprintf('\nMIXED MODEL: Log Initiation Time dm (session-centered) ~ Speed + (Speed|Monkey)\n');
fprintf('Fixed effect (Speed): beta = %.6f ± %.6f (SE), p = %.6g\n', betaSpeed, seSpeed, pSpeed);

%% =========================
% Mixed-effects model (session-demeaned response, no Session RE,
% common slope (fixed) + random intercept by Monkey)
% ==========================
tbl = table(mm_LogRTdm, mm_Speed, categorical(mm_Monkey), ...
    'VariableNames', {'LogRTdm','Speed','Monkey'});

% Common slope for Speed; intercept varies by Monkey
lme = fitlme(tbl, 'LogRTdm ~ Speed + (1|Monkey)');

coefTbl   = lme.Coefficients;
ix        = strcmp(coefTbl.Name, 'Speed');
betaSpeed = coefTbl.Estimate(ix);
seSpeed   = coefTbl.SE(ix);
pSpeed    = coefTbl.pValue(ix);

fprintf('\nMIXED MODEL: Log Initiation Time dm (session-centered) ~ Speed + (1|Monkey)\n');
fprintf('Fixed effect (Speed): beta = %.6f ± %.6f (SE), p = %.6g\n', betaSpeed, seSpeed, pSpeed);
