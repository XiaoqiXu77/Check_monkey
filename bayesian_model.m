%==========================================================================
% Fit the discrete‐speed Bayesian model with sequential updates and compute
% classification accuracy against ground truth speeds.
%
% Assumptions:
% - behav is an M×P matrix already in your workspace.
% - Column indices:
%     20 → session ID
%      9 → ground‐truth speed (0 means “no ground truth”)
%      6 → currGauge (gauge size at check)
%     16 → prevGauge (gauge size at previous check)
%     17 → correctTrials (trials since last check)
% - True speeds ∈ {1,2,3,4}.
%
% Outputs:
%   - estSpeeds: MAP estimates for each valid session
%   - trueSpeeds: ground‐truth speeds per session
%   - accuracy: fraction of sessions correctly classified
%==========================================================================
monkey      = 2;
behav       = behav_all{monkey};
allSessions = unique(behav(:,20));      % session IDs
speeds      = 2:5;                      % candidate speeds s_k
K           = numel(speeds);
max_check   = 9; %max(behav(:, 18));
accuracy    = nan(max_check, 1);
uncertainty = nan(max_check, 1);

figure
for check = 1:max_check
    % reset per‐check
    trueSpeeds = [];
    estSpeeds  = [];
    entropy_est = [];
    
    for sess_id = 1:length(allSessions)
        sess = allSessions(sess_id);
        % only “check” trials 
        rows = find(behav(:,20)==sess & behav(:,5)==0);
        if isempty(rows), continue; end
        
        gt = behav(rows(1),9);            % ground truth
        if gt==0, continue; end          % skip if unknown
        
        % extract gauge/check info
        cg = behav(rows,6);   % current gauge
        pg = behav(rows,16);  % previous gauge
        ct = behav(rows,17);  % # correct trials
        ck = behav(rows,18);  % check index
        
        % only include checks up to this threshold
        valid = ck<=check & ct>0; %  & cg~=pg
        validRows = rows(valid);
        pg_valid = pg(valid);
        if isempty(validRows), continue; end
        
        % compute interval data
        deltaG = cg(valid) - pg(valid);   % observed increments
        deltaT = ct(valid);               % interval lengths
        
        % sequential Bayes over these N intervals
        prior = ones(1,K)/K;              % uniform 1×K prior
        for i = 1:numel(deltaG)
            % 1) compute scalar emissions p_em_k for each s_k
            for k = 1:K
                s = speeds(k);
                u_min    = max(0,    s*deltaG(i) - deltaT(i));
                u_max    = min(s, s*(deltaG(i)+1) - deltaT(i));
                p_em_k(k)= max(0, (u_max - u_min)) / s;   % 1×K 
            end
            
            % 2) compute expected increment under prior
            expG = deltaT(i) * (prior * (1./speeds)');   % scalar 
            
            % — store predicted gauge size in new column 23 —           
            rawPred = pg_valid(i) + expG;
            % round to nearest integer and clamp into [1,7]
            intPred = round(rawPred);
            intPred = min(max(intPred,1),7);
            if i == numel(deltaG)
                behav(validRows(i),23) = intPred;
            end
            
            % 3) prediction error for this check
            delta_i = deltaG(i) - expG;
            
            % store PE in column 22 of behav for this check‐row
            if i == numel(deltaG)
                behav(validRows(i), 22) = delta_i;
            end
            
            % 4) Bayes update: posterior ∝ prior .* likelihood
            unnorm = prior .* p_em_k;                   % 1×K
            if all(unnorm==0)
                post = prior;                           % uninformative
            else
                post = unnorm/sum(unnorm);              % normalize
            end
            prior = post;                               % carry forward
        end
        
        % final MAP speed estimate
        [~, idx] = max(prior);
        est = speeds(idx);
        
        % record for accuracy
        trueSpeeds(end+1) = gt;
        estSpeeds(end+1)  = est - 1;
        
        % also save estimate in behav column 21
        behav(validRows(end),21) = est - 1;
        
        % compute uncertainty measured by the Shannon entropy
        P = prior(prior > 0);
        H = -sum(P .* log2(P));
        
        behav(validRows(end),25) = H;
        entropy_est = [entropy_est, H];
    end
    
    % classification accuracy at this check
    accuracy(check) = mean(estSpeeds == trueSpeeds);
    fprintf('Max check ≤%d: accuracy = %.1f%% (N=%d)\n', ...
            check, accuracy(check)*100, numel(trueSpeeds));
        
    uncertainty(check) = mean(entropy_est);    
        
    % confusion matrix 
    subplot(3, 3, check);
    trueVecs = full(ind2vec(trueSpeeds, 4));
    estVecs  = full(ind2vec(estSpeeds, 4));
    plotconfusion(trueVecs, estVecs);

    classLabels = {"1", "2", "3", "4"};
    set(gca, 'XTickLabel', classLabels, 'YTickLabel', classLabels);
    title(sprintf('Confusion Matrix %d', check));
    
end

% update behav matrix
behav_all{monkey} = behav;

% Plot accuracy vs. checks
figure
plot(1:max_check, accuracy*100, '-o', 'LineWidth', 3)
xlabel('Up to check #n'); ylabel('Accuracy (%)');
ylim([55, 75])
title('Sequential learning based on Bayesian model')
grid on
set(gca, 'FontSize', 18)

figure
plot(1:max_check, uncertainty, '-o', 'LineWidth', 3)
xlabel('Up to check #n'); ylabel('Entropy');
title('Uncertainty of progress rate estimation')
grid on
set(gca, 'FontSize', 18)

%% compute the entropy of each check
figure

for monkey = 1:n_monkeys
    behav = behav_all{monkey};

    prior = 1/K * ones(1, K);
    for check = 1:max_check
        % reset per‐check
        entropy_check = [];
        
        for sess_id = 1:length(allSessions)
            sess = allSessions(sess_id);
            % only “check” trials 
            rows = find(behav(:,20)==sess & behav(:,5)==0);
            if isempty(rows), continue; end

            gt = behav(rows(1),9);            % ground truth
            if gt==0, continue; end          % skip if unknown

            % extract gauge/check info
            cg = behav(rows,6);   % current gauge
            pg = behav(rows,16);  % previous gauge
            ct = behav(rows,17);  % # correct trials
            ck = behav(rows,18);  % check index

            % only include one check
            valid = ck==check & ct>0; %  & cg~=pg
            validRows = rows(valid);
            pg_valid = pg(valid);
            if isempty(validRows), continue; end

            % compute interval data
            deltaG = cg(valid) - pg(valid);   % observed increments
            deltaT = ct(valid);               % interval lengths

            % 1) compute scalar emissions p_em_k for each s_k
            for k = 1:K
                s = speeds(k);
                u_min    = max(0,    s*deltaG - deltaT);
                u_max    = min(s, s*(deltaG+1) - deltaT);
                p_em_k(k)= max(0, (u_max - u_min)) / s;   % 1×K 
            end
            
            unnorm = p_em_k;
            if all(unnorm==0)
                post = prior;                           % uninformative
            else
                post = unnorm/sum(unnorm);              % normalize
            end
            
            % compute the Shannon entropy
            P = post(post > 0);
            H = -sum(P .* log2(P));
            entropy_check = [entropy_check, 1 - H/log2(4)];
        end
        info_check(check) = mean(entropy_check);
    end
    
    subplot(1, n_monkeys, monkey)
    plot(1:max_check, info_check, '-o', 'LineWidth', 3)
    xlabel('Check #n'); ylabel('1 - Entropy/log(#classes))');
    title(sprintf('Monkey %s', monkey_names(monkey)))
    grid on
    set(gca, 'FontSize', 18)
end

%% linear regression: PE ~ RT_next
pe_threshold = 2;
max_delay = 1;
beta = nan(monkey, max_delay);
pval = nan(monkey, max_delay);
for monkey = 1:n_monkeys
    behav = behav_all{monkey};
    
    for delay = 1:max_delay
        idx_check = find(behav(:, 5) == 0 & behav(:, 9) ~= 0 & behav(:, 6) < 7);
        idx_valid = find(behav(idx_check+delay, 5) ~= 0);
        pe = behav(idx_check(idx_valid), 22);
        rt_next = behav(idx_check(idx_valid)+delay, 8);

        % clear outliers
        idx_abnormal = find(abs(pe) > pe_threshold);
        pe(idx_abnormal) = [];
        rt_next(idx_abnormal) = [];

        mdl = fitlm(abs(pe), rt_next);
        beta(monkey, delay) = mdl.Coefficients.Estimate(2);
        pval(monkey, delay) = mdl.Coefficients.pValue(2);
    end
end

%%
[n_monkeys, max_delay] = size(beta);
if max_delay ~= 1
    error('This script expects exactly one delay (max_delay = 1).');
end

%--- Create figure with larger size for publication ---
figure('Units', 'inches', 'Position', [1, 1, 6, 4]);  % 6×4 inches

%--- Bar plot ---
hBar = bar(1:n_monkeys, beta, 0.6, 'FaceColor', [0.2 0.2 0.7], 'LineWidth', 1.5);
hold on;

%--- Mark significance (p < 0.05) with asterisk ---
ylims = ylim;
yRange = ylims(2) - ylims(1);
yOffset = 0.03 * yRange;
for iMonkey = 1:n_monkeys
    if pval(iMonkey) < 0.05
        text(iMonkey, beta(iMonkey) + yOffset, '*', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment',   'bottom', ...
            'FontSize', 20, ...
            'FontWeight', 'bold');
    end
end

%--- Axes labels and title ---
xlabel('Monkey', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('β weight (|PE| → RT_{next})', 'FontSize', 18, 'FontWeight', 'bold');
title('Regression of |PE| on RT_{next}', 'FontSize', 20, 'FontWeight', 'bold');

%--- Ticks and formatting ---
set(gca, 'FontSize', 16, ...                     % larger tick labels
         'XLim', [0.5, n_monkeys + 0.5], ...       % tight x‐limits
         'XTick', 1:n_monkeys, ...                 % integer ticks
         'XTickLabel', arrayfun(@num2str, 1:n_monkeys, 'UniformOutput', false), ...
         'Box', 'off', ...
         'LineWidth', 1.2);
grid on;
