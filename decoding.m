%% PARAMETERS
nIter = 50;        % Number of iterations for random train/test splits
nPseudo = 50;       % Fixed total number of pseudo-trials per condition (to be split into training and test)
winSize = 1;        % Use 3 time bins (centered on current time)
trainRatio = 0.8;   % Ratio of training samples
% rng('default');     % For reproducibility
nBoot = 50;         % number of permutation tests
alpha = 0.05;       % significance level

%% ===========================================
%% DECODING AT CHECK WINDOW (firingRates_ch)
%% ===========================================

% Define decoding tasks at check window.
% Tasks: speed, gauge_size, pre_gs, wl (binned into 4 classes), gauge_diff (gauge_size-pre_gs, range 0:4), and choice.
tasks_check = {'gauge_size', 'pre_gs', 'speed', 'gauge_size', 'pre_gs', 'wl', 'gauge_diff', 'choice'};  %'reward'
taskClasses_check = {1:7, 0:6, 1:4, 1:7, 0:6, 1:4, 0:4, [0 1]};  %1:10
nTasks_check = length(tasks_check);

% Get list of unique areas from data_by_neuron
areas = unique([data_by_neuron.area]);
nAreas = length(areas);

% Preallocate results: decodingResults_check{task, area} will contain a vector of decoding accuracy vs. time.
decodingResults_check = cell(nTasks_check, nAreas);
sig_acc = cell(nTasks_check, nAreas);

%% Loop over areas for check window
for a = 1:nAreas
    currentArea = areas(a);
    % Get indices of neurons from current area.
    idx_area = find([data_by_neuron.area] == currentArea);
    
    % Select neurons with no NaN in firingRates_ch and with at least one trial having nonzero speed_ch.
    valid_neurons_all = [];
    for i = idx_area
        neuron = data_by_neuron(i);
        if all(~isnan(neuron.firingRates_ch(:))) && any(neuron.speed_est ~= 0)
            valid_neurons_all = [valid_neurons_all, i];
        end
    end
    
    % Skip area if no valid neurons.
    if isempty(valid_neurons_all)
        for tsk = 1:nTasks_check
            decodingResults_check{tsk,a} = [];
        end
        continue;
    end
    
    % Use the time vector from the first valid neuron.
    timeVector = data_by_neuron(valid_neurons_all(1)).time;
    nTime = length(timeVector);
    
    %% Preprocess wl: Bin the wl values (check window trials) into 4 classes.
    all_wl = [];
    for i = valid_neurons_all
        neuron = data_by_neuron(i);
        validIdx = (neuron.speed_ch ~= 0);
        all_wl = [all_wl; neuron.wl(validIdx)];
    end
    qEdges = prctile(all_wl, [25,50,75]);
    for i = valid_neurons_all
        neuron = data_by_neuron(i);
        validIdx = (neuron.speed_ch ~= 0);
        wlVals = neuron.wl(validIdx);
        wl_bin = zeros(size(wlVals));
        wl_bin(wlVals <= qEdges(1)) = 1;
        wl_bin(wlVals > qEdges(1) & wlVals <= qEdges(2)) = 2;
        wl_bin(wlVals > qEdges(2) & wlVals <= qEdges(3)) = 3;
        wl_bin(wlVals > qEdges(3)) = 4;
        data_by_neuron(i).wl_bin = nan(size(neuron.wl));
        data_by_neuron(i).wl_bin(validIdx) = wl_bin;
    end
    
    %% Compute gauge_diff: gauge_size - pre_gs for check window trials, keep only trials with difference in [0,4]
    for i = valid_neurons_all
        neuron = data_by_neuron(i);
        validIdx = (neuron.speed_ch ~= 0);
        gauge_diff = neuron.gauge_size(validIdx) - neuron.pre_gs(validIdx);
        validDiff = (gauge_diff >= 0 & gauge_diff <= 4);
        diff_vals = nan(size(gauge_diff));
        diff_vals(validDiff) = gauge_diff(validDiff);
        data_by_neuron(i).gauge_diff = nan(size(neuron.gauge_size));
        data_by_neuron(i).gauge_diff(validIdx) = diff_vals;
    end

    % Preprocess number of rewards: Bin into 9 classes.
    all_wl = [];
    for i = valid_neurons_all
        neuron = data_by_neuron(i);
        validIdx = (neuron.speed_ch ~= 0);
        all_wl = [all_wl; neuron.nb_re(validIdx)];
    end
    qEdges = 4:4:36;
    binEdges = [-Inf, qEdges, Inf];   % 10 bins -> labels 0..9

    for i = valid_neurons_all
        neuron = data_by_neuron(i);
        validIdx = (neuron.speed_ch ~= 0);

        wlVals = neuron.nb_re(validIdx);

        % discretize gives 1..10, so subtract 1 to get 0..9
        wl_bin = discretize(wlVals, binEdges)   ;

        data_by_neuron(i).nre_bin = nan(size(neuron.nb_re));
        data_by_neuron(i).nre_bin(validIdx) = wl_bin;
    end
    
    %% For each decoding task at check window, select neurons that have at least one trial in every required class.
    valid_neurons = cell(nTasks_check,1);
    for tsk = 1:nTasks_check
        valid_idx = [];
        for i = valid_neurons_all
            neuron = data_by_neuron(i);
            validIdx = (neuron.speed_est ~= 0); % valid trials for check window
            switch tasks_check{tsk}
                case 'speed'
                    labs = neuron.speed_est(validIdx);
                case 'gauge_size'
                    labs = neuron.gauge_size(validIdx);
                case 'pre_gs'
                    labs = neuron.pre_gs(validIdx);
                case 'wl'
                    labs = neuron.wl_bin(validIdx);
                case 'gauge_diff'
                    labs = neuron.gauge_diff(validIdx);
                    validIdx = ~isnan(labs);
                    labs = labs(validIdx);
                case 'choice'
                    labs = neuron.choice(validIdx);
                case 'reward'
                    labs = neuron.nre_bin(validIdx);
            end
            reqClasses = taskClasses_check{tsk};
            if all(ismember(reqClasses, unique(labs)))
                valid_idx = [valid_idx, i];
            end
        end
        valid_neurons{tsk} = valid_idx;
    end
    
    %% Initialize accuracy matrices for each check-window task (nIter x nTime)
    accTask = cell(nTasks_check,1);
    for tsk = 1:nTasks_check
        accTask{tsk} = zeros(nIter, nTime);
        accTask_perm{tsk} = zeros(nIter, nTime, nBoot);
    end
    
    %% Main decoding loop for check window
    for iter = 1:nIter
        for t = 1:nTime
            % Determine window indices (using winSize=3, centered on t; adjust at boundaries)
            if t == 1
                win = t:min(nTime, t+1);
            elseif t == nTime
                win = max(1, t-1):t;
            else
                win = t-1:t+1;
            end
            
            % Loop over each task.
            for tsk = 1:nTasks_check
                neurons_idx = valid_neurons{tsk};
                nNeurons = length(neurons_idx);
                if nNeurons == 0
                    accTask{tsk}(iter, t) = NaN;
                    continue;
                end
                
                % For each neuron, extract feature (mean firing rate over win) and label.
                % Then, for each condition, first split the trials into training and testing sets,
                % and then sample with replacement.
                neuronData = cell(nNeurons,1);
                reqClasses = taskClasses_check{tsk};
                nClasses = length(reqClasses);
                for nIdx = 1:nNeurons
                    neuron = data_by_neuron(neurons_idx(nIdx));
                    validIdx = (neuron.speed_est ~= 0);
                    feat = mean(neuron.firingRates_ch(validIdx, win), 2);
                    % Get labels for the task:
                    switch tasks_check{tsk}
                        case 'speed'
                            labs = neuron.speed_est(validIdx);
                        case 'gauge_size'
                            labs = neuron.gauge_size(validIdx);
                        case 'pre_gs'
                            labs = neuron.pre_gs(validIdx);
                        case 'wl'
                            labs = neuron.wl_bin(validIdx);
                        case 'gauge_diff'
                            labs = neuron.gauge_diff(validIdx);
                            validIdx = ~isnan(labs);
                            labs = labs(validIdx);
                            feat = feat(validIdx);
                        case 'choice'
                            labs = neuron.choice(validIdx);
                        case 'reward'
                            labs = neuron.nre_bin(validIdx);
                    end
                    
                    % For each class, split trials into training and testing sets first,
                    % then sample with replacement.
                    neuronData{nIdx} = struct();
                    for c = 1:nClasses
                        currClass = reqClasses(c);
                        idxClass = find(labs == currClass);
                        if isempty(idxClass)
                            neuronData{nIdx}.(['class' num2str(currClass)]) = [];
                        else
                            % Split indices into training and testing.
                            nAvail = length(idxClass);
                            rp = randperm(nAvail);
                            nTrainAvail = max(1, floor(nAvail * trainRatio));
                            training_indices = idxClass(rp(1:nTrainAvail));
                            testing_indices = idxClass(rp(nTrainAvail+1:end));
                            if isempty(testing_indices)
                                testing_indices = training_indices;
                            end
                            % Now sample with replacement.
                            nTrainFixed = round(trainRatio * nPseudo);
                            nTestFixed = nPseudo - nTrainFixed;
                            train_samples = feat(training_indices(randi(length(training_indices), nTrainFixed, 1)));
                            test_samples  = feat(testing_indices(randi(length(testing_indices), nTestFixed, 1)));
                            
                            neuronData{nIdx}.(['class' num2str(currClass)]).train = train_samples;
                            neuronData{nIdx}.(['class' num2str(currClass)]).test  = test_samples;
                        end
                    end
                end
                
                % Assemble pseudo-population for current task and time bin.
                trainPseudo = [];
                testPseudo = [];
                trainLabels = [];
                testLabels = [];
                for c = 1:nClasses
                    currClass = reqClasses(c);
                    nTrainFixed = round(trainRatio * nPseudo);
                    nTestFixed = nPseudo - nTrainFixed;
                    pseudoTrainMat = nan(nTrainFixed, nNeurons);
                    pseudoTestMat  = nan(nTestFixed, nNeurons);
                    for nIdx = 1:nNeurons
                        dataField = ['class' num2str(currClass)];
                        samplesTrain = neuronData{nIdx}.(dataField).train;
                        samplesTest  = neuronData{nIdx}.(dataField).test;
                        pseudoTrainMat(:, nIdx) = samplesTrain;
                        pseudoTestMat(:, nIdx)  = samplesTest;
                    end
                    trainPseudo = [trainPseudo; pseudoTrainMat];
                    testPseudo  = [testPseudo; pseudoTestMat];
                    trainLabels = [trainLabels; repmat(currClass, nTrainFixed, 1)];
                    testLabels  = [testLabels; repmat(currClass, nTestFixed, 1)];
                end
                
                if isempty(trainPseudo) || isempty(testPseudo)
                    accTask{tsk}(iter,t) = NaN;
                else
                    mdl = fitcecoc(trainPseudo, trainLabels, 'Learners', templateSVM('KernelFunction','linear'));
                    preds = predict(mdl, testPseudo);
                    accTask{tsk}(iter,t) = mean(preds == testLabels);
                    
                    % ----------- Save weights for SPEED task only (check window) ------------
                    if strcmp(tasks_check{tsk}, 'speed')
                        % Binary SVMs inside ECOC; get average magnitude across classifiers
                        allW = [];
                        for k = 1:numel(mdl.BinaryLearners)
                            allW = [allW abs(mdl.BinaryLearners{k}.Beta)];
                        end
                        neuron_importance{iter, t, a} = mean(allW, 2);   % neurons × 1
                    end
                    % -------------------------------------------------------------------------
    
                    
                    % permutation test
                    for boot = 1:nBoot
                        train_perm = trainLabels(randperm(length(trainLabels)));
                        mdl = fitcecoc(trainPseudo, train_perm, 'Learners', templateSVM('KernelFunction','linear'));
                        preds = predict(mdl, testPseudo);
                        accTask_perm{tsk}(iter,t,boot) = mean(preds == testLabels);
                    end  % end permutation loop
                end
            end  % end task loop
        end  % end time loop
    end  % end iteration loop
    
    % Average over iterations for each task and store for current area.
    for tsk = 1:nTasks_check
        decodingResults_check{tsk,a} = nanmean(accTask{tsk}, 1);
        
        for t = 1:nTime
            perm_t = accTask_perm{tsk}(:, t, :);
            thresh = prctile(perm_t(:), (1-alpha)*100);
            sig_acc{tsk, a}(t) = decodingResults_check{tsk, a}(t) > thresh;
        end
    end
end

%% Plotting for Check Window: One plot per decoding task comparing two areas.
for tsk = 1:nTasks_check
    figure;
    hold on;
    taskName = tasks_check{tsk};
    colors = lines(2);
    for a = 1:min(2, nAreas)
        if ~isempty(decodingResults_check{tsk,a})
            plot(timeVector, decodingResults_check{tsk,a}, 'LineWidth', 3, 'Color', colors(a,:));
            mask = sig_acc{tsk, a} == 1;
            s = scatter(timeVector(mask), decodingResults_check{tsk,a}(mask), 'ro', 'filled', 'HandleVisibility','off');
            s.SizeData = 70;
        end
    end
    plot(timeVector, repmat(1/length(taskClasses_check{tsk}), ...
                length(timeVector)), 'g--', 'LineWidth', 3)
    xlabel('Time');
    ylabel('Decoding Accuracy');
%     ylim([0 1]);
    switch taskName
        case 'speed'
            title('Progress rate Decoding (Choice Window)');
        case 'gauge_size'
            title('Gauge Size Decoding (Check/Work Window)');
        case 'pre_gs'
            title('Prev GS Decoding (Check/Work Window)');
        case 'wl'
            title('WL Decoding (Check/Work Window)');
        case 'gauge_diff'
            title('Gauge Diff Decoding (Check/Work Window)');
        case 'choice'
            title('Choice Decoding (Check/Work Window)');
        case 'reward'
            title('Number of rewards')
    end
    
    legend({'MCC', 'LPFC', 'Chance'}, 'Location', 'best');
    set(gca, 'FontSize', 18)
    grid on;
    hold off;
end

%%
save('results_rate_choice.mat', 'decodingResults_check', 'sig_acc', 'accTask_perm')

% Plotting for Check Window: One plot per decoding task comparing MCC vs LPFC
% Style to match Fig 5a (Stoll et al., 2016): MCC black, LPFC blue,
% chance grey dashed, significance = bold line segments.

mccColor    = [0 0 0];
lpfcColor   = [0 0.4470 0.7410];    % MATLAB default blue, close to paper
chanceColor = 0.65 * [1 1 1];       % light grey

lwThin   = 2.0;
lwThick  = 4.5;   % significance thickness
lwChance = 1.8;

usePercent = true;  % Fig 5a uses % correct

for tsk = 1:nTasks_check
    figure('Color','w');
    hold on;

    taskName = tasks_check{tsk};

    % --- MCC (a=1) and LPFC (a=2) only ---
    for a = 1:min(2, nAreas)
        y = decodingResults_check{tsk,a};
        if isempty(y), continue; end

        if usePercent
            yPlot = 100 * y(:);
        else
            yPlot = y(:);
        end

        sigMask = (sig_acc{tsk,a}(:) == 1);

        if a == 1
            col = mccColor;
        else
            col = lpfcColor;
        end

        plot_bold_sig(timeVector(:), yPlot, sigMask, col, lwThin, lwThick);
    end

    % --- Chance line (grey dashed) ---
    chance = 1 / numel(taskClasses_check{tsk});
    if usePercent, chance = 100 * chance; end
    plot(timeVector, chance * ones(size(timeVector)), '--', ...
        'Color', chanceColor, 'LineWidth', lwChance, 'HandleVisibility','off');

    % Labels
    xlabel('Time from lever onset (s)');
    if usePercent
        ylabel('Per cent correct decoding (%)');
    else
        ylabel('Decoding accuracy');
    end
    title(task_title(taskName));

    % Axes cosmetics (paper-like)
    ax = gca;
    set(ax, ...
        'FontSize', 18, ...
        'FontName', 'Arial', ...
        'Box', 'off', ...
        'TickDir', 'out', ...
        'LineWidth', 1.5);
    grid off;

%     % In-plot labels instead of a legend
%     yl = ylim;
%     x0 = timeVector(1) + 0.04 * (timeVector(end) - timeVector(1));
%     y0 = yl(2) - 0.06 * (yl(2) - yl(1));
%     text(x0, y0, 'MCC',  'Color', mccColor,  'FontWeight','bold', 'FontSize', 18);
%     text(x0, y0 - 0.10*(yl(2)-yl(1)), 'LPFC', 'Color', lpfcColor, 'FontWeight','bold', 'FontSize', 18);
    % --- Legend (outside the axes so it never overlaps the data) ---
    
    % Make two dummy handles so the legend stays clean even if plot_bold_sig
    % creates multiple line objects per area.
    hMCC  = plot(nan, nan, '-', 'Color', mccColor,  'LineWidth', lwThin, 'DisplayName','MCC');
    hLPFC = plot(nan, nan, '-', 'Color', lpfcColor, 'LineWidth', lwThin, 'DisplayName','LPFC');

    lgd = legend([hMCC hLPFC], {'MCC','LPFC'}, ...
        'Location','northeastoutside', 'Box','off');
    set(lgd, 'FontSize', 18, 'FontName', 'Arial');

    % Make room for the outside legend
    ax = gca;
    ax.Position(3) = ax.Position(3) * 0.80;
    hold off;
end


%% ===========================================
% DECODING AT FEEDBACK WINDOW (firingRates_fb)
% ===========================================
% Here we decode two tasks:
%   1) speed_fb (4 classes) and 
%   2) feedback (binary: 1 or 2)
%   3) gauge size
%   4) work length
%
% We'll process these tasks analogously to the check window.

% Define feedback window tasks.
tasks_fb = {'speed_fbest', 'feedback', 'gs', 'wl'};
taskClasses_fb = {1:4, [1 2], 1:7, 1:4};
nTasks_fb = length(tasks_fb);

% Preallocate results: decodingResults_fb{task, area} for feedback window.
decodingResults_fb = cell(nTasks_fb, nAreas);
sig_acc = cell(nTasks_fb, nAreas);

for a = 1:nAreas
    currentArea = areas(a);
    % Get indices of neurons from current area.
    idx_area = find([data_by_neuron.area] == currentArea);
    
    % Select neurons with no NaN in firingRates_fb and with at least one trial having nonzero speed_fb.
    valid_neurons_fb = [];
    for i = idx_area
        neuron = data_by_neuron(i);
        if all(~isnan(neuron.firingRates_fb(:))) && any(neuron.speed_estfb ~= 0)
            valid_neurons_fb = [valid_neurons_fb, i];
        end
    end
    
    % Skip area if no valid neurons.
    if isempty(valid_neurons_fb)
        for tsk = 1:nTasks_fb
            decodingResults_fb{tsk,a} = [];
        end
        continue;
    end
    
    all_wl = [];
    for i = valid_neurons_fb
        neuron = data_by_neuron(i);
        validIdx = (neuron.speed_estfb ~= 0);
        all_wl = [all_wl; neuron.wl(validIdx)];
    end
    qEdges = prctile(all_wl, [25,50,75]);
    for i = valid_neurons_fb
        neuron = data_by_neuron(i);
        validIdx = (neuron.speed_estfb ~= 0);
        wlVals = neuron.wl(validIdx);
        wl_bin = zeros(size(wlVals));
        wl_bin(wlVals <= qEdges(1)) = 1;
        wl_bin(wlVals > qEdges(1) & wlVals <= qEdges(2)) = 2;
        wl_bin(wlVals > qEdges(2) & wlVals <= qEdges(3)) = 3;
        wl_bin(wlVals > qEdges(3)) = 4;
        data_by_neuron(i).wl_bin = nan(size(neuron.wl));
        data_by_neuron(i).wl_bin(validIdx) = wl_bin;
    end
    
    % Use the time vector from the first valid neuron for feedback window.
    timeVector_fb = data_by_neuron(valid_neurons_fb(1)).time;
    nTime_fb = length(timeVector_fb);
    
    %% For each feedback task, select neurons that have at least one trial in every required class.
    valid_neurons_fb_tasks = cell(nTasks_fb,1);
    for tsk = 1:nTasks_fb
        valid_idx = [];
        for i = valid_neurons_fb
            neuron = data_by_neuron(i);
            % For feedback window, use valid trials with nonzero speed_fb.
            validIdx = (neuron.speed_estfb ~= 0);
            switch tasks_fb{tsk}
                case 'speed_fbest'
                    labs = neuron.speed_estfb(validIdx);
                case 'feedback'
                    labs = neuron.feedback(validIdx);
                case 'gs'
                    labs = neuron.gs_fb(validIdx);
                case 'wl'
                    labs = neuron.wl_bin(validIdx);
            end
            reqClasses = taskClasses_fb{tsk};
            if all(ismember(reqClasses, unique(labs)))
                valid_idx = [valid_idx, i];
            end
        end
        valid_neurons_fb_tasks{tsk} = valid_idx;
    end
    
    %% Initialize accuracy matrices for feedback tasks (nIter x nTime_fb)
    accTask_fb = cell(nTasks_fb,1);
    for tsk = 1:nTasks_fb
        accTask_fb{tsk} = zeros(nIter, nTime_fb);
        accTask_perm{tsk} = zeros(nIter, nTime_fb, nBoot);
    end
    
    %% Main decoding loop for feedback window
    for iter = 1:nIter
        for t = 1:nTime_fb
            % Determine window indices (using winSize=3, centered on t; adjust boundaries)
            if t == 1
                win = t:min(nTime_fb, t+1);
            elseif t == nTime_fb
                win = max(1, t-1):t;
            else
                win = t-1:t+1;
            end
            
            % Loop over each feedback task.
            for tsk = 1:nTasks_fb
                neurons_idx = valid_neurons_fb_tasks{tsk};
                nNeurons = length(neurons_idx);
                if nNeurons == 0
                    accTask_fb{tsk}(iter, t) = NaN;
                    continue;
                end
                
                % For each neuron, extract feature (mean firing rate over win from firingRates_fb)
                % and label for valid trials.
                neuronData = cell(nNeurons, 1);
                reqClasses = taskClasses_fb{tsk};
                nClasses = length(reqClasses);
                for nIdx = 1:nNeurons
                    neuron = data_by_neuron(neurons_idx(nIdx));
                    validIdx = (neuron.speed_fb ~= 0);
                    feat = mean(neuron.firingRates_fb(validIdx, win), 2);
                    switch tasks_fb{tsk}
                        case 'speed_fbest'
                            labs = neuron.speed_estfb(validIdx);
                        case 'feedback'
                            labs = neuron.feedback(validIdx);
                        case 'gs'
                            labs = neuron.gs_fb(validIdx);
                        case 'wl'
                            labs = neuron.wl_bin(validIdx);
                    end
                    feat = feat(validIdx);
                    neuronData{nIdx} = struct();
                    for c = 1:nClasses
                        currClass = reqClasses(c);
                        idxClass = find(labs == currClass);
                        if isempty(idxClass)
                            neuronData{nIdx}.(['class' num2str(currClass)]) = [];
                        else
                            % Split indices into training and testing.
                            nAvail = length(idxClass);
                            rp = randperm(nAvail);
                            nTrainAvail = max(1, floor(nAvail * trainRatio));
                            training_indices = idxClass(rp(1:nTrainAvail));
                            testing_indices = idxClass(rp(nTrainAvail+1:end));
                            if isempty(testing_indices)
                                testing_indices = training_indices;
                            end
                            nTrainFixed = round(trainRatio * nPseudo);
                            nTestFixed = nPseudo - nTrainFixed;
                            train_samples = feat(training_indices(randi(length(training_indices), nTrainFixed, 1)));
                            test_samples  = feat(testing_indices(randi(length(testing_indices), nTestFixed, 1)));
                            
                            neuronData{nIdx}.(['class' num2str(currClass)]).train = train_samples;
                            neuronData{nIdx}.(['class' num2str(currClass)]).test  = test_samples;
                        end
                    end
                end
                
                % Assemble pseudo-population for current feedback task and time bin.
                trainPseudo = [];
                testPseudo = [];
                trainLabels = [];
                testLabels = [];
                for c = 1:nClasses
                    currClass = reqClasses(c);
                    nTrainFixed = round(trainRatio * nPseudo);
                    nTestFixed = nPseudo - nTrainFixed;
                    pseudoTrainMat = nan(nTrainFixed, nNeurons);
                    pseudoTestMat  = nan(nTestFixed, nNeurons);
                    for nIdx = 1:nNeurons
                        dataField = ['class' num2str(currClass)];
                        samplesTrain = neuronData{nIdx}.(dataField).train;
                        samplesTest  = neuronData{nIdx}.(dataField).test;
                        pseudoTrainMat(:, nIdx) = samplesTrain;
                        pseudoTestMat(:, nIdx)  = samplesTest;
                    end
                    trainPseudo = [trainPseudo; pseudoTrainMat];
                    testPseudo  = [testPseudo; pseudoTestMat];
                    trainLabels = [trainLabels; repmat(currClass, nTrainFixed, 1)];
                    testLabels  = [testLabels; repmat(currClass, nTestFixed, 1)];
                end
                
                if isempty(trainPseudo) || isempty(testPseudo)
                    accTask_fb{tsk}(iter,t) = NaN;
                else
                    mdl = fitcecoc(trainPseudo, trainLabels, 'Learners', templateSVM('KernelFunction','linear'));
                    preds = predict(mdl, testPseudo);
                    accTask_fb{tsk}(iter,t) = mean(preds == testLabels);
                    
                    % permutation test
                    for boot = 1:nBoot
                        train_perm = trainLabels(randperm(length(trainLabels)));
                        mdl = fitcecoc(trainPseudo, train_perm, 'Learners', templateSVM('KernelFunction','linear'));
                        preds = predict(mdl, testPseudo);
                        accTask_perm{tsk}(iter,t,boot) = mean(preds == testLabels);
                    end  % end permutation loop
                end
            end  % end task loop for feedback
        end  % end time loop for feedback
    end  % end iteration loop for feedback
    
    % Average over iterations for each feedback task and store for current area.
    for tsk = 1:nTasks_fb
        decodingResults_fb{tsk,a} = nanmean(accTask_fb{tsk}, 1);
        
        for t = 1:nTime_fb
            perm_t = accTask_perm{tsk}(:, t, :);
            thresh = prctile(perm_t(:), (1-alpha)*100);
            sig_acc{tsk, a}(t) = decodingResults_fb{tsk, a}(t) > thresh;
        end
    end
end

%% Plotting for Feedback Window: One plot per feedback task comparing two areas.
for tsk = 1:nTasks_fb
    figure;
    hold on;
    taskName = tasks_fb{tsk};
    colors = lines(2);
    for a = 1:nAreas
        if ~isempty(decodingResults_fb{tsk,a})
            % Use the time vector from the first valid feedback neuron of the area.
            idx_fb = find([data_by_neuron.area]== areas(a) & arrayfun(@(x) all(~isnan(x.firingRates_fb(:))), data_by_neuron));
            if ~isempty(idx_fb)
                timeVector_fb = data_by_neuron(idx_fb(1)).time;
            else
                timeVector_fb = [];
            end
            plot(timeVector_fb, decodingResults_fb{tsk,a}, 'LineWidth', 3, 'Color', colors(a,:));
            mask = sig_acc{tsk, a} == 1;
            s = scatter(timeVector(mask), decodingResults_fb{tsk,a}(mask), 'ro', 'filled', 'HandleVisibility','off');
            s.SizeData = 70;
        end
    end
    plot(timeVector, repmat(1/length(taskClasses_fb{tsk}), ...
            length(timeVector)), 'g--', 'LineWidth', 3)
    legend({'MCC', 'LPFC', 'Chance'}, 'Location', 'best');
    set(gca, 'FontSize', 18)

    xlabel('Time');
    ylabel('Decoding Accuracy');
%     ylim([0 1]);
    switch taskName
        case 'speed_fbest'
            title('Progress rate Decoding (Feedback Window)');
        case 'feedback'
            title('Feedback Decoding (Feedback Window)');
        case 'gs'
            title('Gauge size decoding (Feedback Window)');
        case 'wl'
            title('Work length decoding (Feedback Window)');
    end
    grid on;
    hold off;
end
%%
save('results_rate_fb.mat', 'decodingResults_fb', 'sig_acc', 'accTask_perm')

% Plotting for Check Window: One plot per decoding task comparing MCC vs LPFC
% Style to match Fig 5a (Stoll et al., 2016): MCC black, LPFC blue,
% chance grey dashed, significance = bold line segments.

mccColor    = [0 0 0];
lpfcColor   = [0 0.4470 0.7410];    % MATLAB default blue, close to paper
chanceColor = 0.65 * [1 1 1];       % light grey

lwThin   = 2.0;
lwThick  = 4.5;   % significance thickness
lwChance = 1.8;

usePercent = true;  % Fig 5a uses % correct

for tsk = 1:nTasks_check
    figure('Color','w');
    hold on;

    taskName = tasks_check{tsk};

    % --- MCC (a=1) and LPFC (a=2) only ---
    for a = 1:min(2, nAreas)
        y = decodingResults_fb{tsk,a};
        if isempty(y), continue; end

        if usePercent
            yPlot = 100 * y(:);
        else
            yPlot = y(:);
        end

        sigMask = (sig_acc{tsk,a}(:) == 1);

        if a == 1
            col = mccColor;
        else
            col = lpfcColor;
        end

        plot_bold_sig(timeVector(:), yPlot, sigMask, col, lwThin, lwThick);
    end

    % --- Chance line (grey dashed) ---
    chance = 1 / numel(taskClasses_check{tsk});
    if usePercent, chance = 100 * chance; end
    plot(timeVector, chance * ones(size(timeVector)), '--', ...
        'Color', chanceColor, 'LineWidth', lwChance, 'HandleVisibility','off');

    % Labels
    xlabel('Time (s)');
    if usePercent
        ylabel('Per cent correct decoding (%)');
    else
        ylabel('Decoding accuracy');
    end
    title(task_title(taskName));

    % Axes cosmetics (paper-like)
    ax = gca;
    set(ax, ...
        'FontSize', 18, ...
        'FontName', 'Arial', ...
        'Box', 'off', ...
        'TickDir', 'out', ...
        'LineWidth', 1.5);
    grid off;

%     % In-plot labels instead of a legend 
%     yl = ylim;
%     x0 = timeVector(1) + 0.04 * (timeVector(end) - timeVector(1));
%     y0 = yl(2) - 0.06 * (yl(2) - yl(1));
%     text(x0, y0, 'MCC',  'Color', mccColor,  'FontWeight','bold', 'FontSize', 18);
%     text(x0, y0 - 0.10*(yl(2)-yl(1)), 'LPFC', 'Color', lpfcColor, 'FontWeight','bold', 'FontSize', 18);
    % --- Legend (outside the axes so it never overlaps the data) ---
    
    % Make two dummy handles so the legend stays clean even if plot_bold_sig
    % creates multiple line objects per area.
    hMCC  = plot(nan, nan, '-', 'Color', mccColor,  'LineWidth', lwThin, 'DisplayName','MCC');
    hLPFC = plot(nan, nan, '-', 'Color', lpfcColor, 'LineWidth', lwThin, 'DisplayName','LPFC');

    lgd = legend([hMCC hLPFC], {'MCC','LPFC'}, ...
        'Location','northeastoutside', 'Box','off');
    set(lgd, 'FontSize', 18, 'FontName', 'Arial');

    % Make room for the outside legend
    ax = gca;
    ax.Position(3) = ax.Position(3) * 0.80;
    hold off;
end
