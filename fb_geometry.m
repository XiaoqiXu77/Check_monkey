%% ============================================================
%% Outputs:
%%  Fig1: Accuracy matrices (Observed + Null mean) for MCC & LPFC
%%  Fig2: Dot plot Racc = mean(offdiag A)/mean(diag A) vs null + MCC>LPFC
%%  Fig3: Dot plot Rcos = mean_{i!=j} cos(d(i),d(j)) vs null + MCC>LPFC
%%        where d(s) = v+(s) - v-(s) is the feedback axis at speed s
%%
%% Null model:
%%   - Accuracy: mismatch unit identity across speeds at test time (unit-shuffle per speed)
%%   - Cosine: permute unit identity independently per speed (breaks cross-speed alignment)
%% MCC>LPFC comparison:
%%   - matched neuron count via subsampling, one-sided test "MCC > LPFC"
%% ============================================================

%% ------------------------
%% USER SETTINGS
%% ------------------------
requiredSpeeds = [1 2 3 4];
areas     = [1 2];
areaNames = {'MCC','LPFC'};

winBins   = 1:20;  % ONLY window
winLabel  = 'tmin idx 1:20 (-1 to 0s pre-fb)';

nPseudo   = 100;
nBoot     = 50;
nPerm     = 2000;      % big for stable nulls
trainFrac = 0.8;

% matched-N subsampling for MCC>LPFC tests
nSubsample = 300;

% --- Plot style (publication-ish) ---
FS_AX  = 26;
FS_TIT = 30;
FS_LAB = 26;

% Observed dot styling (MCC blue, LPFC orange)
colObs = { [0 0.45 0.74], [0.85 0.33 0.10] };

msNull = 14;    % null dot size
msObs  = 260;   % observed dot size (big + colored)
lwObs  = 1.8;

nPlotNull = 250; % show only this many null points (stats still use all nPerm)

% Accuracy heatmap colorbar range
climA = [0.5 1.0];

rng(0,'twister');

outDir = 'fb_onewin_accuracy_cosmetric_pubplots';
if ~exist(outDir,'dir'), mkdir(outDir); end

%% ------------------------
%% Sanity checks
%% ------------------------
nTime_fb = size(data_by_neuron(1).firingRates_fb, 2);
if max(winBins) > nTime_fb
    error('winBins exceeds available fb bins. max(winBins)=%d, available=%d.', max(winBins), nTime_fb);
end

fprintf('Window: %s | bins %d..%d | tmin %.4g..%.4g\n', ...
    winLabel, min(winBins), max(winBins), tmin(min(winBins)), tmin(max(winBins)));

%% ============================================================
%% Compute per-area: accuracy matrices + null; cosine metric + null
%% ============================================================
full = repmat(struct(), numel(areas), 1);

for a = 1:numel(areas)
    areaId = areas(a);
    [valid_neurons, nN] = get_valid_neurons_feedback(data_by_neuron, areaId, requiredSpeeds);
    fprintf('%s: %d valid sessions\n', areaNames{a}, nN);

    pools = build_pools_fb(data_by_neuron, valid_neurons, requiredSpeeds, winBins);

    % ---- Accuracy matrix + null (unit identity mismatch across speeds) ----
    [A_obs, A_null_mean, acc] = decode_matrix_and_null(pools, requiredSpeeds, nPseudo, nBoot, nPerm, trainFrac);

    % ---- Cosine metric of feedback axis d(s)=v+(s)-v-(s) across speeds + null ----
    [Rcos_obs, Rcos_null, p_Rcos] = cosine_diff_metric_and_null(pools, requiredSpeeds, nPseudo, nBoot, nPerm);

    full(a).areaName = areaNames{a};
    full(a).nNeurons = nN;

    full(a).A_obs    = A_obs;
    full(a).A_null   = A_null_mean;
    full(a).acc      = acc;            % Racc_obs, Racc_null, p_Racc

    full(a).Rcos_obs  = Rcos_obs;
    full(a).Rcos_null = Rcos_null;
    full(a).p_Rcos    = p_Rcos;

    fprintf('  ACC: Racc=%.3f | p(obs>null)=%.6g\n', acc.Racc_obs, acc.p_Racc);
    fprintf('  COS: Rcos=%.3f | p(obs>null)=%.6g\n', Rcos_obs, p_Rcos);
end

%% ============================================================
%% FIG 1: Accuracy matrices (Observed + Null mean) MCC & LPFC
%% ============================================================
fig1 = figure('Color','w','Position',[100 100 1200 800]);
nRows = 2; nCols = 2;

for a = 1:numel(areas)
    % Observed
    ax = subplot(nRows,nCols,a);
    imagesc(ax, full(a).A_obs);
    axis(ax,'image');
    caxis(ax, climA);
    cb = colorbar(ax); set(cb,'FontSize',FS_AX);
    set(ax,'FontSize',FS_AX,'LineWidth',1.6,'TickDir','out');
    xlabel(ax,'test speed','FontSize',FS_LAB);
    ylabel(ax,'train speed','FontSize',FS_LAB);
    title(ax, sprintf('%s Observed (N=%d)', full(a).areaName, full(a).nNeurons), 'FontSize',FS_TIT);
    set_speed_ticks(ax, requiredSpeeds);

    % Null mean
    ax2 = subplot(nRows,nCols,a+2);
    imagesc(ax2, full(a).A_null);
    axis(ax2,'image');
    caxis(ax2, climA);
    cb2 = colorbar(ax2); set(cb2,'FontSize',FS_AX);
    set(ax2,'FontSize',FS_AX,'LineWidth',1.6,'TickDir','out');
    xlabel(ax2,'test speed','FontSize',FS_LAB);
    ylabel(ax2,'train speed','FontSize',FS_LAB);
    title(ax2, sprintf('%s Null mean', full(a).areaName), 'FontSize',FS_TIT);
    set_speed_ticks(ax2, requiredSpeeds);
end

annotation(fig1, 'textbox', [0 0.96 1 0.04], ...
    'String', sprintf('Cross-speed feedback decoding accuracy (%s)', winLabel), ...
    'EdgeColor','none', 'HorizontalAlignment','center', 'FontWeight','bold', 'FontSize',FS_TIT);

set(fig1,'PaperPositionMode','auto');
print(fig1, fullfile(outDir,'Fig1_accuracy_matrices_obs_and_null.png'), '-dpng', '-r300');
close(fig1);

%% ============================================================
%% Matched-N MCC > LPFC (one-sided) for Racc and Rcos
%% ============================================================
[vn1, n1] = get_valid_neurons_feedback(data_by_neuron, areas(1), requiredSpeeds);
[vn2, n2] = get_valid_neurons_feedback(data_by_neuron, areas(2), requiredSpeeds);
Nmatch = min(n1,n2);
fprintf('Matched-N tests: Nmatch=%d | subsamples=%d\n', Nmatch, nSubsample);

Racc_sub = nan(nSubsample,2);
Rcos_sub = nan(nSubsample,2);

for r = 1:nSubsample
    s1 = vn1(randperm(n1, Nmatch));
    s2 = vn2(randperm(n2, Nmatch));

    pools1 = build_pools_fb(data_by_neuron, s1, requiredSpeeds, winBins);
    pools2 = build_pools_fb(data_by_neuron, s2, requiredSpeeds, winBins);

    % Accuracy metric observed-only
    A1 = decode_matrix_observed_only(pools1, requiredSpeeds, nPseudo, nBoot, trainFrac);
    A2 = decode_matrix_observed_only(pools2, requiredSpeeds, nPseudo, nBoot, trainFrac);
    Racc_sub(r,1) = accuracy_generalizability_metric(A1);
    Racc_sub(r,2) = accuracy_generalizability_metric(A2);

    % Cosine metric observed-only
    Rcos_sub(r,1) = cosine_diff_metric_observed_only(pools1, requiredSpeeds, nPseudo, nBoot);
    Rcos_sub(r,2) = cosine_diff_metric_observed_only(pools2, requiredSpeeds, nPseudo, nBoot);
end

p_area_acc = one_sided_sign_p_greater(Racc_sub(:,1) - Racc_sub(:,2)); % MCC > LPFC
p_area_cos = one_sided_sign_p_greater(Rcos_sub(:,1) - Rcos_sub(:,2)); % MCC > LPFC

fprintf('\n=== MCC > LPFC one-sided tests (matched N=%d, subsamples=%d) ===\n', Nmatch, nSubsample);
fprintf('Accuracy generalizability (Racc): p = %.6g\n', p_area_acc);
fprintf('Cosine feedback-axis similarity (Rcos): p = %.6g\n\n', p_area_cos);

fid = fopen(fullfile(outDir,'stats_summary.txt'),'w');
fprintf(fid, 'MCC > LPFC one-sided tests (matched N=%d, subsamples=%d)\n', Nmatch, nSubsample);
fprintf(fid, 'Racc (offdiag/diag accuracy): p = %.12g\n', p_area_acc);
fprintf(fid, 'Rcos (mean cos(d(i),d(j))): p = %.12g\n', p_area_cos);
fclose(fid);

%% ============================================================
%% FIG 2: Dot plot for accuracy Racc vs null + MCC > LPFC
%% ============================================================
fig2 = figure('Color','w','Position',[100 100 950 700]);
ax = axes(fig2); hold(ax,'on');
set(ax,'FontSize',FS_AX,'LineWidth',1.6,'TickDir','out');
ylabel(ax,'R_{acc} = mean(offdiag A) / mean(diag A)','FontSize',FS_LAB);

xpos = [1 2];

for a = 1:numel(areas)
    R_obs = full(a).acc.Racc_obs;
    R_all = full(a).acc.Racc_null(:);

    % downsample null points for plotting only
    nShow = min(nPlotNull, numel(R_all));
    idxShow = randperm(numel(R_all), nShow);
    R_null = R_all(idxShow);

    jitter = 0.12;
    xs = xpos(a) + (rand(size(R_null))-0.5)*2*jitter;
    scatter(ax, xs, R_null, msNull, [0.65 0.65 0.65], 'filled');

    % observed big colored dot
    scatter(ax, xpos(a), R_obs, msObs, ...
        'MarkerFaceColor', colObs{a}, 'MarkerEdgeColor','k', 'LineWidth', lwObs);

    % star only above observed
    star = pstar(full(a).acc.p_Racc);
    yTop = max([R_obs; R_null]) + 0.03;
    text(ax, xpos(a), yTop, star, 'HorizontalAlignment','center', 'FontSize',FS_TIT-2);
end

% bracket with ONLY "MCC > LPFC ..."
yMaxAll = max([full(1).acc.Racc_obs; full(1).acc.Racc_null(:); full(2).acc.Racc_obs; full(2).acc.Racc_null(:)]);
yBar = yMaxAll + 0.12;
h = 0.02;
plot(ax, [xpos(1) xpos(1) xpos(2) xpos(2)], [yBar-h yBar yBar yBar-h], 'k-', 'LineWidth',2);

% Put text ABOVE the bar (so it doesn't look crossed out)
yTxt = yBar + 0.035;
text(ax, mean(xpos), yTxt, sprintf('MCC > LPFC %s', pstar(p_area_acc)), ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize',FS_TIT);

xlim(ax, [0.5 2.5]);
xticks(ax, xpos);
xticklabels(ax, areaNames);
title(ax, sprintf('Accuracy generalizability vs null (%s)', winLabel), 'FontSize',FS_TIT);
box(ax,'off');

ylim(ax, [min(ylim(ax)) yTxt+0.06]);

set(fig2,'PaperPositionMode','auto');
print(fig2, fullfile(outDir,'Fig2_Racc_dotplot.png'), '-dpng', '-r300');
close(fig2);

%% ============================================================
%% FIG 3: Dot plot for cosine metric Rcos vs null + MCC > LPFC
%% ============================================================
fig3 = figure('Color','w','Position',[100 100 950 700]);
ax = axes(fig3); hold(ax,'on');
set(ax,'FontSize',FS_AX,'LineWidth',1.6,'TickDir','out');
ylabel(ax,'R_{cos} = mean_{i\neq j} cos(d(i), d(j)),  d(s)=v_{+}(s)-v_{-}(s)', 'FontSize',FS_LAB);

for a = 1:numel(areas)
    R_obs = full(a).Rcos_obs;
    R_all = full(a).Rcos_null(:);

    % downsample null points for plotting only
    nShow = min(nPlotNull, numel(R_all));
    idxShow = randperm(numel(R_all), nShow);
    R_null = R_all(idxShow);

    jitter = 0.12;
    xs = xpos(a) + (rand(size(R_null))-0.5)*2*jitter;
    scatter(ax, xs, R_null, msNull, [0.65 0.65 0.65], 'filled');

    % observed big colored dot
    scatter(ax, xpos(a), R_obs, msObs, ...
        'MarkerFaceColor', colObs{a}, 'MarkerEdgeColor','k', 'LineWidth', lwObs);

    % star only above observed
    star = pstar(full(a).p_Rcos);
    yTop = max([R_obs; R_null]) + 0.03;
    text(ax, xpos(a), yTop, star, 'HorizontalAlignment','center', 'FontSize',FS_TIT-2);
end

% bracket with ONLY label
yMaxAll = max([full(1).Rcos_obs; full(1).Rcos_null(:); full(2).Rcos_obs; full(2).Rcos_null(:)]);
yBar = yMaxAll + 0.12;
plot(ax, [xpos(1) xpos(1) xpos(2) xpos(2)], [yBar-h yBar yBar yBar-h], 'k-', 'LineWidth',2);

yTxt = yBar + 0.035;
text(ax, mean(xpos), yTxt, sprintf('MCC > LPFC %s', pstar(p_area_cos)), ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize',FS_TIT);

xlim(ax, [0.5 2.5]);
xticks(ax, xpos);
xticklabels(ax, areaNames);
title(ax, sprintf('Cosine geometry of feedback axis vs null (%s)', winLabel), 'FontSize',FS_TIT);
box(ax,'off');

ylim(ax, [min(ylim(ax)) yTxt+0.06]);

set(fig3,'PaperPositionMode','auto');
print(fig3, fullfile(outDir,'Fig3_Rcos_dotplot.png'), '-dpng', '-r300');
close(fig3);

%% Save results
save(fullfile(outDir,'results_onewin_accuracy_cosmetric.mat'), ...
    'full','Racc_sub','Rcos_sub','p_area_acc','p_area_cos', ...
    'winBins','requiredSpeeds','areas','areaNames','Nmatch', ...
    'nPseudo','nBoot','nPerm','trainFrac','nSubsample');

fprintf('Done. Saved figures + MAT + stats_summary.txt to: %s\n', outDir);


%% ============================================================
%% Helper functions
%% ============================================================

function set_speed_ticks(ax, requiredSpeeds)
    n = numel(requiredSpeeds);
    xticks(ax, 1:n); yticks(ax, 1:n);
    xticklabels(ax, arrayfun(@num2str, requiredSpeeds, 'UniformOutput', false));
    yticklabels(ax, arrayfun(@num2str, requiredSpeeds, 'UniformOutput', false));
end

function star = pstar(p)
    if p < 1e-3
        star = '***';
    elseif p < 1e-2
        star = '**';
    elseif p < 5e-2
        star = '*';
    else
        star = 'n.s.';
    end
end

function p = one_sided_sign_p_greater(d)
    % One-sided sign-style p for H1: median(d) > 0
    p = (1 + sum(d <= 0)) / (numel(d) + 1);
end

function [valid_neurons, nN] = get_valid_neurons_feedback(data_by_neuron, areaId, requiredSpeeds)
    idx_area = find([data_by_neuron.area] == areaId);
    valid_neurons = [];

    for i = idx_area
        fr = data_by_neuron(i).firingRates_fb;
        if any(isnan(fr(:))), continue; end

        spd = data_by_neuron(i).speed_estfb(:);
        fb  = data_by_neuron(i).feedback(:);

        ok = true;
        for s = requiredSpeeds
            idx_s = find(spd == s);
            if isempty(idx_s), ok = false; break; end
            if ~all(ismember([1 2], unique(fb(idx_s)))), ok = false; break; end
        end
        if ok
            valid_neurons(end+1) = i; %#ok<AGROW>
        end
    end
    nN = numel(valid_neurons);
end

function pools = build_pools_fb(data_by_neuron, valid_neurons, requiredSpeeds, winBins)
    % pools{n,s,1} = fb==1 samples, pools{n,s,2} = fb==2 samples
    nN = numel(valid_neurons);
    nSpeed = numel(requiredSpeeds);
    pools = cell(nN, nSpeed, 2);

    for n = 1:nN
        idx = valid_neurons(n);

        X = data_by_neuron(idx).firingRates_fb(:, winBins);
        x = mean(X, 2);

        spd = data_by_neuron(idx).speed_estfb(:);
        fb  = data_by_neuron(idx).feedback(:);

        for sIdx = 1:nSpeed
            sVal = requiredSpeeds(sIdx);
            pools{n,sIdx,1} = x((spd==sVal) & (fb==1));
            pools{n,sIdx,2} = x((spd==sVal) & (fb==2));
        end
    end
end

function R = accuracy_generalizability_metric(A)
    n = size(A,1);
    offMask = ~eye(n);
    DA = mean(diag(A));
    OG = mean(A(offMask));
    R  = OG / max(DA, eps);
end

function [A_obs, A_null_mean, acc] = decode_matrix_and_null(pools, requiredSpeeds, nPseudo, nBoot, nPerm, trainFrac)
    nSpeed = numel(requiredSpeeds);
    nN = size(pools,1);

    permUnits = zeros(nPerm, nSpeed, nN);
    invPerm   = zeros(nPerm, nSpeed, nN);
    for p = 1:nPerm
        for sIdx = 1:nSpeed
            q = randperm(nN);
            permUnits(p,sIdx,:) = q;
            invPerm(p,sIdx,:)   = invert_perm(q);
        end
    end

    acc_boot = nan(nSpeed,nSpeed,nBoot);
    acc_perm = nan(nSpeed,nSpeed,nPerm,nBoot);

    for b = 1:nBoot
        pop = cell(nSpeed,1);

        for sIdx = 1:nSpeed
            X1tr = zeros(nPseudo,nN); X1te = zeros(nPseudo,nN);
            X2tr = zeros(nPseudo,nN); X2te = zeros(nPseudo,nN);

            for n = 1:nN
                d1 = pools{n,sIdx,1};
                d2 = pools{n,sIdx,2};

                [tr, te] = split_train_test_indices_safe(numel(d1), trainFrac);
                X1tr(:,n) = d1(randsample(tr,nPseudo,true));
                X1te(:,n) = d1(randsample(te,nPseudo,true));

                [tr, te] = split_train_test_indices_safe(numel(d2), trainFrac);
                X2tr(:,n) = d2(randsample(tr,nPseudo,true));
                X2te(:,n) = d2(randsample(te,nPseudo,true));
            end

            Xtr = [X1tr; X2tr];
            ytr = [ones(nPseudo,1); 2*ones(nPseudo,1)];
            Xte = [X1te; X2te];
            yte = [ones(nPseudo,1); 2*ones(nPseudo,1)];

            pop{sIdx} = struct('Xtr',Xtr,'ytr',ytr,'Xte',Xte,'yte',yte);
        end

        models = cell(nSpeed,1);
        for trainIdx = 1:nSpeed
            models{trainIdx} = fitcsvm(pop{trainIdx}.Xtr, pop{trainIdx}.ytr, ...
                'KernelFunction','linear', 'Standardize', true);
        end

        % Observed matrix
        A = nan(nSpeed,nSpeed);
        for trainIdx = 1:nSpeed
            for testIdx = 1:nSpeed
                yhat = predict(models{trainIdx}, pop{testIdx}.Xte);
                A(trainIdx,testIdx) = mean(yhat == pop{testIdx}.yte);
            end
        end
        acc_boot(:,:,b) = A;

        % Null: mismatch unit identity across speeds in TEST relative to TRAIN
        for p = 1:nPerm
            Ap = nan(nSpeed,nSpeed);
            for trainIdx = 1:nSpeed
                inv_train = squeeze(invPerm(p,trainIdx,:))';
                for testIdx = 1:nSpeed
                    p_test = squeeze(permUnits(p,testIdx,:))';
                    compPerm = p_test(inv_train); % composition
                    Xte_null = pop{testIdx}.Xte(:, compPerm);
                    yhat = predict(models{trainIdx}, Xte_null);
                    Ap(trainIdx,testIdx) = mean(yhat == pop{testIdx}.yte);
                end
            end
            acc_perm(:,:,p,b) = Ap;
        end
    end

    A_obs = mean(acc_boot,3);
    A_null_mean = mean(mean(acc_perm,4),3);

    Racc_obs = accuracy_generalizability_metric(A_obs);

    Racc_null = nan(nPerm,1);
    for p = 1:nPerm
        A0 = mean(acc_perm(:,:,p,:),4);
        Racc_null(p) = accuracy_generalizability_metric(A0);
    end

    acc.Racc_obs  = Racc_obs;
    acc.Racc_null = Racc_null;
    acc.p_Racc    = (1 + sum(Racc_null >= Racc_obs)) / (nPerm + 1); % one-sided obs>null
end

function A_obs = decode_matrix_observed_only(pools, requiredSpeeds, nPseudo, nBoot, trainFrac)
    nSpeed = numel(requiredSpeeds);
    nN = size(pools,1);
    acc_boot = nan(nSpeed,nSpeed,nBoot);

    for b = 1:nBoot
        pop = cell(nSpeed,1);

        for sIdx = 1:nSpeed
            X1tr = zeros(nPseudo,nN); X1te = zeros(nPseudo,nN);
            X2tr = zeros(nPseudo,nN); X2te = zeros(nPseudo,nN);

            for n = 1:nN
                d1 = pools{n,sIdx,1};
                d2 = pools{n,sIdx,2};

                [tr, te] = split_train_test_indices_safe(numel(d1), trainFrac);
                X1tr(:,n) = d1(randsample(tr,nPseudo,true));
                X1te(:,n) = d1(randsample(te,nPseudo,true));

                [tr, te] = split_train_test_indices_safe(numel(d2), trainFrac);
                X2tr(:,n) = d2(randsample(tr,nPseudo,true));
                X2te(:,n) = d2(randsample(te,nPseudo,true));
            end

            Xtr = [X1tr; X2tr];
            ytr = [ones(nPseudo,1); 2*ones(nPseudo,1)];
            Xte = [X1te; X2te];
            yte = [ones(nPseudo,1); 2*ones(nPseudo,1)];

            pop{sIdx} = struct('Xtr',Xtr,'ytr',ytr,'Xte',Xte,'yte',yte);
        end

        models = cell(nSpeed,1);
        for trainIdx = 1:nSpeed
            models{trainIdx} = fitcsvm(pop{trainIdx}.Xtr, pop{trainIdx}.ytr, ...
                'KernelFunction','linear', 'Standardize', true);
        end

        A = nan(nSpeed,nSpeed);
        for trainIdx = 1:nSpeed
            for testIdx = 1:nSpeed
                yhat = predict(models{trainIdx}, pop{testIdx}.Xte);
                A(trainIdx,testIdx) = mean(yhat == pop{testIdx}.yte);
            end
        end
        acc_boot(:,:,b) = A;
    end

    A_obs = mean(acc_boot,3);
end

function [Rcos_obs, Rcos_null, p_Rcos] = cosine_diff_metric_and_null(pools, requiredSpeeds, nPseudo, nBoot, nPerm)
    % d(s) = v+(s)-v-(s) across neurons, then Rcos = mean offdiag cosine(d(i),d(j))
    nSpeed = numel(requiredSpeeds);
    nN = size(pools,1);
    offMask = ~eye(nSpeed);

    permUnits = zeros(nPerm, nSpeed, nN);
    for p = 1:nPerm
        for sIdx = 1:nSpeed
            permUnits(p,sIdx,:) = randperm(nN);
        end
    end

    Rboot = zeros(nBoot,1);
    Rperm_sum = zeros(nPerm,1);

    for b = 1:nBoot
        D = zeros(nSpeed, nN); % rows=speeds, cols=neurons
        for sIdx = 1:nSpeed
            for n = 1:nN
                dneg = pools{n,sIdx,1}; % fb==1
                dpos = pools{n,sIdx,2}; % fb==2
                mneg = mean(dneg(randsample(1:numel(dneg), nPseudo, true)));
                mpos = mean(dpos(randsample(1:numel(dpos), nPseudo, true)));
                D(sIdx,n) = (mpos - mneg);
            end
        end

        C = cosine_matrix(D);
        Rboot(b) = mean(C(offMask));

        for p = 1:nPerm
            Dp = D;
            for sIdx = 1:nSpeed
                q = squeeze(permUnits(p,sIdx,:))';
                Dp(sIdx,:) = Dp(sIdx,q);
            end
            Cp = cosine_matrix(Dp);
            Rperm_sum(p) = Rperm_sum(p) + mean(Cp(offMask));
        end
    end

    Rcos_obs  = mean(Rboot);
    Rcos_null = Rperm_sum / nBoot;
    p_Rcos    = (1 + sum(Rcos_null >= Rcos_obs)) / (nPerm + 1); % one-sided obs>null
end

function Rcos = cosine_diff_metric_observed_only(pools, requiredSpeeds, nPseudo, nBoot)
    nSpeed = numel(requiredSpeeds);
    nN = size(pools,1);
    offMask = ~eye(nSpeed);

    vals = zeros(nBoot,1);
    for b = 1:nBoot
        D = zeros(nSpeed, nN);
        for sIdx = 1:nSpeed
            for n = 1:nN
                dneg = pools{n,sIdx,1};
                dpos = pools{n,sIdx,2};
                mneg = mean(dneg(randsample(1:numel(dneg), nPseudo, true)));
                mpos = mean(dpos(randsample(1:numel(dpos), nPseudo, true)));
                D(sIdx,n) = (mpos - mneg);
            end
        end
        C = cosine_matrix(D);
        vals(b) = mean(C(offMask));
    end
    Rcos = mean(vals);
end

function C = cosine_matrix(V)
    n = size(V,1);
    C = zeros(n,n);
    for i = 1:n
        for j = 1:n
            C(i,j) = cosine_sim(V(i,:), V(j,:));
        end
    end
end

function c = cosine_sim(x,y)
    x = x(:); y = y(:);
    c = (x' * y) / (norm(x)*norm(y) + eps);
end

function invp = invert_perm(p)
    invp = zeros(size(p));
    invp(p) = 1:numel(p);
end

function [train_idx, test_idx] = split_train_test_indices_safe(n, fracTrain)
    if n <= 0
        error('Empty condition cell encountered.');
    elseif n == 1
        train_idx = 1; test_idx = 1; return;
    end
    idx = randperm(n);
    nTrain = round(fracTrain*n);
    nTrain = max(1, min(n-1, nTrain));
    train_idx = idx(1:nTrain);
    test_idx  = idx(nTrain+1:end);
end