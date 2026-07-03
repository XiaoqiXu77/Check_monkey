function [tau, lags_ms, ac_smooth, R2] = compute_tau_fontanier(spikeTrials)
% spikeTrials = cell array; each cell contains spike times (in seconds).

    % --- PARAMETERS (Fontanier et al.) ---
    maxOrder   = 100;
    maxLag_ms  = 1000;
    bin_ms     = 3.33;
    remove_ms  = 10;
    loessFrac  = 0.1;
    r2_thresh  = 0.6;      % <<< ADDED: fit quality threshold

    % ==============================================================
    % 1) Concatenate spike times across trials
    % ==============================================================
    allSpikes = [];
    for t = 1:numel(spikeTrials)
        s = spikeTrials{t};
        allSpikes = [allSpikes; s(:)];
    end
    allSpikes = sort(allSpikes);

    if numel(allSpikes) < 50
        tau = NaN; lags_ms = []; ac_smooth = []; R2 = NaN;
        return;
    end

    % ==============================================================
    % 2) Autocorrelogram via forward diffs
    % ==============================================================
    deltas = [];
    for i = 1:length(allSpikes)
        lastIdx = min(length(allSpikes), i + maxOrder);
        dt = allSpikes(i+1:lastIdx) - allSpikes(i);
        dt = dt(dt <= maxLag_ms/1000);
        deltas = [deltas; dt];
    end

    if isempty(deltas)
        tau = NaN; lags_ms = []; ac_smooth = []; R2 = NaN;
        return;
    end

    % ==============================================================
    % 3) Bin into AC density
    % ==============================================================
    bins = (0:bin_ms:maxLag_ms) / 1000;
    counts = histcounts(deltas, bins);
    centers_ms = ((bins(1:end-1) + bins(2:end)) / 2) * 1000;
    ac_raw = (counts / sum(counts)) / (bin_ms/1000);

    % ==============================================================
    % 4) Remove first 10 ms and LOESS smooth
    % ==============================================================
    valid = centers_ms >= remove_ms;
    xs = centers_ms(valid);
    ys = ac_raw(valid);

    ac_smooth = zeros(size(ac_raw));
    ac_smooth(valid) = simple_loess(xs, ys, loessFrac);

    % ==============================================================
    % 5) Peak detection
    % ==============================================================
    [~, peakIdx] = max(ac_smooth(valid));
    absIdx = find(valid,1,'first') + peakIdx - 1;

    if absIdx == find(valid,1,'first')
        rel = ac_smooth(valid);
        p = find(diff(sign(diff(rel))) < 0);
        if ~isempty(p)
            absIdx = find(valid,1,'first') + p(1);
        end
    end

    LAT_ms = centers_ms(absIdx);

    % ==============================================================
    % 6) Drop neurons showing dips / second peak
    % ==============================================================
    dipMask = centers_ms >= LAT_ms & centers_ms <= LAT_ms + 100;
    seg = ac_smooth(dipMask);

    if isempty(seg) || min(seg) < 0.75 * (max(ac_smooth) - min(ac_smooth))
        tau = NaN; lags_ms = centers_ms; R2 = NaN;
        return;
    end

    % ==============================================================
    % 7) Exponential fit
    % ==============================================================
    fitMask = centers_ms >= LAT_ms;
    t = centers_ms(fitMask)' - LAT_ms;
    y = ac_smooth(fitMask)';

    expfun = @(b,t) b(1).*exp(-t./b(2)) + b(3);
    b0 = [max(y) 200 min(y)];

    opts = optimset('Display','off','TolFun',1e-8,'TolX',1e-8);
    lb = [0 1e-3 0];
    ub = [Inf 5000 Inf];

    try
        b = lsqcurvefit(expfun, b0, t, y, lb, ub, opts);
        tau = b(2);

        % ============================
        % 8) FIT QUALITY CONTROL (R²)
        % ============================
        yhat = expfun(b, t);
        SSres = sum((y - yhat).^2);
        SStot = sum((y - mean(y)).^2);
        R2 = 1 - SSres / SStot;

        if R2 < r2_thresh
            tau = NaN;
        end

    catch
        tau = NaN;
        R2 = NaN;
    end

    lags_ms = centers_ms;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Toolbox-free LOESS function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y_smooth = simple_loess(x, y, frac)
    n = numel(x);
    span = max(3, ceil(frac * n));  % # of neighbors
    y_smooth = zeros(size(y));

    for i = 1:n
        % window around point
        idxStart = max(1, i - floor(span/2));
        idxEnd   = min(n, i + floor(span/2));
        idx = idxStart:idxEnd;

        % weighted local linear regression
        xi = x(idx)'; yi = y(idx)';

        w = (1 - ((xi - x(i)) / (xi(end)-xi(1))).^2).^2;
        w(w < 0) = 0;

        X = [ones(length(xi),1), xi - x(i)];
        W = diag(w);
        beta = (X' * W * X) \ (X' * W * yi);

        y_smooth(i) = beta(1); % fitted value at center (zero offset)
    end
end
