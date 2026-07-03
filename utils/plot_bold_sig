function plot_bold_sig(t, y, sigMask, col, lwThin, lwThick)
    % Base thin line across all time
    plot(t, y, 'Color', col, 'LineWidth', lwThin);

    % Overlay thick segments where significant (bold line segments)
    sigMask = sigMask(:) & ~isnan(y(:));
    if ~any(sigMask), return; end

    d = diff([false; sigMask; false]);
    starts = find(d == 1);
    ends   = find(d == -1) - 1;

    for k = 1:numel(starts)
        idx = starts(k):ends(k);
        plot(t(idx), y(idx), 'Color', col, 'LineWidth', lwThick);
    end
end
