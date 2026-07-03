function [h, crit_p, adj_p, sorted_p] = fdr_bh(pvals, q, method, report)
    % Benjamini & Hochberg (1995) FDR control
    % (Adapted version works in MATLAB 2018a)

    if nargin < 2, q = 0.05; end
    if nargin < 3, method = 'pdep'; end
    if nargin < 4, report = 'no'; end

    p = pvals(:);
    [sorted_p, sort_ids] = sort(p);
    m = length(p);
    adj_p = zeros(m,1);

    if strcmp(method, 'pdep')
        % Benjamini–Hochberg
        thresh = (1:m)'/m * q;
    elseif strcmp(method, 'dep')
        % Benjamini–Yekutieli
        c = sum(1./(1:m));
        thresh = (1:m)'/m * q / c;
    else
        error('method must be ''pdep'' or ''dep''');
    end

    w = find(sorted_p <= thresh);
    if isempty(w)
        crit_p = 0;
        h = zeros(m,1);
    else
        crit_p = sorted_p(max(w));
        h = p <= crit_p;
    end

    % Adjusted p-values
    for i = 1:m
        adj_p(sort_ids(i)) = min(sorted_p(i)*m/i,1);
    end

    if strcmp(report,'yes')
        fprintf('FDR critical p = %g\n', crit_p);
    end
end
