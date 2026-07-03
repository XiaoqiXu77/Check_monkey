% function to extract firing rate in a vector of time intervals
% tmin and tmax are vectors of size (1, n_intervals)
function fr = firing_rate_interval(spiketimes, tmin, tmax)
    n_intervals = size(tmin, 2);
    smin = zeros(1, n_intervals);
    smax = zeros(1, n_intervals);
    for t = 1:n_intervals
        if spiketimes(1) < tmin(t) && spiketimes(end) > tmax(t)
            smin(t) = find(spiketimes >= tmin(t), 1);
            smax(t) = find(spiketimes > tmax(t), 1);
        else
            smin(t) = NaN;
        end
    end
    fr = (smax - smin)./(tmax - tmin);
end