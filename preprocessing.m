clc
clear

neuro_path = '/Users/nils/Documents/Check_mk/NEURONS';
files = dir(fullfile(neuro_path, '*.mat'));
load('neuro_info.mat');

n_monkeys = 2;
monkey_names = ["A", "H"];
n_areas = 2;  % 1 -- MCC and 2 -- LPFC
needed_cor = [14, 21, 30, 36];

% sliding window of 200 ms with 150 ms overlap
length_win = 200;
step = 50;

% periodes of interest: work/check decision, feedback, gauge reveal
% corresponding event codes: 160, 65/165, 151
% each window includes 1000 ms before stimulus onset to 1000 ms after
before = 1000;
after = 1000;
% creat a vector of time bins for one window
tmin = -before:step:(after - length_win);
tmax = tmin + length_win;

% convert from ms to s
tmin = tmin./1000;
tmax = tmax./1000;

%% make a matrix containing all the information
% (monkey id, block id, trial id, event index of start of trial, check 0 / cor 1 / inc 2, 
%  current gauge size, cumulative correct trials, RT, condition (1-4), progress, unit id
%  extra working trials before getting the bonus (only for full gauge trials)
%  time of dual lever onset, time of feedback, time of gauge reveal, area)

n_features = 15;
bad_unit = [];
behav_all = [];
neuro_all = [];
error_code = [235, 252, 253, 254, 250, 251];

for monkey = 1:n_monkeys
    
    monkey_name = monkey_names(monkey);
    behav_monkey = [];
    area_monkey = [];
    TF = startsWith({files.name}, monkey_name);
    n_files = size(TF, 2);
    
    firing_rates_choice = [];
    firing_rates_fb = [];
    firing_rates_gauge = [];
    firing_rates_choiceRes = [];
    spikeCounts_tr = {};
    spikeTimes = {};
    
    for id_unit = 1:n_files
        if TF(id_unit)
            file_name = files(id_unit).name;
            load(file_name)
            behav_session = [];
            
            % determine area MCC/LPFC
            if find(neuro_info.unit == file_name)
                area = neuro_info.area(neuro_info.unit == file_name);
            else
                continue
            end
            
            if isfield(SPK, 'originalevent')
                BEHAV.event = SPK.originalevent;
            else
                BEHAV.event = SPK.event;
            end
            
            % find start/end of good (check/work) trials
            start_trial = find(BEHAV.event.data == 100);
            end_trial = find(BEHAV.event.data == 101);
            size_diff = size(start_trial, 1) - size(end_trial, 1);
            switch size_diff
                case 1
                    start_trial(end) = [];
                case -1
                    end_trial(1) = [];
                case 0
                    if start_trial(1) > end_trial(1)
                        start_trial(end) = [];
                        end_trial(1) = [];
                    end
                otherwise
                    disp("error epoching")
                    fprintf('unit %d\n', id_unit)
                    fprintf('different of size: %d\n', size_diff)
                    bad_unit = [bad_unit, id_unit];
                    continue
            end
            
            area_monkey = [area_monkey, area];
            
            % drop short trials
            nb_evt_tr = end_trial - start_trial + 1;
            start_trial = start_trial(nb_evt_tr >= 7);
            end_trial = end_trial(nb_evt_tr >= 7);
            n_trial = size(start_trial, 1);
            
            cum_cor = 0;
            flag_full = 0;
            block = 1;        
            new_block = 1;
            for trial = 1:n_trial
                % drop trial which contains error
                if sum(ismember(error_code, BEHAV.event.data(start_trial(trial):end_trial(trial)))) > 0
                    continue
                end

                % get current gauge size in range 1-7
                if monkey == 1
                    gauge_size = 216 - BEHAV.event.data(start_trial(trial) + 2);
                else
                    gauge_size = 216 - BEHAV.event.data(start_trial(trial) + 3);
                end 
                
                if new_block
                    begin_size = gauge_size;
                    new_block = 0;
                end
                
                behav_trial = zeros(1, n_features);
                behav_trial(1) = monkey;
                behav_trial(2) = block;
                behav_trial(3) = trial;
                behav_trial(4) = start_trial(trial);
                behav_trial(11) = id_unit;
                behav_trial(12) = 0;
                behav_trial(13) = 0;
                behav_trial(14) = 0;
                behav_trial(15) = 0;
                
                if gauge_size == 7 && flag_full == 0
                    [~, condition] = min(abs(needed_cor - cum_cor));
    %                     % compute the slope to determine the condition
    %                     condition = ceil((cum_cor + 1)/(7 - begin_size)) - 2;
                    cum_cor_full = cum_cor;
                    flag_full = 1;
                end
                
                % find the dual lever ON
                idx_dual_lever_on = find(BEHAV.event.data(start_trial(trial):end) == 160, 1);
                if ~isempty(idx_dual_lever_on)
                    behav_trial(13) = BEHAV.event.timestamp(start_trial(trial) + idx_dual_lever_on - 1); 
                    fr_temp = firing_rate_interval(SPK.clipped.timestamp, behav_trial(13)+tmin, behav_trial(13)+tmax);
                    firing_rates_choice = [firing_rates_choice; fr_temp];
                    
                    choice_res_t = BEHAV.event.timestamp(start_trial(trial) + idx_dual_lever_on); 
                    fr_temp = firing_rate_interval(SPK.clipped.timestamp, choice_res_t+tmin, choice_res_t+tmax);
                    firing_rates_choiceRes = [firing_rates_choiceRes; fr_temp];
                    
                    switch BEHAV.event.data(start_trial(trial) + idx_dual_lever_on)
                        case 66
                            touch = find(BEHAV.event.data(start_trial(trial):end) == 74, 1);
                            target_on = find(BEHAV.event.data(start_trial(trial):end) == 80, 1);
                            behav_trial(8) = BEHAV.event.timestamp(start_trial(trial)+touch-1) ...
                                 - BEHAV.event.timestamp(start_trial(trial)+target_on-1);
                             
                            if ismember(65, BEHAV.event.data(start_trial(trial):end_trial(trial)))
                                cum_cor = cum_cor + 1;
                                behav_trial(5) = 1;
                                behav_trial(14) = BEHAV.event.timestamp(start_trial(trial) + touch);
                            else
                                behav_trial(5) = 2;
                                behav_trial(14) = BEHAV.event.timestamp(start_trial(trial) + touch);
                            end
                            firing_rates_fb = [firing_rates_fb; ...
                                firing_rate_interval(SPK.clipped.timestamp, behav_trial(14)+tmin, behav_trial(14)+tmax)];
                        case 150
                            behav_trial(5) = 0;
                            behav_trial(15) = BEHAV.event.timestamp(...
                                start_trial(trial) + idx_dual_lever_on + 1);
                            firing_rates_gauge = [firing_rates_gauge; ...
                                firing_rate_interval(SPK.clipped.timestamp, behav_trial(15)+tmin, behav_trial(15)+tmax)];
                            if gauge_size == 7
                                % only take full blocks
                                if begin_size == 1
                                    behav_trial(9) = condition;
                                    behav_trial(10) = 1;
                                    behav_trial(12) = cum_cor - cum_cor_full;
                                    % fill recursively for the whole block
                                    stepback = 0;
                                    while size(behav_session, 1) > stepback
                                        if behav_session(end-stepback, 2) == block
                                            behav_session(end-stepback, 9) = condition;
                                            behav_session(end-stepback, 10) = ...
                                                min(behav_session(end-stepback, 7)/needed_cor(condition), 1);
                                        else
                                            break
                                        end
                                        stepback = stepback + 1;
                                    end
                                end
                                
                                cum_cor = 0;
                                flag_full = 0;
                                block = block + 1;
                                new_block = 1;
                            end
                        otherwise
                            firing_rates_choice(end, :) = [];
                            continue  % skip error trials
                    end
                else
                    continue  % skip error trials
                end
                
                behav_trial(6) = gauge_size;    
                behav_trial(7) = cum_cor;               
                behav_session = cat(1, behav_session, behav_trial);
                
                t_start_trial = BEHAV.event.timestamp(start_trial(trial));
                t_end_trial = BEHAV.event.timestamp(end_trial(trial));
%                 disp(fprintf("start at %.2f, end at %.2f\n", t_start_trial, t_end_trial));
                spikeCounts_tr{end+1} = spike_count(SPK.clipped.timestamp, t_start_trial, t_end_trial);
                spikeTimes{end+1} = SPK.timestamp(find(SPK.timestamp >= t_start_trial & SPK.timestamp < t_end_trial));
            end
            
            behav_monkey = cat(1, behav_monkey, behav_session);
            
            if size(firing_rates_choice, 1) ~= size(behav_monkey, 1)
                disp('error firing rates')
                return
            end
                        
        else
            continue
        end        
    end
    behav_all{monkey} = behav_monkey;
    neuro_all{monkey}.area = area_monkey;
    neuro_all{monkey}.firing_rates_choice = firing_rates_choice;   
    neuro_all{monkey}.firing_rates_fb = firing_rates_fb;
    neuro_all{monkey}.firing_rates_gauge = firing_rates_gauge;
    neuro_all{monkey}.firing_rates_choiceRes = firing_rates_choiceRes;
    neuro_all{monkey}.spikeCounts_trial = spikeCounts_tr;
    neuro_all{monkey}.spikeTimes = spikeTimes;
end

%% adjust id_unit for the second monkey
behav_all{2}(:, 11) = behav_all{2}(:, 11) - min(behav_all{2}(:, 11)) + 1;

%% add columns to behav matrix
% get gauge size at previous check (for all trials)
for monkey = 1:2
    behav = behav_all{monkey};
    idx_trials = find(behav(:, 9) ~= 0 & behav(:, 5) == 0);
    curr_check = behav(idx_trials, 6);    
    for trial = 2:size(behav, 1)
        idx_pre_check = find(behav(1:trial-1, 5) == 0);
        if isempty(idx_pre_check)
            continue
        else
            idx_last_check = idx_pre_check(end);
            gauge_last_check = behav(idx_last_check, 6);
            if gauge_last_check ~= 7
                behav(trial, 16) = gauge_last_check;
            else
                behav(trial, 16) = 0;
            end
        end
    end
    behav_all{monkey} = behav;
end

% add work length to all trials
for monkey = 1:n_monkeys
    behav = behav_all{monkey};   
    n_trials = size(behav, 1);
    consec = 0;
    for trial = 1:n_trials
        if behav(trial, 5) == 1
            consec = consec + 1;
        elseif behav(trial, 5) == 0
            behav(trial, 17) = consec;
            consec = 0;
            continue
        end
        behav(trial, 17) = consec;
    end
    
    behav_all{monkey} = behav;
end

% mark the first checks for each block
n_checks = 10;  % max number of first few checks to be marked
for monkey = 1:n_monkeys
    behav = behav_all{monkey};
    current_block = 0;
    n_trials = size(behav, 1);
    for trial = 1:n_trials
        if behav(trial, 2) == current_block
            continue
        else
            % mark the beginning of a block           
            behav(trial, 18) = -1;
            current_block = behav(trial, 2);
            
            idx_check = find(behav(trial:end, 5) == 0);
            
            for i = 1:min(n_checks, length(idx_check))
                idx = idx_check(i);
                if behav(trial+idx-1, 2) == current_block
                    behav(trial+idx-1, 18) = i;
                else
                    break
                end
            end
                    
        end
    end
    behav_all{monkey} = behav;
end

% numerate blocks (20th col)
for monkey = 1:n_monkeys
    behav = behav_all{monkey};
    
    current_block = 1;
    n_trials = size(behav, 1);
    behav(1, 20) = current_block;
    for trial = 2:n_trials
        if behav(trial, 2) == behav(trial - 1, 2)
            behav(trial, 20) = current_block;
        else
            current_block = current_block + 1;
            behav(trial, 20) = current_block;
        end
    end
    
    behav_all{monkey} = behav;
end

%% --- Data Organization from behav_all and neuro_all ---

% Preallocate an empty structure array for efficiency.
% This structure will contain data for each neuron.
data_by_neuron = struct('neuronID', [], 'monkey', [], 'area', [], ...
    'firingRates_ch', [], 'choice', [], 'firingRates_fb', [],'feedback', [], ...
    'speed_ch', [], 'speed_fb', [], 'gauge_size', [], 'pre_gs', [], ...
    'gs_fb', [], 'wl', [], 'wl_fb', [], 'id_trial', [], 'time', [], ...
    'rt', [], 'rt_ch', [], 'pe', [], 'speed_est', [], 'gs_est', [], ...
    'speed_estfb', [], 'gs_estfb', [], 'pe_label', []);

ctr = 1;
for monkey = 1:n_monkeys
    behav = behav_all{monkey};  % behavioral matrix for current monkey
    neuro = neuro_all{monkey};  % neural data structure for current monkey
    for area = 1:n_areas
        % Get indices of units for the given area.
        idx_unit_area = find(neuro.area == area);
        n_units = length(idx_unit_area);
        
        for u = 1:n_units
            id_unit = idx_unit_area(u);
            % Find trials for this unit (using column 11 for matching neuron ID)
            idx_trials = find(behav(:,11) == id_unit + min(behav(:,11)) - 1);
            idx_work_trials = find(behav(behav(:, 5) ~= 0,11) == id_unit + min(behav(:,11)) - 1);
            idx_trials_work = find(behav(:,11) == id_unit + min(behav(:,11)) - 1 & behav(:, 5) ~= 0);
            
            % Extract task variables from behav matrix:
            choice = behav(idx_trials,5) ~= 0;      % 1 for work, 0 for check
            speed_ch = behav(idx_trials,9);       
            gauge_size = behav(idx_trials,6);         
            pre_gs = behav(idx_trials,16);           
            wl = behav(idx_trials,17);
            speed_est = behav(idx_trials,21);
            pe = behav(idx_trials,22);
            pe_label = behav(idx_trials,24);
            gs_est = behav(idx_trials,23);
            rt_ch = behav(idx_trials,8);
            nb_re = behav(idx_trials,7);
            
            fb = behav(idx_trials_work,5);
            speed_fb = behav(idx_trials_work,9);
            speed_estfb = behav(idx_trials_work,21);
            gs_fb = behav(idx_trials_work,6);
            gs_estfb = behav(idx_trials_work,23);
            wl_fb = behav(idx_trials_work,17);   
            rt = behav(idx_trials_work,8);
            
            % Remove trials with 0 speed immediately:
            validIdx = (speed_ch ~= 0);
            validIdx_work = behav(idx_trials_work,9)~=0;
            if ~any(validIdx)
                continue;  % skip this unit if no valid trial
            end
            choice = choice(validIdx);
            speed_ch = speed_ch(validIdx);
            gauge_size = gauge_size(validIdx);
            pre_gs = pre_gs(validIdx);
            wl = wl(validIdx);
            speed_est = speed_est(validIdx);
            gs_est = gs_est(validIdx);
            pe = pe(validIdx);
            pe_label = pe_label(validIdx);
            rt_ch = rt_ch(validIdx);
            nb_re = nb_re(validIdx);
            
            fb = fb(validIdx_work);
            speed_fb = speed_fb(validIdx_work);
            gs_fb = gs_fb(validIdx_work);
            wl_fb = wl_fb(validIdx_work);
            rt = rt(validIdx_work);
            speed_estfb = speed_estfb(validIdx_work);
            gs_estfb = gs_estfb(validIdx_work);
            
            % Get firing rates for these trials
            y_ch = neuro.firing_rates_choice(idx_trials, :);
            y_ch = y_ch(validIdx, :);
            y_fb = neuro.firing_rates_fb(idx_work_trials, :);
            y_fb = y_fb(validIdx_work, :);
%             y = neuro.firing_rates_trial(idx_trials, :);
            
            % Skip this unit if not enough trials
            if size(y_fb, 1) < 5
                continue;
            end
            
            % Store the data in the structure:
            data_by_neuron(ctr).neuronID = id_unit;
            data_by_neuron(ctr).monkey = monkey;
            data_by_neuron(ctr).area = area;
            data_by_neuron(ctr).firingRates_ch = y_ch;  % [nTrials x nTimeBins]
            data_by_neuron(ctr).firingRates_fb = y_fb;  % [nTrials x nTimeBins]
%             data_by_neuron(ctr).spikeCounts_tr = y;  % [nTrials x nTimeBins]
            data_by_neuron(ctr).choice = choice;    % [nTrials x 1]
            data_by_neuron(ctr).feedback = fb;    % [nTrials x 1]
            data_by_neuron(ctr).speed_ch = speed_ch;      % [nTrials x 1]
            data_by_neuron(ctr).speed_fb = speed_fb;      % [nTrials x 1]
            data_by_neuron(ctr).speed_est = speed_est;      % [nTrials x 1]
            data_by_neuron(ctr).speed_estfb = speed_estfb;
            data_by_neuron(ctr).gs_est = gs_est;
            data_by_neuron(ctr).gs_estfb = gs_estfb;
            data_by_neuron(ctr).pe = pe;
            data_by_neuron(ctr).pe_label = pe_label;
            data_by_neuron(ctr).rt_ch = rt_ch;
            data_by_neuron(ctr).gauge_size = gauge_size;
            data_by_neuron(ctr).gs_fb = gs_fb;
            data_by_neuron(ctr).pre_gs = pre_gs;
            data_by_neuron(ctr).wl = wl;
            data_by_neuron(ctr).wl_fb = wl_fb;
%             data_by_neuron(ctr).id_trial = [1:size(choice,1)]';
            data_by_neuron(ctr).time = tmin;
            data_by_neuron(ctr).rt = rt;
            data_by_neuron(ctr).nb_re = nb_re;

            data_by_neuron(ctr).checks = behav(idx_trials(validIdx), 18);
            data_by_neuron(ctr).block  = behav(idx_trials(validIdx), 2);
            
            % time constant 
            % Get spike times for these trials
            y = neuro.spikeTimes(idx_trials);
            data_by_neuron(ctr).spikeTimes  = y;
            
            % compute autocorrelogram TAU for this neuron
            [tau, lags, rvals] = compute_tau_fontanier(y);
            data_by_neuron(ctr).tau   = tau;
            data_by_neuron(ctr).lags  = lags;
            data_by_neuron(ctr).rvals = rvals;   % smoothed autocorrelogram
            
            ctr = ctr + 1;
        end
    end
end

%%
%% prepare data for RNN
behav = behav_all{1};

% Ensure all variables exist and are column vectors
fb = behav(:, 5);
gs = behav(:, 6);
prev_gs = behav(:, 16);
session = behav(:, 11);

% % Replace modular 0 with 7
% prev_gs(prev_gs == 0) = 7;

% --- Filter invalid gauge values (gs or prev_gs outside 1–7) ---
valid_idx = (gs >= 1 & gs <= 7) & (prev_gs >= 0 & prev_gs <= 6);

% Keep only valid rows
fb       = fb(valid_idx);
gs       = gs(valid_idx);
prev_gs  = prev_gs(valid_idx);
session  = session(valid_idx);

fprintf('Removed %d invalid trials (%.2f%% of total)\n', ...
        sum(~valid_idx), 100 * sum(~valid_idx) / numel(valid_idx));

% --- Now save the cleaned data ---
save('gauge_data.mat', 'fb', 'gs', 'prev_gs', 'session');


% Sanity check
N = numel(fb);
assert(numel(gs) == N && numel(prev_gs) == N && numel(session) == N, ...
    'All vectors must have the same length.');

% Save
save('gauge_data.mat', 'fb', 'gs', 'prev_gs', 'session');
