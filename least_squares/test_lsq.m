% Get our dataset
D = get_dataset();
n_vars = size(D, 2) - 1;

% Define the different values of frac we'll be testing
step_size = 0.1;
fracs = 0.2:step_size:0.9;

% Define how many times we should repeat for each frac, and create storage
% vectors for the values
n_iters = 1000;
mean_sq_errors = zeros(n_iters, 1);
ws = zeros(n_vars, n_iters);

% And the statistics for these:
mean_mse = zeros(length(fracs), 1);
sd_mse = mean_mse;
mean_w = zeros(n_vars, length(fracs));
sd_w = mean_w;

% Run the tests
for i = 1:length(fracs)
    frac = fracs(i);
    for j = 1:n_iters
       [train_D, test_D] = random_split(D, frac); 
       ws(:,j) = lsq_regression(train_D);
       mean_sq_errors(j) = compute_mean_squared_error(test_D, ws(:,j));
    end
    % Calculate the statistics
    mean_mse(i) = mean(mean_sq_errors, 'omitnan');
    sd_mse(i) = std(mean_sq_errors, 'omitnan');
    mean_w(:,i) = mean(ws, 2, 'omitnan');
    sd_w(:,i) = std(ws, 0, 2, 'omitnan');
    
    % Tabulate the statistics for w
    mean_column = [round(mean_w(:,i), 4); NaN];
    sd_column = [round(sd_w(:,i), 4); NaN];
    frac_column = [zeros(n_vars, 1) + frac; NaN];
    var_names = {'Frac', 'Mean', 'SD'};
    row_names = {'CRIM'; 'ZN'; 'INDUS'; 'CHAS'; 'NOX'; 'RM'; 'AGE'; 'DIS'; ...
        'RAD'; 'TAX'; 'PTRATIO'; 'B'; 'LSTAT'; '_'};
    w_stats = table(frac_column, mean_column, sd_column, ...
        'VariableNames', var_names, 'RowNames', row_names);
    
    % Write the table to file
    if i == 1 
        write_mode = 'overwrite';   % Clear old data if this is the first value of frac
    else
        write_mode = 'append';      % Otherwise add the data to the bottom of the spreadsheet
    end
    writetable(w_stats, 'w_stats.xlsx', 'FileType', 'spreadsheet', ...
        'WriteMode', write_mode, 'WriteRowNames',true)
end

% Tabulate the statistics for error
error_stats = table(fracs', round(mean_mse, 4), round(sd_mse, 4), ...
    'VariableNames', {'Frac', 'Mean', 'SD'});

% Write the table to file
writetable(error_stats, 'error_stats.xlsx', 'FileType', 'spreadsheet', 'WriteMode', 'overwrite')



