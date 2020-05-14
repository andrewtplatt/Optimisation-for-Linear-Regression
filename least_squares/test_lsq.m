% Get our dataset
D = get_dataset();
n_vars = size(D, 2) - 1;

% Define the different values of frac we'll be testing
step_size = 0.1;
fracs = 0.2:step_size:0.9;

% Define how many times we should repeat for each frac, and create storage
% vectors for the values
n_iters = 10;
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
    mean_mse(i) = mean(mean_sq_errors);
    sd_mse(i) = std(mean_sq_errors);
    mean_w(:,i) = mean(ws, 2);
    sd_w(:,i) = std(ws, 0, 2);
    
    % Tabulate the statistics for w
    mean_column = [round(mean_w(:,i), 4); NaN];
    sd_column = [round(sd_w(:,i), 4); NaN];
    frac_column = [zeros(length(mean_column)-1, 1) + frac; NaN];
    var_names = {'Frac', 'Mean', 'SD'};
    row_names = {'CRIM'; 'ZN'; 'INDUS'; 'CHAS'; 'NOX'; 'RM'; 'AGE'; 'DIS'; 'RAD'; 'TAX'; 'PTRATIO'; 'B'; 'LSTAT'; '_'};
    w_stats = table(frac_column, mean_column, sd_column, ...
        'VariableNames', var_names, 'RowNames', row_names);
    
    % Write the table to file
    writetable(w_stats, 'lsq_stats.xlsx', 'FileType', 'spreadsheet', 'WriteMode', 'append', 'WriteRowNames',true)
end

