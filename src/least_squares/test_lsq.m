clear variables
addpath('..\shared');
addpath('..\errors');
% Get our dataset
rng(0)
D = get_dataset();
% Just use a portion of the dataset for testing purposes
% [D, ~] = random_split(D, 0.1);
n_params = size(D, 2) - 1;

% Define the different values of frac we'll be testing
step_size = 0.01;
fracs = 0.2:step_size:0.9;

% Define how many times we should repeat for each frac, and create storage
% vectors for the values
n_iterations = 1000;
mean_sq_errors = zeros(n_iterations, 1);
runtimes = zeros(n_iterations, 1);
ws = zeros(n_params, n_iterations);

% And the statistics for these:
mean_mse = zeros(length(fracs), 1);
sd_mse = mean_mse;
mean_w = zeros(n_params, length(fracs));
sd_w = mean_w;
mean_runtime = zeros(length(fracs), 1);
sd_runtime = mean_runtime;

% % Run the tests
% for i = 1:length(fracs)
%     frac = fracs(i);
%     for j = 1:n_iters
%        [train_D, test_D] = random_split(D, frac);
%        tic
%        ws(:,j) = lsq_regression(train_D);
%        runtimes(j) = toc;
%        mean_sq_errors(j) = compute_mean_squared_error(test_D, ws(:,j));
%     end
%     % Calculate the statistics
%     mean_mse(i) = mean(mean_sq_errors);
%     sd_mse(i) = std(mean_sq_errors);
%     mean_w(:,i) = mean(ws, 2);
%     sd_w(:,i) = std(ws, 0, 2);
%     mean_runtime(i) = mean(runtimes);
%     sd_runtime(i) = std(runtimes);
    
%     % Tabulate the statistics for w
%     mean_column = [round(mean_w(:,i), 4); NaN];
%     sd_column = [round(sd_w(:,i), 4); NaN];
%     frac_column = [zeros(n_vars, 1) + frac; NaN];
%     var_names = {'Frac', 'Mean', 'SD'};
%     w_stats = table(frac_column, mean_column, sd_column, ...
%         'VariableNames', var_names);
%     
%     % Write the table to file
%     if i == 1 
%         write_mode = 'overwrite';   % Clear old data if this is the first value of frac
%     else
%         write_mode = 'append';      % Otherwise add the data to the bottom of the spreadsheet
%     end
%     writetable(w_stats, 'w_stats.xlsx', 'FileType', 'spreadsheet', ...
%         'WriteMode', write_mode, 'WriteRowNames',true)
% end

% Tabulate the statistics for error
% error_stats = table(fracs', round(mean_mse, 4), round(sd_mse, 4), ...
%     'VariableNames', {'Frac', 'Mean', 'SD'});

% Write the table to file
% writetable(error_stats, 'error_stats.xlsx', 'FileType', 'spreadsheet', 'WriteMode', 'overwrite');
% p = polyfit(fracs, mean_mse, 2);
% pp = p(3) + p(2).*fracs + p(1).*fracs.^2;
% 
% r = polyfit(fracs, mean_runtime, 2);
% rr = r(3) + r(2).*fracs + r(1).*fracs.^2;
% 
% meanplusstd = mean_mse + sd_mse;
% meanminusstd = mean_mse - sd_mse;
% meanplusstd2 = mean_runtime + sd_runtime;
% meanminusstd2 = mean_runtime - sd_runtime;
% 
% figure
% subplot(2,1,1)
% plot(fracs, mean_mse);
% hold on
% f = fill( [fracs fliplr(fracs)],  [meanplusstd' fliplr(meanminusstd')], 'b');
% alpha(.1)
% set(f, 'edgecolor', 'none')
% % errorbar(fracs, mean_mse, sd_mse)
% 
% plot(fracs, pp)
% xlabel('Fraction of dataset used for training')
% ylabel('Mean Cost')
% 
% 
% subplot(2,1,2)
% plot(fracs, mean_runtime)
% hold on
% f = fill( [fracs fliplr(fracs)],  [meanplusstd2' fliplr(meanminusstd2')], 'b');
% set(f, 'edgecolor', 'none')
% alpha(.1)
% plot(fracs, rr)
% xlabel('Fraction of dataset used for training')
% grid
% ylabel('Mean Runtime')

[train_D, test_D] = random_split(D, 0.7);
iteration_mse = zeros(n_iterations, 1);
iteration_runtimes = zeros(n_iterations, 1);
iteration_w = zeros(n_params, n_iterations);
for i = 1:n_iterations
%     [trainval_D, test_D] = random_split(D, frac);
    tic
    iteration_w(:, i) = lsq_regression(train_D);
    iteration_runtimes(i) = toc;
    iteration_mse(i) = compute_mean_squared_error(test_D, iteration_w(:, i));
end
w = mean(iteration_w, 2);
mse = mean(iteration_mse);
sd = std(iteration_mse);
runtime = mean(iteration_runtimes);
runtime_sd = std(iteration_runtimes);
% w = ridge_regression(trainval_D, lambda);
% mse = compute_mean_squared_error(test_D, w);

% Write the results to a file
formatSpec = "Using a lambda value of %g we obtain a w vector:\n";
wSpec = "%.3g\n";
mseSpec = "Which results in a Mean Squared Error of: %.3g\n";
stdSpec = "With standard deviation: %.3g\n";
runtimeSpec = "The mean runtime was: %.3g\n";
runtimeSDSpec = "With standard deviation: %.3g\n";
fileID = fopen('lsqresults.txt', 'w');
fprintf(fileID, formatSpec, 0);
fprintf(fileID, wSpec, w);
fprintf(fileID, mseSpec, mse);
fprintf(fileID, stdSpec, sd);
fprintf(fileID, runtimeSpec, runtime);
fprintf(fileID, runtimeSDSpec, runtime_sd);
fclose('all');

restoredefaultpath;
