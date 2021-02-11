clear variables
addpath('..\shared');
addpath('..\errors');
frac = 0.7;
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1];
batch_sizes = [1, 2, 4, 8, 16, 32, 64];
n_iterations = 50;     % Number of times to average for the final mean squared error
iteration_mae = zeros(n_iterations, 1);
iteration_runtime = iteration_mae;
maes = zeros(length(batch_sizes), 1);
sds = maes;
runtimes = maes;
runtime_sds = maes;
maes2 = zeros(length(lambdas), 1);
sds2 = maes2;
runtimes2 = maes2;
runtime_sds2 = maes2;

% Get the dataset and split it up
D = get_dataset();
rng(0)
% Only use a small portion for testing
% [D, ~] = random_split(D, 0.1);
[trainval_D, test_D] = random_split(D, frac);

% n_trainval_data = size(trainval_D, 1);
n_params = size(D, 2) - 1;

% Section size for deciding value of batch size
n_sections = 5;         % Number of splits to make to work out which value of lambda to use
n_repeats = 5;
section_mae = zeros(n_sections*n_repeats, 1);
section_runtime = section_mae;

% Choose the best number for batch size
% for i = 1:length(batch_sizes)
%   for k = 0:n_repeats - 1
%     datasets = split_n(trainval_D, n_sections);
%     for j = 1:n_sections
%       % Get our datasets
%       val_D = datasets(:, :, j);
%       train_sets = datasets;
%       train_sets(:, :, j) = [];
%       train_D = permute(train_sets, [1 3 2]);
%       train_D = reshape(train_D, [], n_params+1, 1);
%       
%       % Remove padded empty rows
%       val_D = val_D(~all(val_D == 0, 2),:);
%       train_D = train_D(~all(train_D == 0, 2),:);
%       % Get the w vector and calculate the error for this value of B
%       tic
%       w = smoothed_stochastic_l1_regression(train_D, 1e-3, batch_sizes(i));
%       section_runtime(j + k*n_sections) = toc;
%       section_mae(j + k*n_sections) = compute_mean_abs_error(val_D, w);
%     end
%   end
%   maes(i) = mean(section_mae);
%   sds(i) = std(section_mae);
%   runtimes(i) = mean(section_runtime);
%   runtime_sds(i) = std(section_runtime);
% end
% 
% figure
% subplot(2,1,1)
% errorbar(batch_sizes, maes, sds)
% xlabel('Batch Size')
% ylabel('Mean Absolute Error')
% set(gca, 'XScale', 'log');
% xlim([0.8 100]);
% grid
% subplot(2,1,2)
% errorbar(batch_sizes, runtimes, runtime_sds)
% xlabel('Batch Size')
% ylabel('Runtime /s')
% set(gca, 'XScale', 'log');
% xlim([0.8 100]);
% grid


% Choose the best value for batch size
% [~, b_index] = min(maes);
% B = batch_sizes(b_index);
B = 64;

% % Choose the best number for lambda using this batch size

% for i = 1:length(lambdas)
%   for k = 0:n_repeats - 1
%     datasets = split_n(trainval_D, n_sections);
%     for j = 1:n_sections
%       % Get our datasets
%       val_D = datasets(:, :, j);
%       train_sets = datasets;
%       train_sets(:, :, j) = [];
%       train_D = permute(train_sets, [1 3 2]);
%       train_D = reshape(train_D, [], n_params+1, 1);
% 
%       % Remove padded empty rows
%       val_D = val_D(~all(val_D == 0, 2),:);
%       train_D = train_D(~all(train_D == 0, 2),:);
%       % Get the w vector and calculate the error for this value of lambda
%       tic
%       w = smoothed_stochastic_l1_regression(train_D, lambdas(i), B);
%       section_runtime(j + k*n_sections) = toc;
%       section_mae(j + k*n_sections) = compute_mean_abs_error(val_D, w);
%     end
%   end
%   maes2(i) = mean(section_mae);
%   sds2(i) = std(section_mae);
%   runtimes2(i) = mean(section_runtime);
%   runtime_sds2(i) = std(section_runtime);
% end
% 
% figure
% subplot(2, 1, 1)
% errorbar(lambdas, maes2, sds2);
% set(gca, 'XScale', 'log');
% xlabel('\lambda');
% ylabel('Mean Absolute Error');
% grid
% xlim([6e-7 1.5e-1]);
% 
% subplot(2, 1, 2)
% errorbar(lambdas, runtimes2, runtime_sds2);
% set(gca, 'XScale', 'log');
% xlabel('\lambda');
% ylabel('Runtime /s');
% grid
% xlim([6e-7 1.5e-1]);


% Choose the best value for lambda
% [~, l_index] = min(maes2);
% lambda = lambdas(l_index);
lambda = 1e-3;


iteration_w = zeros(n_params, n_iterations);
for i = 1:n_iterations
%     [trainval_D, test_D] = random_split(D, frac);
    tic
    iteration_w(:, i) = smoothed_stochastic_l1_regression(trainval_D, lambda, B);
    iteration_runtime(i) = toc;
    iteration_mae(i) = compute_mean_abs_error(test_D, iteration_w(:, i));
end
w = mean(iteration_w, 2);
mae = mean(iteration_mae);
sd = std(iteration_mae);
runtime = mean(iteration_runtime);
runtime_sd = std(iteration_runtime);
% 
% % Write the results to a file
% formatSpec = "Using a lambda value of %g and batch size of %d we obtain a w vector:\n";
% wSpec = "%.3g\n";
% maeSpec = "Which results in a Mean Absolute Error of: %.3g\n";
% stdSpec = "With standard deviation: %.3g\n";
% runtimeSpec = "In a mean time of %.2g s\n";
% runtimeSDSpec = "With standard deviation: %.2g s\n";
% fileID = fopen('smoothed_stochastic_l1_results_momentum.txt', 'w');
% fprintf(fileID, formatSpec, lambda, B);
% fprintf(fileID, wSpec, w);
% fprintf(fileID, maeSpec, mae);
% fprintf(fileID, stdSpec, sd);
% fprintf(fileID, runtimeSpec, runtime);
% fprintf(fileID, runtimeSDSpec, runtime_sd);
% fclose('all');

restoredefaultpath;
