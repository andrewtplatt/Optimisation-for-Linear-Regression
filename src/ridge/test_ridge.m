clear variables
addpath('..\shared');
addpath('..\errors');
% Define some parameters for use later.
frac = 0.7;

rng(0)
% For getting the best lambda
lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3];
n_sections = 5;         % Number of splits to make to work out which value of lambda to use
section_mse = zeros(n_sections, 1);
section_runtime = zeros(n_sections, 1);
lambda_mse = zeros(length(lambdas), 1);
lambda_sd = zeros(length(lambdas), 1);
lambda_runtime = zeros(length(lambdas), 1);
lambda_runtime_sd = zeros(length(lambdas), 1);

% Final data
n_iterations = 1000;     % Number of times to average for the final mean squared error
iteration_mse = zeros(n_iterations, 1);
iteration_runtimes = zeros(n_iterations, 1);

% Get our dataset and split it appropriately
D = get_dataset();
% Just use a portion of the dataset for testing purposes
% [D, ~] = random_split(D, 0.1);
[trainval_D, test_D] = random_split(D, frac);
n_params = size(D, 2) - 1;


% for i = 1:length(lambdas)
%     datasets = split_n(trainval_D, n_sections);
%     for j = 1:n_sections
%         % Get our datasets
%         val_D = datasets(:, :, j);
%         train_sets = datasets;
%         train_sets(:, :, j) = [];
%         train_D = permute(train_sets, [1 3 2]);
%         train_D = reshape(train_D, [], n_params+1, 1);
%         
%         % Remove padded empty rows
%         val_D = val_D(~all(val_D == 0, 2),:);
%         train_D = train_D(~all(train_D == 0, 2),:);
% 
%         % Get the w vector and calculate the error for this value of lambda
%         tic
%         w = ridge_regression(train_D, lambdas(i));
%         section_runtime(j) = toc;
%         section_mse(j) = compute_mean_squared_error(val_D, w);
%     end
%     lambda_mse(i) = mean(section_mse);
%     lambda_sd(i) = std(section_mse);
%     lambda_runtime(i) = mean(section_runtime);
%     lambda_runtime_sd(i) = std(section_runtime);
% end
% 
% 
% figure
% subplot(2,1,1)
% errorbar(lambdas, lambda_mse, lambda_sd)
% set(gca, 'XScale', 'log')
% xlabel('\lambda')
% ylabel('Mean Squared Error')
% subplot(2,1,2)
% errorbar(lambdas, lambda_runtime, lambda_runtime_sd)
% set(gca, 'XScale', 'log')
% xlabel('\lambda')
% ylabel('Runtime /s')

lambda = 1e-3;

% % [~, index] = min(mses);
iteration_w = zeros(n_params, n_iterations);
for i = 1:n_iterations
%     [trainval_D, test_D] = random_split(D, frac);
    tic
    iteration_w(:, i) = ridge_regression(trainval_D, lambda);
    iteration_runtimes(i) = toc;
    iteration_mse(i) = compute_mean_squared_error(test_D, iteration_w(:, i));
end
w = mean(iteration_w, 2);
mse = mean(iteration_mse);
sd = std(iteration_mse);
runtime = mean(iteration_runtimes);
runtime_sd = std(iteration_runtimes);

% Write the results to a file
formatSpec = "Using a lambda value of %g we obtain a w vector:\n";
wSpec = "%.3g\n";
mseSpec = "Which results in a Mean Squared Error of: %.3g\n";
stdSpec = "With standard deviation: %.3g\n";
runtimeSpec = "The mean runtime was: %.3g\n";
runtimeSDSpec = "With standard deviation: %.3g\n";
fileID = fopen('ridgeresults.txt', 'w');
fprintf(fileID, formatSpec, lambda);
fprintf(fileID, wSpec, w);
fprintf(fileID, mseSpec, mse);
fprintf(fileID, stdSpec, sd);
fprintf(fileID, runtimeSpec, runtime);
fprintf(fileID, runtimeSDSpec, runtime_sd);
fclose('all');

restoredefaultpath;