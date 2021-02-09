clear variables
addpath('..\shared');
addpath('..\errors');
frac = 0.7;
lambdas = [1e-3, 1e-2, 0.1, 1, 10];
n_iterations = 1000;     % Number of times to average for the final mean squared error
iteration_mae = zeros(n_iterations, 1);
maes = zeros(length(lambdas), 1);
sds = maes;
runtimes = maes;
runtime_sds = maes;

% Get the dataset and split it up
D = get_dataset();
% Only use a small portion for testing
% [D, ~] = random_split(D, 0.5);
rng(0)
[trainval_D, test_D] = random_split(D, frac);

% n_trainval_data = size(trainval_D, 1);
n_params = size(D, 2) - 1;

% Section size for deciding value of lambda
n_sections = 5;         % Number of splits to make to work out which value of lambda to use
section_mae = zeros(n_sections, 1);
section_runtime = section_mae;

%% Choose the best number for lambda
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
%         % Get the w vector and calculate the error for this value of lambda
%         tic
%         w = smoothed_l1_regression(train_D, lambdas(i), 1e-3);
%         section_runtime(j) = toc;
%         section_mae(j) = compute_mean_abs_error(val_D, w);
%     end
%     maes(i) = mean(section_mae);
%     sds(i) = std(section_mae);
%     runtimes(i) = mean(section_runtime);
%     runtime_sds(i) = std(section_runtime);
% end
%
% figure
% subplot(2,1,1)
% errorbar(lambdas, maes, sds)
% xlabel('\lambda')
% ylabel('Mean Absolute Error')
% grid
% set(gca, 'XScale', 'log');
% subplot(2,1,2)
% errorbar(lambdas, runtimes, runtime_sds)
% xlabel('\lambda')
% ylabel('Runtime /s')
% grid
% set(gca, 'XScale', 'log');


% % Choose the best value for lambda
% [~, index] = min(maes);
% lambda = lambdas(index);
lambda = 1e-3;

%% Find the best constant value of eta
% etas = [1e-3, 1e-2, 1e-1];
% eta_mae = zeros(size(etas));
% eta_sd = eta_mae;
% eta_runtime = eta_mae;
% eta_runtime_sd = eta_mae;
% for j = 1:length(etas)
%   eta = etas(j);
%   datasets = split_n(trainval_D, n_sections);
%   for i = 1:n_sections
%     % Get our datasets
%     val_D = datasets(:, :, j);
%     train_sets = datasets;
%     train_sets(:, :, j) = [];
%     train_D = permute(train_sets, [1 3 2]);
%     train_D = reshape(train_D, [], n_params+1, 1);
%
%     % Remove padded empty rows
%     val_D = val_D(~all(val_D == 0, 2),:);
%     train_D = train_D(~all(train_D == 0, 2),:);
%     tic
%     w = smoothed_l1_regression(trainval_D, lambda, eta);
%     section_runtime(i) = toc;
%     section_mae(i) = compute_mean_abs_error(test_D, w);
%   end
%   eta_mae(j) = mean(section_mae);
%   eta_sd(j) = std(section_mae);
%   eta_runtime(j) = mean(section_runtime);
%   eta_runtime_sd(j) = std(section_runtime);
% end
%
% figure
% subplot(2,1,1)
% errorbar(etas, eta_mae, eta_sd)
% xlabel('\eta')
% ylabel('Mean Absolute Error')
% grid
% set(gca, 'XScale', 'log');
% subplot(2,1,2)
% errorbar(etas, eta_runtime, eta_runtime_sd)
% xlabel('\eta')
% ylabel('Runtime /s')
% grid
% set(gca, 'XScale', 'log');

eta = 1e-1;


%% Final test
iteration_w = zeros(n_params, n_iterations);
iteration_runtime = zeros(n_iterations, 1);

for i = 1:n_iterations
  tic
  iteration_w(:, i) = smoothed_l1_regression(trainval_D, lambda, eta);
  iteration_runtime(i) = toc;
  iteration_mae(i) = compute_mean_abs_error(test_D, iteration_w(:, i));
  disp(iteration_mae(i));  
end

w = mean(iteration_w, 2);
mae = mean(iteration_mae);
sd = std(iteration_mae);
runtime = mean(iteration_runtime);
runtime_sd = std(iteration_runtime);


% Write the results to a file
formatSpec = "Using a lambda value of %g we obtain a w vector:\n";
wSpec = "%.3g\n";
maeSpec = "Which results in a Mean Absolute Error of: %.3g";
stdSpec = "With standard deviation: %.3g\n";
runtimeSpec = "In a mean time of %.2g s\n";
runtimeSDSpec = "With standard deviation: %.2g s\n";
fileID = fopen('smoothed_l1_results.txt', 'w');
fprintf(fileID, formatSpec, lambda);
fprintf(fileID, wSpec, w);
fprintf(fileID, maeSpec, mae);
fprintf(fileID, stdSpec, sd);
fprintf(fileID, runtimeSpec, runtime);
fprintf(fileID, runtimeSDSpec, runtime_sd);
fclose('all');

restoredefaultpath;