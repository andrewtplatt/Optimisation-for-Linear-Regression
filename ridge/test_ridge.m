% Define some parameters for use later.
frac = 0.6;
frac2 = 0.8;
lambdas = [1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3];
n_iterations = 100;
iteration_mse = zeros(n_iterations, 1);
mses = zeros(length(lambdas), 1);

% Get our dataset and split it appropriately
D = get_dataset();
[trainval_D, test_D] = random_split(D, frac);

for i = 1:length(lambdas)
    for j = 1:n_iterations
        [train_D, val_D] = random_split(trainval_D, frac2);
        % Get the w vector and calculate the error for this value of lambda
        w = ridge_regression(train_D, lambdas(i));
        iteration_mse(j) = compute_mean_squared_error(val_D, w);
    end
    mses(i) = mean(iteration_mse);
end

[~, index] = min(mses);
w = ridge_regression(trainval_D, lambdas(index))
mse = compute_mean_squared_error(test_D, w)

