% Get our dataset
D = get_dataset();

% Define the different values of frac we'll be testing
step_size = 0.1;
fracs = 0.2:step_size:0.9;

% Define how many times we should repeat for each frac, and create storage
% vectors for the values
n_iters = 1000;
mean_sq_errors = zeros(length(fracs), n_iters);

% And the statistics for these:
mean_mse = zeros(length(fracs), 1);
sd_mse = mean_mse;

% Run the tests
for i = 1:length(fracs)
    for j = 1:n_iters
       [train_D, test_D] = random_split(D, fracs(i)); 
       w = lsq_regression(train_D);
       mean_sq_errors(i, j) = compute_mean_squared_error(test_D, w);
    end
    % Calculate the statistics
    mean_mse(i) = mean(mean_sq_errors(i,:));
    sd_mse(i) = std(mean_sq_errors(i,:));
end



