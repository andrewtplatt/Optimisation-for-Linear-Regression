clear variables
frac = 0.7;
frac2 = 0.8;
lambda = 1e-3;
batch_sizes = [1, 5, 10, 20];
n_iterations = 10;     % Number of times to average for the final mean squared error
iteration_mae = zeros(n_iterations, 1);
maes = zeros(length(batch_sizes), 1);

% Get the dataset and split it up
D = get_dataset();
% Only use a small portion for testing
[D, ~] = random_split(D, 0.1);
[trainval_D, test_D] = random_split(D, frac);

% n_trainval_data = size(trainval_D, 1);
n_params = size(D, 2) - 1;

% Section size for deciding value of batch size
n_sections = 5;         % Number of splits to make to work out which value of lambda to use
section_mae = zeros(n_sections, 1);

% Get the different sections
[sect_5, sect_1] = random_split(trainval_D, 4/5);
[sect_5, sect_2] = random_split(sect_5, 3/4);
[sect_5, sect_3] = random_split(sect_5, 2/3);
[sect_5, sect_4] = random_split(sect_5, 1/2);
% Choose the best number for batch size
for i = 1:length(batch_sizes)
    for j = 1:n_sections
        if j == 1
            train_D = [sect_2; sect_3; sect_4; sect_5];
            val_D = sect_1;
        elseif j == 2
            train_D = [sect_1; sect_3; sect_4; sect_5];
            val_D = sect_2;
        elseif j == 3
            train_D = [sect_1; sect_2; sect_4; sect_5];
            val_D = sect_3;
        elseif j == 4
            train_D = [sect_1; sect_2; sect_3; sect_5];
            val_D = sect_4;
        else
            train_D = [sect_1; sect_2; sect_3; sect_4];
            val_D = sect_5;
        end
        % Get the w vector and calculate the error for this value of lambda
        w = smoothed_stochastic_l1_regression(train_D, lambda, batch_sizes(i));
        section_mae(j) = compute_mean_abs_error(val_D, w);
    end
    maes(i) = mean(section_mae);
end

% Choose the best value for batch size
[~, index] = min(maes);
iteration_w = zeros(n_params, n_iterations);
for i = 1:n_iterations
%     [trainval_D, test_D] = random_split(D, frac);
    iteration_w(:, i) = smoothed_stochastic_l1_regression(trainval_D, lambda, batch_sizes(index));
    iteration_mae(i) = compute_mean_abs_error(test_D, iteration_w(:, i));
end
w = mean(iteration_w, 2);
mae = mean(iteration_mae);

% Write the results to a file
formatSpec = "Using a lambda value of %g and batch size of %d we obtain a w vector:\n";
wSpec = "%.3g\n";
maeSpec = "Which results in a Mean Absolute Error of: %.3g";
fileID = fopen('smoothed_stochastic_l1_results.txt', 'w');
fprintf(fileID, formatSpec, lambda, batch_sizes(index));
fprintf(fileID, wSpec, w);
fprintf(fileID, maeSpec, mae);
fclose('all');
