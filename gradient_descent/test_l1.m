clear variables
frac = 0.6; 
frac2 = 0.8;
lambdas = [1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3];
n_iterations = 1000;     % Number of times to average for the final mean squared error
iteration_mae = zeros(n_iterations, 1);
maes = zeros(length(lambdas), 1);

% Get the dataset and split it up
D = get_dataset();
[trainval_D, test_D] = random_split(D, frac);
[train_D, val_D] = random_split(trainval_D, frac2);

n_trainval_data = size(trainval_D, 1);
n_params = size(D, 2) - 1;

% Section size for deciding value of lambda
n_sections = 5;         % Number of splits to make to work out which value of lambda to use
n_section_data = floor(n_trainval_data/n_sections);

% Choose the best number for lambda
for i = 1:length(lambdas)
    for j = 1:n_sections
        % Get which section indices are for training, and which are for
        % validation
        train_sections = mod((1:n_sections - 1) + j, n_sections);
        val_section = mod(n_sections + j, n_sections);
        % We want indices of n_sections, rather than the 0 the mod function
        % will give us.
        train_sections(~train_sections) = n_sections;
        train_sections = sort(train_sections);
        if ~val_section
            val_section = n_sections;    
        end
        % Get the actual row numbers needed
        train_rows = ((train_sections(1)-1)*n_section_data + 1):((train_sections(end))*n_section_data);
        val_rows = (val_section-1)*n_section_data + 1:(val_section-1)*n_section_data + n_section_data;
        % Get our datasets
        train_D = trainval_D(train_rows, :);
        val_D = trainval_D(val_rows, :);
        % Get the w vector and calculate the error for this value of lambda
        w = smoothed_l1_regression(train_D, lambdas(i));
        iteration_mae(j) = compute_mean_abs_error(val_D, w);
    end
    maes(i) = mean(iteration_mae);
end

[~, index] = min(maes);
iteration_w = zeros(n_params, n_iterations);
for i = 1:n_iterations
    [trainval_D, test_D] = random_split(D, frac);
    iteration_w(:, i) = smoothed_l1_regression(trainval_D, lambdas(index));
    iteration_mae(i) = compute_mean_abs_error(test_D, w);       % Reuse the same vector as before
end
w = mean(iteration_w, 2);
mae = mean(iteration_mae);

% Write the results to a file
formatSpec = "Using a lambda value of %g we obtain a w vector:\n";
wSpec = "%.3g\n";
maeSpec = "Which results in a Mean Absolute Error of: %.3g";
fileID = fopen('smoothed_l1_results.txt', 'w');
fprintf(fileID, formatSpec, lambdas(index));
fprintf(fileID, wSpec, w);
fprintf(fileID, maeSpec, mae);
fclose('all');
