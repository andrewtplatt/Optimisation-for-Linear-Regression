% Returns the best estimate parameters w to fit the training data to the
% costs.
function w = lsq_regression(train_D)
    [n_data, n_params] = size(train_D);
    % Solve as Aw = b
    A = zeros(n_params-1);
    b = zeros(n_params-1, 1);
    % Get A and b
    for i = 1:n_data
        result = train_D(i,end);
        params = train_D(i,1:end-1)';   % Parameter values as a column vector
        A = A + params*params';         % Outer product
        b = b + result*params;
    end
    % Compute w
    w = A\b;
end