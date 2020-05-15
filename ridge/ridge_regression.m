% Returns the best estimate parameters w to fit the training data to the
% costs.
function w = ridge_regression(train_D, lambda)
    [n_data, n_params] = size(train_D);
    
    % Get the ridge offset to penalise large terms
    ridge = n_data * lambda/2 * eye(n_params - 1);
    
    % Solve as Aw = b
    A = zeros(n_params-1);
    b = zeros(n_params-1, 1);
    
    % Get A and b
    for i = 1:n_data
        result = train_D(i,end);
        params = train_D(i,1:end-1)';       % Parameter values as a column vector
        A = A + ridge + params*params';     % Outer product
        b = b + result*params;
    end
    
    % Compute w
    w = inv(A)*b;
end