% Returns the best estimate parameters w to fit the training data to the
% costs.
function w = ridge_regression(train_D, lambda)
    [n_data, n_params] = size(train_D);
    
    % Get the ridge offset to penalise large terms
    ridge = n_data * lambda/(2*n_params);
    
    % Solve as Aw = b
    M = train_D(:,1:end-1);
    results = train_D(:,end);
    A = M'*M + ridge;
    b = M'* results;
%     % Get A and b
%     for i = 1:n_data
%         params = train_D(i,1:end-1)';       % Parameter values as a column vector
%         A = A + ridge + params*params';     % Outer product
%     end
%     b = train_D(:,1:end-1)'*train_D(:,end);
    
    % Compute w
    w = inv(A)*b;
end