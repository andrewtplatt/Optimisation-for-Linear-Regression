% Returns the best estimate parameters w to fit the training data to the
% costs.
function w = ridge_regression(train_D, lambda)
    [n_data, ~] = size(train_D);
    
    % Get the ridge offset to penalise large terms
    ridge = n_data * lambda/2;
    
    % Solve as Aw = b
    M = train_D(:,1:end-1);
    results = train_D(:,end);
    A = M'*M + ridge;
    b = M'* results;
    
    % Compute w
    w = inv(A)*b;
end