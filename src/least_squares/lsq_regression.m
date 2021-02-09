% Returns the best estimate parameters w to fit the training data to the
% costs.
function w = lsq_regression(train_D)
    % Solve as Aw = b
    M = train_D(:,1:end-1);
    results = train_D(:,end);
    A = M'*M;
    b = M'*results;
    % Compute w
    w = A\b;
end