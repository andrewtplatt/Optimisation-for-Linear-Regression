function w = smoothed_l1_regression(train_D, lambda)
    tau = 0.1;                % Smoothing parameter
    eta = 0.2;                % Step size for gradient descent
    % Criteria for the loop to end
    max_n_iterations = 1e4;
    cutoff_loss = 1e-6;
    
    % Parameters obtained from the training matrix
    n_params = size(train_D, 2) - 1;
    n_data = size(train_D, 1);
    results = train_D(:, end);
    M = train_D(:, 1:(end-1));      % Matrix of parameters
    
    % Initialisation of the w vector
    w = zeros(n_params, 1);
    % Get the first loss function value and store it ready for comparison
    % with the next one
    loss = [get_loss(train_D, w, lambda, tau), 0];
    
    % Iterate and travel downhill until the loss function is small enough
    iter = 1;
    while iter < max_n_iterations && (loss(1) - loss(2)) > cutoff_loss
        if iter ~= 1
            loss(1) = loss(2);
        end
        grad = get_grad(train_D, w, tau);
        w = w - grad*eta;
        loss(2) = get_loss(train_D, w, lambda, tau);
        iter = iter + 1;
    end
end