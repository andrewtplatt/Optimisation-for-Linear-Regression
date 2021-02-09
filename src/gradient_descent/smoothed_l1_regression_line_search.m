function w = smoothed_l1_regression_line_search(train_D, lambda)
    tau = 1e-2;                   % Smoothing parameter
    eta_0 = 1;                    % Step size for gradient descent
    eta = eta_0;
    
    % Criteria for the loop to end
    max_n_iterations = 2e5;
    eta_min = 1e-40;              % Step size at which to cut off and end
    cutoff_grad = 2e-3;           % End
    
    % Parameters obtained from the training matrix
    n_params = size(train_D, 2) - 1;
    
    % Initialisation of the w vector
    w = randn(n_params, 1);
    best_w = w;
    
    % Get the first loss function value and store it ready for comparison
    % with the next one and the best one
    current_loss = get_loss(train_D, w, lambda, tau);
    prev_loss = 1e100;
    best_loss = current_loss;
    mod_grad = zeros(max_n_iterations, 1);
    mod_grad(1) = norm(get_grad(train_D, w, lambda, tau));
    
    % Iterate and travel downhill until the loss function is small enough
    iter = 1;
    while iter < max_n_iterations && mod_grad(iter) > cutoff_grad && prev_loss - current_loss > 0
        tau = 1e-2/iter;
        prev_w = w;
        prev_loss = current_loss;
        grad = get_grad(train_D, w, lambda, tau);
        mod_grad(iter+1) = norm(grad);
        w = prev_w - grad*eta;
        current_loss = get_loss(train_D, w, lambda, tau);
        % Perform Line Search
        eta = eta_0;
        while current_loss > prev_loss && eta > eta_min
           eta = eta/2;
           w = prev_w - grad*eta;
           current_loss = get_loss(train_D, w, lambda, tau);
        end
        % If our new w is the best then set it as such.
        if current_loss < best_loss
            best_loss = current_loss;
            best_w = w;
        end
        iter = iter + 1;
    end
    
    w = best_w;
    fprintf('\x03bb: %.1g  iter: %d\tgradient magnitude: %.5g\tLoss: %.5g\teta: %.2g\n', lambda, iter, mod_grad(iter), current_loss, eta)
end