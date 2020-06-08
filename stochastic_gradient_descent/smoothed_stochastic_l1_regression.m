function w = smoothed_stochastic_l1_regression(train_D, lambda, B)
    tau = 1e-2;                   % Smoothing parameter
    eta = 1;                      % Step size for gradient descent
    
    % Criteria for the loop to end
    max_n_iterations = 1e5/B;
    
    % Parameters obtained from the training matrix
    n_params = size(train_D, 2) - 1;
    n_data = size(train_D, 1);
    
    % Initialisation of the w vector
    w = randn(n_params, 1);
    best_w = w;
    
    % Get the first loss function value and store it ready for comparison
    % with the next one and the best one
    current_loss = get_loss(train_D, w, lambda, tau);
    best_loss = current_loss;
    mae = zeros(max_n_iterations, 1);
    losses = zeros(max_n_iterations, 1);
    
    % Iterate and travel downhill until the loss function is small enough
    iter = 1;
    while iter < max_n_iterations
        tau = 1e-2/iter;
        mae(iter) = compute_mean_abs_error(train_D, w);
        losses(iter) = current_loss;
        
        % Pick a random subset of the data to use for this iteration and
        % calculate the gradient with it
        [grad_calc_D, ~] = random_split(train_D, B/n_data);
        grad = get_grad(grad_calc_D, w, lambda, tau);
        
        % Update step size 
        eta = 1/iter;
        % Step downhill
        w = w - grad*eta;
        current_loss = get_loss(train_D, w, lambda, tau);
        if mod(iter, 1000) == 1
            fprintf('lambda: %.1e\tB.S.: %.2g  iter: %d\tLoss: %.5g\teta: %.2g\n', lambda, B, iter, current_loss, eta)
        end
        % If our new w is the best then set it as such.
        if current_loss < best_loss
            best_loss = current_loss;
            best_w = w;
        end
        iter = iter + 1;
    end
    
    w = best_w;
    fprintf('lambda: %.1e\tB.S.: %.2g  iter: %d\tLoss: %.5g\teta: %.2g\n', lambda, B, iter, current_loss, eta)
    
    % plot the error
    mae = mae(1:iter-1);
    losses = losses(1:iter-1);
    subplot(2,1,1)
    plot(mae)
    xlabel('Iteration')
    ylabel('Mean Absolute Error')
    subplot(2,1,2)
    plot(losses)
    ylabel('Loss')
    drawnow
end