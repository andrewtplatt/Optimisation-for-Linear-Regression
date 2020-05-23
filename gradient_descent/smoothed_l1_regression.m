function w = smoothed_l1_regression(train_D, lambda)
    tau = 5e-5;                   % Smoothing parameter
    eta = 2;                      % Step size for gradient descent
    % Criteria for the loop to end
    max_n_iterations = 5e5;
    cutoff_grad = 5e-5;
    
    % Parameters obtained from the training matrix
    n_params = size(train_D, 2) - 1;
    
    % Initialisation of the w vector
    w = zeros(n_params, 1);
    best_w = w;
    
    % Get the first loss function value and store it ready for comparison
    % with the next one
    loss = [get_loss(train_D, w, lambda, tau), 0];
    mae = zeros(max_n_iterations, 1);
    mod_grad = mae;
    mod_grad(1) = 1;
    % Iterate and travel downhill until the loss function is small enough
    iter = 1;
    while iter < max_n_iterations && mod_grad(iter) > cutoff_grad
        mae(iter) = compute_mean_abs_error(train_D, w);
        if iter ~= 1
            loss(1) = loss(2);
        end
        grad = get_grad(train_D, w, lambda, tau);
        mod_grad(iter+1) = norm(grad);
        w = w - grad*eta;
        loss(2) = get_loss(train_D, w, lambda, tau);
        % If our new w is worse, reduce the step size and continue
        if loss(2) > loss(1)
             if eta > 1e-4
                eta = eta/1.5;
             end
        else
            best_w = w;
        end
        iter = iter + 1;
    end
    
    w = best_w;
    
    % plot the error
%     mae = mae(1:iter-1);
%     mod_grad = mod_grad(1:iter-1);
%     subplot(2,1,1)
%     plot(mae)
%     xlabel('Iteration')
%     ylabel('Mean Absolute Error')
%     subplot(2,1,2)
%     plot(mod_grad)
%     ylabel('Gradient Magnitude')
%     drawnow
end