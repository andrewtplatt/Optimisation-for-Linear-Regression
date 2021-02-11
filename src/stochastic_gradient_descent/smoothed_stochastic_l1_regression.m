function w = smoothed_stochastic_l1_regression(train_D, lambda, B)
  tau = 1e-2;                     % Smoothing parameter
  eta0 = 5;                       % Step size for gradient descent
  eta = eta0;
  beta = 0;                       % Momentum parameter
  
  % Criteria for the loop to end
  eta_min = 2e-3;
  
  % Parameters obtained from the training matrix
  n_params = size(train_D, 2) - 1;
  n_data = size(train_D, 1);
  n_batches = ceil(n_data/B);
  batches = split_n(train_D, n_batches);
  
  % Initialisation of the w vector
  w = randn(n_params, 1);
  best_w = w;
  
  % Get the first loss function value and store it ready for comparison
  % with the next one and the best one
  current_loss = get_loss(train_D, w, lambda, tau);
  best_loss = current_loss;
  
  % Iterate and travel downhill until the loss function is small enough
  iter = 1;
  batch = 1;
  while eta > eta_min
    tau = 1e-2/iter;
    
    % Pick a random subset of the data to use for this iteration and
    % calculate the gradient with it
    grad_calc_D = batches(:,:,batch);
    grad = get_grad(grad_calc_D, w, lambda, tau);
    if (iter == 1)
      prev_grad_prime = grad;
    end
    
    % Update step size
    eta = eta0/iter;
    % Step downhill
    grad_prime = (1 - beta)*grad + beta*prev_grad_prime;
    w = w - grad_prime*eta;
    current_loss = get_loss(train_D, w, lambda, tau);
    
    % If our new w is the best then set it as such.
    if current_loss < best_loss
      best_loss = current_loss;
      best_w = w;
    end
    
    iter = iter + 1;
    batch = batch + 1;
    if batch > length(batches)
      batch = batch - length(batches);
    end
    prev_grad_prime = grad_prime;
  end
  
  w = best_w;
  fprintf('lambda: %.1e\tB.S.: %.2g  iter: %d\tLoss: %.5g\teta: %.2g\n', lambda, B, iter, current_loss, eta)
end