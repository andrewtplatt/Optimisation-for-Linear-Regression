function grad = get_grad(D, w, lambda, tau)
    % Parameters of D
    n_data = size(D, 1);
    n_params = size(D, 2) - 1;
    M = D(:, 1:n_params);
    results = D(:, end);
    
    % The error in our current vector w
    errs = M*w - results;
    
    % The loss gradient associated with this error
    err_grad =2*n_params/n_data * M'*(tanh(errs/tau) / 2);

    % The loss gradient associated with the 1 norm of our vector
    norm_grad = tanh(w/tau) / 2;
    
    % The gradient is the sum of the above
    grad = lambda * norm_grad +  err_grad;
end