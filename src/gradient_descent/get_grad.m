function grad = get_grad(D, w, lambda, tau)
    % Parameters of D
    n_data = size(D, 1);
    A = D(:, 1: end - 1);
    results = D(:, end);
    
    % The error in our current vector w
    errs = A*w - results;
    
    % The loss gradient associated with this error
    err_grad = 1/n_data * A'*tanh(errs/tau);

    % The loss gradient associated with the 1 norm of our vector
    norm_grad = lambda * tanh(w/tau) / 2;
    
    % The gradient is the sum of the above
    grad = norm_grad + err_grad;
end