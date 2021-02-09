function f_tau = get_loss(D, w, lambda, tau)
    % Parameters of D
    n_data = size(D, 1);
    results = D(:, end);
    A = D(:, 1:(end-1));
    
    % The error in our current vector w
    errs = A*w - results;
    
    % The non-smoothed function
    f_w = lambda/2 * norm(w,1);
    f_err = 1/n_data * norm(errs, 1);
    f = f_w + f_err;
    
    % The loss associated with this error
    x = tau/n_data * sum(log(exp(errs - abs(errs)) + exp(-errs - abs(errs))));

    % The loss associated with the size of w 
    y = lambda*tau/2 * sum(log(exp(w - abs(w)) + exp(-w - abs(w))));
    
    % Our smoothed function
    f_tau = f + x + y;
end