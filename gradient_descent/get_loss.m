function f_tau = get_loss(D, w, lambda, tau)
    % Parameters of D
    n_params = length(w);
    n_data = size(D, 1);
    results = D(:, end);
    M = D(:, 1:(end-1));
    
    % The error in our current vector w
    errs = M*w - results;
    
    % The non-smoothed function
    f = lambda/2 * norm(w, 1) + 1/n_params * norm(errs, 1);
    
    % The loss associated with this error
    x = sum(log(1 + exp(-2*abs(errs)/tau)));

    % The loss associated with the size of w 
    y = sum(log(1 + exp(-2*abs(w))));
    
    % Our smoothed function
    f_tau = f + tau*lambda/2 * y + tau/n_data * x;
end