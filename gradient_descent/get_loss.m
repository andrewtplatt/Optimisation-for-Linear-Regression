function f_tau = get_loss(D, w, lambda, tau)
    n_params = size(D, 2) - 1;
    n_data = size(D, 1);
    results = D(:, end);
    M = D(:, 1:(end-1));
    
    % The non-smoothed function
    f = lambda/2 * norm(w, 1) + 1/n_params * norm(M*w - results, 1);

    % Create a temporary variable to sum for f_tau
    x = zeros(n_data, 1);
    for i = 1:n_data
        params = M(i, :);
        x(i) = 1/tau * (params*w - results(i));
    end
    % Our smoothed function
    f_tau = tau*log(sum(exp(x)));
end