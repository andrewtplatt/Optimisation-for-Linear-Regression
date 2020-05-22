function grad = get_grad(D, w, tau)
    n_data = size(D, 1);
    n_params = size(D, 2) - 1;
    M = D(:, 1:n_params);
    results = D(:, end);
    
    grad = zeros(n_params, 1);
    numer = grad; denom = 0;
    for i = 1:n_data
       numer = numer + M(i,:)'*n_params*exp((M(i,:)*w - results(i))/tau);
       denom = denom + exp((M(i,:)*w - results(i))/tau);
    end
    grad = numer / denom
end