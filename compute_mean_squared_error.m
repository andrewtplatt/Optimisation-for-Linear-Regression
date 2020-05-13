function mse = compute_mean_squared_error(test_D, w)
    n_data = size(test_D, 1);
    errors = test_D(:,1:end-1)*w - test_D(:,end);
    mse = sum(errors.^2) / n_data;
end