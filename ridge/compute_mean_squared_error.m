function mse = compute_mean_squared_error(test_D, w)
    errors = test_D(:,1:end-1)*w - test_D(:,end);
    mse = mean(errors.^2);
end