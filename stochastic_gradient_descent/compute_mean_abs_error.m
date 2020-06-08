function mae = compute_mean_abs_error(val_D, w)
    errors = val_D(:,1:end-1)*w - val_D(:,end);
    mae = mean(abs(errors));
end