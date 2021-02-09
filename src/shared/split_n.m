function datasets = split_n(D_in, n)
%SPLIT_N Splits a dataset into n equal parts
%   Returns a 3-dimensional array for an 2-dimensional D_in

  out_dim = [ceil(size(D_in, 1)/n), size(D_in, 2), n];        % Dimensions of our output
  datasets = zeros(out_dim);                                  % Output 3D array
  remainder = D_in;                                           % Stores the remaining datapoints to assign
  
  for i = n:-1:1
      frac = (i - 1)/i;
      [remainder, new_set] = random_split(remainder, frac);
      new_set_length = size(new_set, 1);
      datasets(1:new_set_length, :, n - i + 1) = new_set;
  end
end

