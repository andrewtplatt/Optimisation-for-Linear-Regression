function w = lp_l1_regression(train_D, lambda)
    n_params = size(train_D, 2) - 1;
    n_data = size(train_D, 1);
    M = train_D(:, 1:n_params);
    y = train_D(:, end);
    
    % Create A and b such that the constraints are as Ax <= b
    A_rows = 2*n_params + 2*n_data;
    A_cols = 2*n_params + n_data;
    A = sparse(A_rows, A_cols);
    b = zeros(A_rows, 1);
    
    %% Define our objective function
    % In the order w > v > xi
    f_w = zeros(n_params, 1);
    f_v = lambda/2*ones(n_params, 1);
    f_xi = 1/n_data*ones(n_data, 1);
    f = [f_w; f_v; f_xi];
    
    %% Populate A
    % Put M in the bottom left corner
    A((2*n_params+1):(2*n_params+n_data), 1:n_params) = sparse(M);
    A((2*n_params+n_data+1):end, 1:n_params) = sparse(-M);
    % Put -I in the bottom right corner
    A((2*n_params+1):(2*n_params+n_data), 2*n_params+1:end) = -speye(n_data);
    A((2*n_params+n_data+1):end, 2*n_params+1:end) = -speye(n_data);
    % Fill the top left corner to satisfy vj >= wj, -wj
    row = 1; col = 1;
    while row <= 2*n_params
       A(row, col) = -1;
       A(row, col+n_params) = -1;
       A(row+1, col) = 1;
       A(row+1, col+n_params) = -1;
       row = row + 2;
       col = col + 1;
    end
    A = sparse(A);
    
    %% Populate b
    b((2*n_params+1):(2*n_params+n_data)) = y;
    b((2*n_params+n_data+1):end) = -y;
    
    %% Calculate the result
    % Set the optimoptions
    options = optimoptions(@linprog, 'Algorithm', 'interior-point');
    res = linprog(f, A, b, [], [], [], [], options);
    w = res(1:n_params);
    
end