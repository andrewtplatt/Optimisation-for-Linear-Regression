% Splits a training data set randomly into two.
% 0 < frac < 1 of the rows are assigned to train_D, and the rest to test_D.

function [train_D, test_D] = random_split(D, frac)
    n_rows = size(D,1);
    cutoff = floor(frac*n_rows);
    rand_rows = randperm(n_rows);
    train_D = D(rand_rows(1:cutoff),:);
    test_D = D(rand_rows(cutoff + 1:end),:);
end