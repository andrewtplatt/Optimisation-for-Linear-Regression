% Puts the dataset in the workspace
function D = get_dataset()
    D = load('..\..\CaliforniaHousingDataSet.txt');
    for column = 1:size(D, 2)
       D(:, column) = (D(:, column) - mean(D(:, column))) / std(D(:, column)); 
    end
end