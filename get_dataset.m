% Puts the dataset in the workspace for easy use. Probably temporary??
function D = get_dataset()
    fileID = fopen('/home/andrew/Documents/B Finals/B1 Engineering Computation/B1 Project B/CaliforniaHousingDataSet.txt', 'r');
    formatSpec = '%f';
    % As many rows as we like, but we need to specify the number of
    % columns.
    sizeD = [9, inf];
    D = fscanf(fileID, formatSpec, sizeD);
    D = D';
    fclose('all');
    for column = 1:size(D, 2)
       D(:, column) = (D(:, column) - mean(D(:, column))) / std(D(:, column)); 
    end
end