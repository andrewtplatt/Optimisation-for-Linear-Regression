% Puts the dataset in the workspace for easy use. Probably temporary??
function D = get_dataset()
    fileID = fopen('/home/andrew/Documents/B Finals/B1 Engineering Computation/B1 Project B/Dataset.txt', 'r');
    formatSpec = '%f';
    % As many rows as we like, but we need to specify the number of
    % columns.
    sizeD = [14, inf];
    D = fscanf(fileID, formatSpec, sizeD);
    D = D';
    fclose('all');
end