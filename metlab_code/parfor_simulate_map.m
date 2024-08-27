function parfor_simulate_map1()
    numFiles = 10; 
    results = cell(1, numFiles);

    
    if ~isempty(gcp('nocreate'))
        delete(gcp('nocreate'));
    end

    
    parpool('local');

    parfor i = 1:numFiles
        input_filename = sprintf('/Users/han/地图/input_file3_%d.csv', i);
        output_filename = sprintf('/Users/han/地图/blast_results_from_file3_%d.csv', i);
        results{i} = parfor_blast_furnace_simulation1(input_filename, output_filename, '/Users/han/Documents/blast_furnace/BlastFurnace-master/chemdata.csv', '/Users/han/Documents/blast_furnace/BlastFurnace-master/reactdata.csv');
    end

    
    delete(gcp('nocreate'));

    
    disp(results);
end