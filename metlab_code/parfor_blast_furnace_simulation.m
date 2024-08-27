% This file is part of the existing blast furnace model developed by Gabriel Eyre, and was used in this study solely for generating data samples.
% The original blast furnace model is available at: https://github.com/gabrieleyre/BlastFurnace

function result = parfor_blast_furnace_simulation1(input_filename, output_filename, chemdata_filename, reactdata_filename)
  
    opts = detectImportOptions(input_filename, 'NumHeaderLines', 0);
    opts = setvartype(opts, 'double');
    data = readtable(input_filename, opts);
    inputs = table2array(data);

  
    ChemData = readtable(chemdata_filename);
    ReactionData = readtable(reactdata_filename);

   
    tspan = 50000; 
    HearthDiameter = 20; 
    HearthHeight = 10; 
    WallThickness = 2; 
    ThermalConductivity = 30; 
    HotBlastTemp = 1200; 
    OxygenEnrichment = 3; 
    Temp_ext = 40; 
    Temp_initial = 40; 
    SilicaImpurity = 2; 
    nZones = 3; 
    Res = 1; 

   
    fileID = fopen(output_filename, 'w');
  
    fprintf(fileID, 'CokeOreRatio,HotBlastRate,f1,f2,f3,f4,f5,f6,f7,lastFe,lastCo2,CO2_Fe_Ratio\n');


    for j = 1:size(inputs, 1)
    
        input_combination = inputs(j, :);
        disp(j)
        CokeOreRatio = input_combination(1);
        HotBlastRate = input_combination(2);
        f_factors = input_combination(3:end);

    
        ReactionData.f_factor = f_factors(:); 

    
        [RateCoef, Rate, Hr, SpeciesVolume] = deal(zeros([tspan/Res height(ReactionData) nZones]));
        [nTotal, CpnAv, VolUsed, VolUsedFraction, VolAvailable, VolAvailableFraction, HrNet] = deal(zeros([tspan/Res nZones]));
        [dTdt, dhWall, dhProducts, dhBurdenDescentNet, dhGasAscentNet, dhHotBlast, HNet] = deal(zeros([tspan/Res nZones]));
        [FeedRate, dnOut] = deal(zeros([tspan/Res 13]));
        [dnBurdenDescent, dnGasAscent] = deal(zeros([tspan/Res 13 nZones]));
        [VolUsedNew, VolAvailableBurden] = deal(zeros([(tspan/Res) 1]));
        n = zeros([(tspan/Res)+1 13 nZones]);
        dn = zeros([tspan/Res 13 nZones]);
        Temp = zeros([(tspan/Res)+1 nZones]);

 
        [RateCoef, Rate, Hr, SpeciesVolume, nTotal, CpnAv, VolUsed, VolUsedFraction, VolAvailable, VolAvailableFraction, HrNet, dTdt, dhWall, dhProducts, dhBurdenDescentNet, dhGasAscentNet, dhHotBlast, HNet, FeedRate, dnOut, dnBurdenDescent, dnGasAscent, VolUsedNew, VolAvailableBurden, n, dn, Temp, t] = blastfurnace(ChemData, ReactionData, tspan, CokeOreRatio, HearthDiameter, HearthHeight, WallThickness, ThermalConductivity, HotBlastRate, HotBlastTemp, OxygenEnrichment, Temp_ext, Temp_initial, SilicaImpurity, nZones, Res);

  
        FeoutCumulative = cumsum(dnOut(:, 8));
        Co2outCumulative = cumsum(dnOut(:, 4));
        lastFe = FeoutCumulative(end);
        lastCo2 = Co2outCumulative(end);
        CO2_Fe_Ratio = lastCo2 / lastFe;

       
        fprintf(fileID, '%.10f,%.10f,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10f,%.10f,%.10f\n', CokeOreRatio, HotBlastRate, f_factors(1), f_factors(2), f_factors(3), f_factors(4), f_factors(5), f_factors(6), f_factors(7), lastFe, lastCo2, CO2_Fe_Ratio);
    end

   
    fclose(fileID);

   
    result = sprintf('Results saved to %s', output_filename);
end
