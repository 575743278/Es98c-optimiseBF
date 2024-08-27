% This file is part of the existing blast furnace model developed by Gabriel Eyre, and was used in this study solely for generating data samples.
% The original blast furnace model is available at: https://github.com/gabrieleyre/BlastFurnace

function [RateCoef,Rate,Hr,SpeciesVolume,nTotal,CpnAv,VolUsed,VolUsedFraction,VolAvailable,VolAvailableFraction,HrNet,dTdt,dhWall,dhProducts,dhBurdenDescentNet,dhGasAscentNet,dhHotBlast,HNet,FeedRate,dnOut,dnBurdenDescent,dnGasAscent,VolUsedNew,VolAvailableBurden,n,dn,Temp,t] = blastfurnace(ChemData,ReactionData,tspan,CokeOreRatio,HearthDiameter,HearthHeight,WallThickness,ThermalConductivity,HotBlastRate,HotBlastTemp,OxygenEnrichment,Temp_ext,Temp_initial,SilicaImpurity,nZones,Res)

% Group chemical species by type:

SpeciesBurden = [1 5 6 7 8 10 11 12 13]; % Define Burden (solid/liquid) species
SpeciesGas = [2 3 4 9]; % Define Gas species
SpeciesProducts = [8 13]; % Define BF product species

% Pre-define variables for efficiency:

[RateCoef,Rate,Hr,SpeciesVolume] = deal(zeros([tspan/Res height(ReactionData) nZones]));
[nTotal,CpnAv,VolUsed,VolUsedFraction,VolAvailable,VolAvailableFraction,HrNet] = deal(zeros([tspan/Res nZones]));
[dTdt,dhWall,dhProducts,dhBurdenDescentNet,dhGasAscentNet,dhHotBlast,HNet] = deal(zeros([tspan/Res nZones]));
[FeedRate,dnOut] = deal(zeros([tspan/Res 13]));
[dnBurdenDescent,dnGasAscent] = deal(zeros([tspan/Res 13 nZones]));
[VolUsedNew,VolAvailableBurden] = deal(zeros([(tspan/Res) 1]));
n = zeros([(tspan/Res)+1 13 nZones]);
dn = zeros([tspan/Res 13 nZones]);
Temp = zeros([(tspan/Res)+1 nZones]);

% Define the Stoichiometric Coefficient Matrix:

M = [[-2,-1,2,0,0,0,0,0,0,0,0,0,0,0];... % Reaction 1: 2C + O2 -> 2CO
[0,-1,-2,2,0,0,0,0,0,0,0,0,0,0];...      % Reaction 2: 2CO + O2 -> 2CO2
[0,0,-1,1,-3,2,0,0,0,0,0,0,0,0];...      % Reaction 3: 3Fe2O3 + CO -> 2Fe3O4 + CO2
[0,0,-1,1,0,-1,3,0,0,0,0,0,0,0];...      % Reaction 4: Fe3O4 + CO -> 3FeO + CO2
[0,0,-1,1,0,0,-1,1,0,0,0,0,0,0];...      % Reaction 5: FeO + CO -> Fe + CO2
[0,0,0,1,0,0,0,0,0,-1,1,0,0,0];...       % Reaction 6: CaCO3 -> CaO + CO2
[0,0,0,0,0,0,0,0,0,0,-1,-1,1,0]]';       % Reaction 7: CaO + SiO2 -> CaSiO3

% INITIAL CALCULATIONS

% Calculate Molar Volumes for each species as a new column in ChemData:

ChemData.Mvol = ChemData.Mmass./ChemData.Density;

% Hearth dimension calculations:

HearthVolume = pi*((0.5*HearthDiameter)^2)*HearthHeight; % Total hearth volume [m^3]
AreaWall = 2*pi*(0.5*HearthDiameter)*HearthHeight; % Total internal side wall area [m^2]
AreaWallZone = AreaWall / nZones; % Internal side wall area per zone [m^2]

% Calculate mols of Oxygen in hot blast [mol/s]:

OxygenInAir = 0.21;
HotBlastRateO2 = (HotBlastRate*Res)/ChemData.Mvol(14)*(OxygenInAir+(OxygenEnrichment/100)); 
HotBlastRateN2 = (HotBlastRate*Res)/ChemData.Mvol(14)*(1-(OxygenInAir+OxygenEnrichment/100));

% Calculate Heat of Reactions at standard temperature (25°C):

StdHr = sum(M.*ChemData.stdHForm); % Using Stoichiometric Coefficient Matrix, M

% Calculate coefficient for wall heat loss calculations:

kThermalWall = ThermalConductivity / WallThickness;

% Calculate volumetric factors for burden concentrations:

A(1) = ChemData.Mvol(1); % Factor for Coke
A(2) = ChemData.Mvol(5)/CokeOreRatio; % Factor for Fe2O3
A(3) = ChemData.Mvol(12)*(SilicaImpurity/100)/CokeOreRatio; % Factor for SiO2
A(4) = ChemData.Mvol(10)*(SilicaImpurity/100)/CokeOreRatio; % Factor for CaCO3

% Set initial values for concentrations in each zone:

n(1,1,:) = (A(1)*(HearthVolume/nZones))/((ChemData.Mvol(1))*sum(A)); % Inital Coke concentration in all zones
n(1,5,:) = (A(2)*(HearthVolume/nZones))/(ChemData.Mvol(5)*sum(A)); % Iitial Fe2O3 concentration in all zones
n(1,10,:) = (A(4)*(HearthVolume/nZones))/(ChemData.Mvol(10)*sum(A)); % Initial CaCO3 concentration in all zones
n(1,12,:) = (A(3)*(HearthVolume/nZones))/(ChemData.Mvol(12)*sum(A)); % Initial SiO2 concentration in all zones
n(1,2,nZones) = HotBlastRateO2; % Initial O2 concentration in bottom zone
n(1,9,nZones) = HotBlastRateN2; % Initial N2 concentraiton in bottom zone

% Set initial temperature in all zones (linear distribution):

Temp(1,:) = Temp_ext+(((1:nZones)-1).*(Temp_initial-Temp_ext)/(nZones-1));

% Define timespan vector:

t = 1:(tspan/Res);

% Initiate loop over timespan vector:

for i = t

% BEGINNING OF MASS BALANCE MODEL

% 1) Calculate Kinetic Coefficients as a function of current Temperature (Arrhenius):

    RateCoef(i,:,:) = ReactionData.f_factor(:).*exp(-1*(ReactionData.a_energy(:))./(8.31*(Temp(i,:)+273.15)));
 
% 2) Calculate Reaction Rates (cannot exceed the number of mols of reactants):

    Rate(i,1,:) = min([RateCoef(i,1,:).*n(i,1,:).*n(i,2,:) n(i,1,:) n(i,2,:)]);       % Reaction 1: 2C + O2 -> 2CO
    Rate(i,2,:) = min([RateCoef(i,2,:).*n(i,3,:).*n(i,2,:) n(i,3,:) n(i,2,:)]);       % Reaction 2: 2CO + O2 -> 2CO2
    Rate(i,3,:) = min([RateCoef(i,3,:).*n(i,5,:).*n(i,3,:) n(i,5,:) n(i,3,:)]);       % Reaction 3: 3Fe2O3 + CO -> 2Fe3O4 + CO2
    Rate(i,4,:) = min([RateCoef(i,4,:).*n(i,6,:).*n(i,3,:) n(i,6,:) n(i,3,:)]);       % Reaction 4: Fe3O4 + CO -> 3FeO + CO2
    Rate(i,5,:) = min([RateCoef(i,5,:).*n(i,7,:).*n(i,3,:) n(i,7,:) n(i,3,:)]);       % Reaction 5: FeO + CO -> Fe + CO2
    Rate(i,6,:) = min([RateCoef(i,6,:).*n(i,10,:) n(i,10,:)]);                        % Reaction 6: CaCO3 -> CaO + CO2
    Rate(i,7,:) = min([RateCoef(i,7,:).*n(i,11,:).*n(i,12,:) n(i,11,:) n(i,12,:)]);   % Reaction 7: CaO + SiO2 -> CaSiO3
    
% 3) Calculate change in mols of each species from reactions ('dn'):

    dn(i,1,:) = Res * (-2 * Rate(i,1,:));
    dn(i,2,:) = Res * (-Rate(i,1,:) - Rate(i,2,:));
    dn(i,3,:) = Res * (2 * Rate(i,1,:) - 2 * Rate(i,2,:) - Rate(i,3,:) - Rate(i,4,:) - Rate(i,5,:));
    dn(i,4,:) = Res * (2 * Rate(i,2,:) + Rate(i,3,:) + Rate(i,4,:) + Rate(i,5,:) + Rate(i,6,:));
    dn(i,5,:) = Res * (-3 * Rate(i,3,:));
    dn(i,6,:) = Res * (2 * Rate(i,3,:) - Rate(i,4,:));
    dn(i,7,:) = Res * (3 * Rate(i,4,:) - Rate(i,5,:));
    dn(i,8,:) = Res * (Rate(i,5,:));
    % N2 (species 9) does not react.
    dn(i,10,:) = Res * (-Rate(i,6,:));
    dn(i,11,:) = Res * (Rate(i,6,:) - Rate(i,7,:));
    dn(i,12,:) = Res * (-Rate(i,7,:));
    dn(i,13,:) = Res * (Rate(i,7,:));
    
% 4) Calculate volume used in each zone, post reactions and product flows (burden only):
    
    VolUsed(i,:) = sum(ChemData.Mvol(SpeciesBurden).*squeeze(n(i,SpeciesBurden,:)+dn(i,SpeciesBurden,:)));
    VolAvailable(i,:) = (HearthVolume/nZones) - VolUsed(i,:);
    VolUsedFraction(i,:) = VolUsed(i,:)/(HearthVolume/nZones);
    VolAvailableFraction(i,:) = 1 - VolUsedFraction(i,:);
    SpeciesVolume(i,SpeciesBurden,:) = (ChemData.Mvol(SpeciesBurden).*squeeze(n(i,SpeciesBurden,:) + dn(i,SpeciesBurden,:)))./VolUsed(i,:);
    
% 5) Gas species rise through the BF, from the bottom to top zone:
    
    % Calculate the mols of gas lost from each zone (dnGasAscent):
        
    dnGasAscent(i,SpeciesGas,:) = n(i,SpeciesGas,:) + dn(i,SpeciesGas,:);
        
    % Bottom zone: Receives gas from Hot Blast, loses gas to zone-1:
        
    n(i+1,2,nZones) = HotBlastRateO2; % Oxygen concetration for t = i+1
    n(i+1,3,nZones) = 0; % CO concentration for t = i+1
    n(i+1,4,nZones) = 0; % CO2 concentration for t = i+1
    n(i+1,9,nZones) = HotBlastRateN2; % N2 concentration for t = i+1

    % Middle and Top zones: Receive gas from zone+1, lose gas to zone-1:
    
    for zone = flip(1:nZones-1)
        
        n(i+1,:,zone) = dnGasAscent(i,:,zone+1);
        
    end
    
% 6) Now evaluate movement for Burden species, from top zone to bottom zone:
    
    % Mass only moves down BF when there is volume available in zone below.
    
    % Calculate amount of burden to be lost to zone below (dnBurdenDescent):
    
    for zone = 1:nZones-1
        
        dnBurdenDescent(i,SpeciesBurden,zone) = (SpeciesVolume(i,SpeciesBurden,zone)*VolAvailable(i,zone+1))./ChemData.Mvol(SpeciesBurden)';
    
    end
    
    % Top zone: Loses burden to zone +1:

    n(i+1,SpeciesBurden,1) = n(i,SpeciesBurden,1) + dn(i,SpeciesBurden,1) - dnBurdenDescent(i,SpeciesBurden,1);

    % Middle & Bottom Zones: Receive burden from zone-1, lose burden to zone+1:
    
    for zone = 2:nZones-1
        
        n(i+1,SpeciesBurden,zone) = n(i,SpeciesBurden,zone) + dn(i,SpeciesBurden,zone) + dnBurdenDescent(i,SpeciesBurden,zone-1) - dnBurdenDescent(i,SpeciesBurden,zone);
    
    end

    % Bottom Zone: Receives burden from zone-1.
        
    n(i+1,SpeciesBurden,nZones) = n(i,SpeciesBurden,nZones) + dn(i,SpeciesBurden,nZones) + dnBurdenDescent(i,SpeciesBurden,nZones-1); 
    
% 7) Evaluate Blast Furnace outputs:
    
    % Calculate Top Gas output from top zone = 1:
   
    dnOut(i,SpeciesGas) = dnGasAscent(i,SpeciesGas,1);

    % Caclulate Iron and Slag output from bottom zone = nZones:
        
    dnOut(i,SpeciesProducts) = n(i+1,SpeciesProducts,nZones);
    
    n(i+1,SpeciesProducts,nZones) = 0; % Reset concentrations to zero for Iron and Slag in bottom zone = nZones:
      
% 8) Calculate Burden Feed rates needed to re-fill the top zone:
    
    % Calculate new volume available in top zone:
    
    VolUsedNew(i) = sum(ChemData.Mvol(SpeciesBurden).*n(i+1,SpeciesBurden,1)');
    
    VolAvailableBurden(i) = max([(HearthVolume/nZones)-VolUsedNew(i) 0]);
    
    % Calculate corresponding Feed Rates to fill this available volume [mol/timestep]:

    FeedRate(i,1) = VolAvailableBurden(i)/(sum(A)); % Feed Rate for Coke
    FeedRate(i,5) = FeedRate(i,1)/CokeOreRatio; % Feed rate for Ore
    FeedRate(i,10) = (FeedRate(i,1)/CokeOreRatio)*(SilicaImpurity/100); % Feed rate for Lime
    FeedRate(i,12) = (FeedRate(i,1)/CokeOreRatio)*(SilicaImpurity/100); % Feed rate for Silica
    
% 9) Add Burden from Burden feed to concentrations for next time interval:
    
    n(i+1,:,1) = n(i+1,:,1) + FeedRate(i,:);
 
% END OF MASS BALANCE MODEL

% BEGINNING OF HEAT BALANCE MODEL
    
% Calculate heat losses/gains in period i:

% 1) Reaction heats, adjusted to temperature T (Kirchoff equation) [N.b. +ve = exothermic]:
    
    Hr(i,:,:) = -Res * squeeze(Rate(i,:,:)) .* (StdHr' + sum(M.*ChemData.Cpn)'.* (Temp(i,:)-25));

    HrNet(i,:) = sum(Hr(i,:,:)); % Sum across reactions 1-7 for net reaction heat value

% 2) Heat gain/loss from Hot Blast:
    
    dhHotBlast(i,nZones) = ((HotBlastRate*Res)/ChemData.Mvol(14))*ChemData.Cpn(14)*(HotBlastTemp-Temp(i,nZones));
    
% 3) Heat is lost through the walls of all zones:
    
    dhWall(i,:) = -Res*kThermalWall * AreaWallZone * (Temp(i,:) - Temp_ext);
        
% 4) Heat is lost from Liqiud Iron and Liquid Slag outflow in bottom zone (nZones):
    
    dhProducts(i,nZones) = -1*sum(ChemData.Cpn(SpeciesProducts).*dnOut(i,SpeciesProducts)'*Temp(i,nZones));

% 5) Heat is transferred between zones by gas ascent:
        
    % Top and Middle Zones: Lose gas heat to zone-1, receive from zone+1:
    
    for zone = 1:nZones-1
        
        dhGasAscentNet(i,zone) = sum(ChemData.Cpn(SpeciesGas).*dnGasAscent(i,SpeciesGas,zone+1)'*Temp(i,zone+1)) - sum(ChemData.Cpn(SpeciesGas).*dnGasAscent(i,SpeciesGas,zone)'*Temp(i,zone));
        
    end
    
    % Bottom Zone: loses gas heat to zone above:

    dhGasAscentNet(i,nZones) = -1*sum(ChemData.Cpn(SpeciesGas).*dnGasAscent(i,SpeciesGas,zone)'*Temp(i,zone));
        
% 6) Heat is transferred between zones by burden descent:

    % Top Zone: Loses burden heat to zone+1:
    
    dhBurdenDescentNet(i,1) = -1*sum(ChemData.Cpn(SpeciesBurden).*dnBurdenDescent(i,SpeciesBurden,1)'*Temp(i,1));
    
    % Middle Zones: Receive burden heat from zone-1, lose heat to zone+1:
    
    for zone = 2:nZones-1
        
        dhBurdenDescentNet(i,zone) = sum(ChemData.Cpn(SpeciesBurden).*dnBurdenDescent(i,SpeciesBurden,zone-1)'*Temp(i,zone-1)) - sum(ChemData.Cpn(SpeciesBurden).*dnBurdenDescent(i,SpeciesBurden,zone)'*Temp(i,zone));
        
    end
    
    % Bottom Zone: Receives burden heat from zone-1:

    dhBurdenDescentNet(i,nZones) = sum(ChemData.Cpn(SpeciesBurden).*dnBurdenDescent(i,SpeciesBurden,nZones-1)'*Temp(i,nZones-1));
    
% 7) Net heat change for each zone is the sum of steps 1-6:
   
    HNet(i,:) = HrNet(i,:) + dhWall(i,:) + dhHotBlast(i,:) + dhProducts(i,:) + dhGasAscentNet(i,:) + dhBurdenDescentNet(i,:);
        
% 8) Calculate the total mols and average heat capacity in each zone:
    
    nTotal(i,:) = sum(n(i,:,:)); % Not including dn

    CpnAv(i,:) = sum(ChemData.Cpn(1:13).*squeeze(n(i,1:13,:)))./nTotal(i,:); % Not including dn
    
% 9) Calculate change in temperature in each zone arising from net heat in period i:
        
    dTdt(i,:) = HNet(i,:)./(CpnAv(i,:).*(nTotal(i,:)));
            
% 10) Calculate new zone temperature for next time period:
    
    Temp(i+1,:) = Temp(i,:) + dTdt(i,:);
        
end % Time-loop end
end % Function end
