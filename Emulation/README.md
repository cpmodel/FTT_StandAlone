## Emulation
This branch is dedicated to emulation and uncertainty analysis as proposed in the working paper:
'Policy Robustness & Uncertainty in Model-based Decision Support for the Energy Transition'
Burton, I.J, Njisse, F.J.M.M, Salter, J.M - https://arxiv.org/abs/2510.11177



Workflow:
1. Parameterisation
    a. Creating policy scenarios
        - Masterfiles in the Inputs/_MasterFiles/FTT-P folder act as the parameter bounds for policy inputs
        - S0 is the baseline and S3 (changeable) represents the upper bounds for policy vars
        MEWT (subsidies), MEFI (feed-in tariffs), MEWR (regulations)
        - CARBON PRICE upper bound is edited at Emulation\data\variable_data\carbon_price\REPPX_amb.csv 
        -
        - Technoeconomic parameters are edited
         
    b. Config
2. Simulation
3. Emulation
4. Analysis

