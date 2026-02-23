## Emulation
This branch is dedicated to emulation and uncertainty analysis as proposed in the working paper:
'Policy Robustness & Uncertainty in Model-based Decision Support for the Energy Transition'
Burton, I.J, Njisse, F.J.M.M, Salter, J.M - https://arxiv.org/abs/2510.11177


Workflow:

1. Parameterisation
    -   ALL SECTIONS OF PARAMETERISATION AND SIMULATION CAN BE PERFORMED CONTINUOUSLY SETTING ALL "run" sections in config to "true"
    a. Creating policy scenarios
        - Masterfiles in the Inputs/_MasterFiles/FTT-P folder act as the parameter bounds for policy inputs
        - S0 is the baseline and S3 (changeable) represents the upper bounds for policy vars
        MEWT (subsidies), MEFI (feed-in tariffs), MEWR (regulations)
        - CARBON PRICE upper bound is edited at Emulation\data\variable_data\carbon_price\REPPX_amb.csv 
        - choose upper bounds for these vars
        - edit Emulation/config/config.json "run_compare_scenarios": true -  with all other "run" modules set to "false"
        - edit file paths
        - run Emulation/code/simulation_code/main.py
    b. Config
        - edit the number of scenarios to create in config "N_scens" 
        - Technoeconomic parameters are edited in config "ranges"
        - Regions and groupings can be edited using the "regions" & "region_groups" sections
        - Rollbacks can be added for regions in "reg_rollback"
        - edit config - "run_generate_scenarios": true -  with all other "run" modules set to "false"
        - edit file paths
        - run Emulation/code/simulation_code/main.py

2. Simulation
    a. edit scenarios to run in "scens_run" (editable subgroup of scenarios for testing) 
    b. edit config "run_ambition_vary": true, "run_run_simulations": true, "run_output_manipulation": true,-  with all other "run" modules set to "false"
    c. run Emulation/code/simulation_code/main.py (eta 45s/scenario)
3. Emulation
    a.  edit filepaths and vars in FTT_StandAlone/Emulation/code/emulation_code/autobuild.R
    b. run autobuild.R
4. Prediciton/Analysis
    a. Sensitivity analysis
        - edit filepaths and vars in FTT_StandAlone/Emulation/code/emulation_code/sensitivity_analysis.R
    b. Prediction
        - edit filepaths and vars in FTT_StandAlone/Emulation/code/emulation_code/prediction.R
        - run prediction.R
    c. Plotting
        - edit filepaths and vars in FTT_StandAlone/Emulation/code/emulation_code/plotting.R
        - run plotting.R

