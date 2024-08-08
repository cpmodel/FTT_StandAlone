# Instructions for updating FTT:Power
Ideally, FTT:Power is updated every two years. The last data update was done early 2022. Final figures for IEA shares come out in August. For costs, BNEF is very rapid, a few months behind. Unknown if GNESTE keeps updating. **You may need to have different start years for cost and generation in the code**

## Frequent updates
### Historical generation
1. Update the historical generation. We use the IEA World Energy Balances to update generation data. This data is freely available for universities. People with a UK institutional log-in can find it at the [UK data services under the International Energy Agency](https://stats2.digitalresources.jisc.ac.uk/index.aspx?r=721229&DataSetCode=IEA_CO2_AB). People at CE also have access **Describe how**
    1. The datafiles to update are Inputs/_MasterFiles/FTT-P/FTT-P-24x70_2021_S[0-1-2].xlsx. The generation is in the MEWG sheet
    2. This step is done manually. If you create a python script for this, please add it to the pre-processing repository.  
2. The IEA World Energy Balances does not distinguish between onshore and offshore. For consistency, we use the overall wind generation data from IEA, but split it out by country using the [historical generation from IRENA](https://www.irena.org/publications/2022/Apr/Renewable-Capacity-Statistics-2022).
    1. The datafile is the same as above
    2. This step is done in excel. If you create a python script for this, please add it to the pre-processing repository.  
3. The generation data is often not quite up-to-date. You can get more up-to-date capacity data from IRENA. This can be added to the exogenous capacity (policy) variable, the MWKA variable. Do this for fast-changing technologies if relevant (offshore, solar PV). 
    4. The python file to do this is can be found in the pre-processing repository (change_IRENA_hist_capacity_MWKA.py). 
    5. The (2022) data is found at https://irena.org/Statistics/Download-query-tools 
5. For technologies like CSP and offshore, introduce a 'seed' in countries without. The FTT code base does not allow for a new technology to appear if a region does not have any capacity in that technology. To combat this limitation, we add 1% of total wind energy in a country as offshore and 1% of solar PV as CSP in regions without any (and regions with less than 1%).
    1. Use the same script as above in the pre-processing file. Better seeding can be introduced in MWKA, to make sure we're accurately representing historical production.
6. Edit the end-years in FTT-Standalone/Utilities/Titles/VariableListing.csv. For instance, change J3 from 2021 to 2022 after you've updated the historical generation to include 2022 data. 

### Calibration (the gamma values)
The FTT model is calibrated to ensure a historical trends do not suddenly change in the absense of new policies. We ensure the first derivative of the shares (MEWS) variable is approximately zero. We estimate a gamma value per country and per technology. 
1. To calibrate the gamma values, run the frontend of the standalone version (FTT_Stand_Alone_Launcher.cmd). Navigate to GAMMA. Initialise the power sector model, and do the following by country
2. Pick a start date which gives you 5 years of historical data, and an end date with 5 year of future data
3. Per technology, choose a gamma value that ensures historical trends continue. The gamma value is considered a "price premium". Positive gamma values will make the technology less attractive, negative values will make it more attractive. If gamma values are often larger than 30, there may be structural errors in the model, so feel free to contact an experienced modeller. 
4. Save the gamma values in Inputs/_MasterFiles/FTT-P/FTT-P-24x70_2021_S[0-1-2].xlsx.

### Differing start dates for costs and capacity
1. If your cost data is not from the same year as your final generation data, a code change needs to be made. In ftt_p_main.py, you will need to add an if statement (if year == (cost data year + 1)).., to run the learning-by-doing up to the proper start of the simulation. 
    1. TODO Option 1: switch to BNEF for cost data
    2. TODO Option 2: make a new variable in FTT-Standalone/Utilities/Titles/VariableListing.csv which contains the date of the cost data, and adjust the code to reflect, so that code doesn't need updating.

### Costs
1. Update the costs of CAPEX, OPEX and the standard deviation of both using a the average of BNEF data, GNESTE and Enerdata (2 out of 3 is sufficient). Exeter has access to BNEF data, the WB to Enerdata and [GNESTE data](https://github.com/iain-staffell/GNESTE) is open-access. Do not add raw data from BNEF or Enerdata to the repository; they are not open access! 
    1. Ensure the sources use the same currency (note that $2020USD is different from $2023USD). 
    2. Assume standard deviation is 30% for CAPEX and OPEX, and verify this assumption using BNEF data, which has a range per country. If the range is smaller or larger, adjust this update manual.
    3. No script is yet available for the update. If you write a Python script, please add it to the [support repo](https://github.com/cpmodel/FTT_Standalone-support)
2. Update the fuel costs
    1.  Fast update: do the same as above for fuel costs. Note that costs are higher for technologies with CCS.
    2.  Higher-quality update: The [UK data services under the International Energy Agency](https://stats2.digitalresources.jisc.ac.uk/index.aspx?r=721229&DataSetCode=IEA_CO2_AB) is down at the moment, but Exeter has access until May 2025, so ask Ian. The dataset is the World Energy Prices Yearly. Sector is Industry. Take the average of the last 5 years to account for fluctuations. Take the sample standard deviation over the last 5 years for the standard deviation of fuel costs. Convert the units for coal and for oil into MWh. 
4. Update the learning rate. We use learning rates from literature. It may be worth revisiting every 5 to 10 years, depending on the novelty of the technology and speed of deployment. The last update was done in early 2022 for solar and wind technologies, as well as storage technologies. 
5. Edit the end-years in FTT-Standalone/Utilities/Titles/VariableListing.csv. For instance, change J5 from 2020 to 2023 after you've updated the cost data. 
6. Adjust the currency conversions in the code to the new currency (for instance, from 2013USD to 2023USD). Ideally, this is done automatically from histend.

## Less frequent updates

### Technical parameters
5. Update the technical potential. The last update for the technical potential for onshore, offshore and solar was done early 2022. 
6. Update Cost-Supply Curves to reflect maximum capacity factor by country. These go up steadily over time for solar and wind, as efficiency and capacity factors improve.
7. You can also check efficiency and GHG emissions, which change with slow changes of technologies.  Note that emissions are computed bottom-up, but there is a top-down correction to ensure our overall emissions are correct in E3ME.
8. Update efficiencies of storage technologies if needed.

### Classification
1. Update which technologies should be included. Are there new technologies we should include? In particular storage technologies / hydrogen? And have some technologies been relegated to the dustbin?
