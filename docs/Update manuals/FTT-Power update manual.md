# Instructions for updating FTT:Power
Ideally, FTT:Power is updated every one or two years. The last data update was done in September/October 2024. Final figures for IEA shares come out in August. For costs, BNEF is very rapid, a few months behind. Unknown if GNESTE keeps updating, but the IRENA data within GNESTE does, in August or September. **You may need to have different start years for cost and generation in the code**

## Frequent updates
### Historical generation
1. Update the historical generation. We use the IEA World Energy Balances to update generation data. Exeter has paid access to the full data until May 2025. This data was freely available for UK universities, and hopefully in the future again via the [UK data services under the International Energy Agency](https://stats2.digitalresources.jisc.ac.uk/index.aspx?r=721229&DataSetCode=IEA_CO2_AB). People at CE also have access via license to used energy balances with E3ME the data can be accessed via IEA WDS service. You can find a script to do this in the [pre-processing repository](https://github.com/cpmodel/FTT_Standalone-support/tree/main/FTT-Power%20updates), called "Process Generation, capacity and load data.py".  
    1. The datafiles to update are Inputs/_MasterFiles/FTT-P/FTT-P-24x70_2021_S[0-1-2].xlsx. The generation is in the MEWG sheet
3. The IEA World Energy Balances does not distinguish between onshore and offshore. For consistency, we use the overall wind generation data from IEA, but split it out by country using the [historical generation from IRENA](https://pxweb.irena.org/pxweb/en/IRENASTAT). Due to data extraction limits, capacity and generation data should be extracted as separate files.
4. The generation data is often not quite up-to-date (typically a year behind the capacity data available). You can get more up-to-date capacity data from IRENA. This can be added to the exogenous capacity (policy) variable, the MWKA variable. The IRENA and EMBER datasets provide capacity estimate for all main technologies which allow for an accurate update of the MWLO variable to the last year of history (2022)   
5. For technologies like CSP and offshore, introduce a 'seed' in countries without. The FTT code base does not allow for a new technology to appear if a region does not have any capacity in that technology. To combat this limitation, we add 1% of total wind energy in a country as offshore and 1% of solar PV as CSP in regions without any (and regions with less than 1%). CCS tech is also seeded at 0.01% of current fossil generation and 1% of Biomass for BECCS.
6. Edit the end-years in FTT-Standalone/Utilities/Titles/VariableListing.csv. For instance, change J3 from 2021 to 2022 after you've updated the historical generation to include 2022 data. 
7. Update RERY (how?)

### Costs
1. Update the costs of CAPEX, OPEX and the standard deviation of both using a the average of IRENA data, IEA Energy Prices data, BNEF data, GNESTE. Exeter has access to BNEF data and IEA database. [GNESTE data](https://github.com/iain-staffell/GNESTE) & [IRENA data](https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2024/Sep/IRENA_Renewable_power_generation_costs_in_2023.pdf) are open-access. Do not add raw data from BNEF to the repository; it is not open access! 
    1. Ensure the sources use the same currency (note that $2020USD is different from $2023USD). IRENA data doesn't require this conversion as the currency is $2023.
    2. Fuel costs of coal and gas from IEA have been averaged over 2019-2023 with appropriate conversion of units if necessary.
    3. Assume standard deviation is 30% for CAPEX and OPEX, and verify this assumption using BNEF data, which has a range per country. If the range is smaller or larger, adjust this update manual.
    4. Updated costs and cost update scripts are added to the [FTT_Standalone-support repository](https://github.com/cpmodel/FTT_Standalone-support/tree/main/FTT-Power%20updates/Cost%20update%202024) that generates updated BCET Masterfile.
2. Update the fuel costs
    1.  Fast update: do the same as above for fuel costs. Note that costs are higher for technologies with CCS.
    2.  Higher-quality update: The [UK data services under the International Energy Agency](https://stats2.digitalresources.jisc.ac.uk/index.aspx?r=721229&DataSetCode=IEA_CO2_AB) is down at the moment, but Exeter has access until May 2025, so ask Ian. The dataset is the World Energy Prices Yearly. Sector is Industry. Take the average of the last 5 years to account for fluctuations. Take the sample standard deviation over the last 5 years for the standard deviation of fuel costs. Convert the units for coal and for oil into MWh. 
3. Edit the BCET "History end" in FTT-Standalone/Utilities/Titles/VariableListing.csv. This is found in column J. This ensures learning-by-doing starts in the right year.
4. Verify that LCOE estimates in the model are roughly in accordance with independent estimates. You can compare for instance compare with BNEF if you have access or [Lazard prices](https://www.lazard.com/research-insights/levelized-cost-of-energyplus/).
5. Adjust the currency conversions in the code to the new currency (for instance, from 2013USD to 2023USD). Ideally, this is done automatically from histend.

Note - WB has access to Enerdata that wasn't used in the current, but might be used in the future!

### Calibration (the gamma values)
The FTT model is calibrated to ensure a historical trends do not suddenly change in the absence of new policies. We ensure the first derivative of the shares (MEWS) variable is approximately zero. We estimate a gamma value per country and per technology. 
1. To calibrate the gamma values, run the frontend of the standalone version (FTT_Stand_Alone_Launcher.cmd). Navigate to GAMMA. Initialize the power sector model, and do the following by country
2. Pick a start date which gives you 5 years of historical data, and an end date with 5 year of future data
3. Per technology, choose a gamma value that ensures historical trends continue. The gamma value is considered a "price premium". Positive gamma values will make the technology less attractive, negative values will make it more attractive. If gamma values are often larger than 30, there may be structural errors in the model. Investigate why or contact an experienced modeller. 
4. Save the gamma values in Inputs/_MasterFiles/FTT-P/FTT-P-24x70_2021_S[0-1-2].xlsx. 

## Less frequent updates

### Technical parameters
1. Update the technical potential. The last update for the technical potential for onshore, offshore and solar was done early 2022 (see [the solar momentum paper](https://www.nature.com/articles/s41467-023-41971-7?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20231017&utm_content=10.1038/s41467-023-41971-7#Sec6)) for more details. 
2. Update Cost-Supply Curves to reflect maximum capacity factor by country. These go up steadily over time for solar and wind, as efficiency and capacity factors improve. We used the BNEF data for the 2022 update, and extrapolated based on latitute and proxies. The exact proxies are found in the large CSC files, ask Femke for the latest file.
3. You can also check efficiency and [GHG emissions](https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-iii.pdf#page=7), which change with slow changes of technologies.  Note that emissions are computed bottom-up, but there is a top-down correction to ensure our overall emissions are correct in E3ME.
4. Update efficiencies of storage technologies if needed.
5. Update the learning rate. We use learning rates from literature. It may be worth revisiting every 5 to 10 years, depending on the novelty of the technology and speed of deployment. The last update was done in early 2022 for solar and wind technologies, as well as storage technologies based on [Way et al](https://www.sciencedirect.com/science/article/pii/S254243512200410X). 

### Classification
1. Update which technologies should be included. Are there new technologies we should include? Do we have the data to split rooftop solar from utility-scale solar? Can we add storage technologies / hydrogen? And have some technologies been relegated to the dustbin?
