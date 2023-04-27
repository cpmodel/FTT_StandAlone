# Instructions for updating FTT:Power
Ideally, FTT:Power is updated every two years. The last data update was done early 2022. Not all data is available yearly. An important data gap is the costs, which the IEA only publishes every five years. **You may need to have different start years for cost and generation in the code**

## Frequent updates
### Historical generation
1. Update the historical generation. We use the IEA World Balances to update generation data. This data is freely available for universities. People with a UK institutional log-in can find it at the [UK data services in the macrodata section](https://stats2.digitalresources.jisc.ac.uk/index.aspx?r=721229&DataSetCode=IEA_CO2_AB). People at CE also have access **Describe how**
    1. The datafiles to update are Inputs/_MasterFiles/FTT-P/FTT-P-24x70_2021_S[0-1-2].xlsx. The generation is in the MEWG sheet
    2. Is there a python file to do this?
2. The IEA World Energy Balances does not distinguish between onshore and offshore. For consistency, we use the overall wind generation data from IEA, but split it out by country using the [historical generation from IRENA](https://www.irena.org/publications/2022/Apr/Renewable-Capacity-Statistics-2022). **Start a pre-processing repo, and ask Marina to add her python script there**
    1. The datafile is the same as above
    2. The file is **ask Marina**
3. The generation data is often not quite up-to-date. You can get more up-to-date capacity data from IRENA. This can be added to the exogenous capacity (policy) variable, the MWKA parameter. Do this for fast-changing technologies if relevant (offshore, solar PV). 
    4. The python file to do this is   
5. For technologies like CSP and offshore, introduce a 'seed' in countries without. The FTT code base does not allow for a new technology to appear if a region does not have any capacity in that technology. To combat this limitation, we add 1% of total wind energy in a country as offshore and 1% of solar PV as CSP in regions without any (and regions with less than 1%). **add my script to new repo**
6. Update MEWE with the cumulative historical capacity by technology. Use the output of the model over the historical period. This is a step that can be automated, so feel free to improve the code.
7. Edit the end-years in FTT-Standalone/Utilities/Titles/VariableListing.xlsx. For instance, change J3 from 2018 to 2019 after you've updated the historical generation to include 2019 data. 

### Calibration (the gamma values)
The FTT model is calibrated to ensure a historical trends do not suddenly change in the absense of new policies. We estimate a gamma value per country and per technology. 
1. To calibrate the gamma values, run the frontend of the standalone version (FTT_Stand_Alone_Launcher.cmd). Navigate to GAMMA. Initialise the power sector model, and do the following by country
2. Pick a start date which gives you 5 years of historical data, and an end date with 5 year of future data
4. Per technology, choose a gamma value that ensures historical trends continue. The gamma value is considered a "price premium". Positive gamma values will make the technology less attractive, negative values will make it more attractive. If gamma values are often larger than 30, there may be structural errors in the model, so feel free to contact an experienced modeller. 
5. Save the gamma values in Inputs/_MasterFiles/FTT-P/FTT-P-24x70_2021_S[0-1-2].xlsx.

### Split the Masterfiles by country
1. The back-end does not read the masterfile directly, but requires 2D files. Use Upload_FTT_data.py in the ``_Masterfiles`` folder to split the data by country.

## Less frequent updates
### Costs
1. Update the costs of CAPEX, OPEX and the  standard deviation of both using the IEA's [Projected Cost of Generating Electricity](https://www.iea.org/reports/projected-costs-of-generating-electricity-2020) file. An [xlsx file can be found](https://iea.blob.core.windows.net/assets/2df33f6b-eba0-4639-926a-bc1c3d3e3268/IEA-NEAProjectedCostsofGeneratingElectricity2020-Datafile.xlsx) with this data at the IEA data server (**how??**) xlsx file at ). We take the standard deviation as the sample standard deviation of the project by country. This data is updated every 5 years, which is suboptimal, so let the team know if you know a different data set. Updates are manual.
3. Update the learning rate. We use learning rates from literature. It may be worth revisiting every 5 to 10 years, depending on the novelty of the technology and speed of deployment. The last update was done in early 2022 for solar and wind technologies, as well as storage technologies.
4. Edit the end-years in FTT-Standalone/Utilities/Titles/VariableListing.xlsx. For instance, change J5 from 2016 to 2020 after you've updated the cost data. 

### Technical parameters
5. Update the technical potential. The last update for the technical potential for onshore, offshore and solar was done early 2022. 
6. Update Cost-Supply Curves to reflect maximum capacity factor by country. For wind, these have gone up steadily over time, as turbines have grown in height.
7. You can also check efficiency and GHG emissions, which change with slow changes of technologies.  Note that emissions are computed bottom-up, but there is a top-down correction to ensure our overall emissions are correct.
8. Update efficiencies of storage technologies if needed.
