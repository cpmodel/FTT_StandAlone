# Instructions for updating FTT:Power
Ideally, FTT: Transport is updated every year. In particular, it is important to update the share data every year when the latest sales data become available. It is important to update the price and other data in countries undergoing rapid changes in the model variety. 

## Frequent updates
### Update share data
1. Share data are collected per country and per technology. For EU countries, new vehicle sales data by vehicle technology and size are collected from Eurostat (https://ec.europa.eu/eurostat) directly. For countries outside the EU, the data for new registration per vehicle model type and technology are obtained from MarkLines Information Platform (https://www.marklines.com/portal_top_en.html) (private data that requires annual subscription) and matched engine and battery sizes, collected from the official websites of the manufacturers. In some cases when certain models are globally available, it is possible to cross-match technological specification (e.g. collect engine sizes for the US and use the number for the UK) to speed up the share update process. Marklines data is comprehensive and covers 62 countries around the world.   
2. Then the next step is to split the sales data by engine size (for ICEVs including PHEVs) or by battery size (kWh). Currently, to be consistent with Eurostat definition, Econ cars are defined as smaller than 1400cc or 30kWh, Mid cars are defined as engine size between 1400cc and 2000cc or battery size between 30 kwh and 70 kWh, and luxury cars are defined as cars with battery size larger than 70 kWh.  
3. the sales data by technology is added the existing fleet number, considering the historical fleet number (implied in the historical shares)  that needs to be retired from the car population with a survival function. Then, the share by technology can be calculated by fleet number per engine/battery technology. 
4. However, note that it is not always possible to update all the E3ME/FTT regions shares with Marklines raw data every year since the process can sometimes be time consuming. In this case, it is possible to refer to IEA EV Data Tool (https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer) to obtain the total EV and PHEV numbers in a country and assume that the size in the current year follows the previous years (i.e. follow the size split in the previous years). This process is not always accurate because EV market is going through transformation in models (hence the size split may change), but it is better than not carrying out any updates.

### Calibration (the gamma values)
Same process as FTT-Power model. 

### Update cost data 
1. Price data are updated annually (but can be done every two years) for the major regions, including Europe, the US, China, India, Brazil and Japan. For other regions, if not specifically requested, prices are first mapped from a proxy region (e.g. assume that Canada and the US share the same vehicle prices) (see proxy regions in excel sheet x), then the weighted averages and standard deviation are calculated based on the mapped prices. The mapping processes are currently all done in excel. Note that price data are collected and processed alongside the engine and technological specification data. Currently the calculation is carried out in excel (could be coded in the long run). 
2. Fuel cost: collected GlobalPetrolPrices (https://www.globalpetrolprices.com/); World Bank/IMF prices. This variable is connected to E3ME (?) so it is only needed in the standalone version.
3. Rare mineral prices: collected from USGS (https://www.usgs.gov/)

### Update other technological specification
This is only carried out countries where cost data are collected, because the technolgoical specification is collected alongside prices. 
1. Battery sizes: collected in manufacturers’ website alongside with the prices 
2. Fuel economy (ICEVs) and energy/electricity consumption (EVs): same as above
3. Energy intensity of batteries: from literature such as ICCT and BNEF
4. Battery chemistry: this part requires annual update because of the rapid changes in battery chemistry

## Less frequent updates
### Other cost data and learning rates
1. Maintenance cost: Collect from a 2021 U.S. Department of Energy comprehensive quantification of total ownership costs, including maintenance costs for BEVs, PHEVs, HEVs, and ICEVs.
2. Update the learning rate. We use learning rates from literature. It is worth updating every 2-3 years for EVs, but more importantly for fuel cell vehicles where learning rates are very uncertain. 
3. Edit the end-years in FTT-Standalone/Utilities/Titles/VariableListing.xlsx. 

### Technical parameters
5. Mechanical survival rates: Taken from literature for ICEVs and assumed that EVs follow the same survival rates. Updates on the mechanical survival rates for EVs and fuel cell cars are recommended when the data become available

