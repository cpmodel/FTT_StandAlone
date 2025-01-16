# Instructions for updating FTT-Freight
Author - c.lynch4@exeter.ac.uk

FTT-Freight is a fairly simple FTT model that simulates technology uptake and competition amongst three-wheeler freight vehicles (India only), light-commercial vehicles (vans), medium duty truck, heavy duty trucks, and buses. Because of the very different technology types represented in the model, we chose to prohibit decision-makers switching between vehicle segments. This means a decision-maker cannot replace a diesel LCV with a battery electric bus. As a result, each vehicle segment has its own demand profile and should be considered seperate from one another.
Updating FTT-Freight is similar to FTT-Transport. With the exception of the [IEA's EV outlook](https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer), publically available data is not readily available. Instead, the model makes use of data from the ICCT. A data processing script can be found in the FTT_Standalone-support repository.

## Aggregate fleet sizes (RFLZ)
1. Each vehicle segment has its own projections for fleet size. 
2. The ICCT provides fleet size projections from its Roadmap model for Canada, China, the EU, India, and the US. This covers the period 2020-2050. Using a combination of these projections and historical stock data, we backward extrapolate fleet size to 2010.
3. We further disaggregate the EU projections into EU member states using the split of fleet sizes obtained from the historical stock data (2018-2023).
4. For regions outside of the ICCT's coverage, we apply a rough fleet size estimation based on economic output from the 'Land Transport' sector in E3ME. This is where road freight (among other things) is accounted for in economies. For the countries we do have stock data for, we derive an approxmiate ratio of 45 freight vehicles to 1 million euro of output in this sector. Fleet size numbers for major missing economies (Brazil, Japan, Russia) are checked to verify rough accuracy.
5. The estimated fleet sizes are then split by vehicle segment. We apply the EU's average vehicle splits (that change over time) in regions such as the UK, Japan, and Korea. We apply the US segment splits elsewhere.
6. FTT-Freight also includes three-wheeler freight vehicles in India. As this segment is not included in the ICCT's projections, we estimate three-wheeler stock growth by extrapolating the size of the fleet from the period 2016-2023 where data is provided by WRI-India. 

## Market shares (ZEWS)
1. Historical stock is sourced from the ICCT for China and India (2020-2023) and the EU (at member state level) and the US (2018-2023).
2. Additional stock data on three-wheeler freight vehicles in India is provided by WRI India (2016-2023).
3. Stock is backward extrapolated to 2010 and market shares are calculated. Shares are not a function of the total national stock but rather the share of the total segment stock. Because of this, market shares for a region should sum to 4 (or 5 in India because of three-wheelers) to reflect the number of vehicle segments.
4. For countries without ICCT stock data, we assign shares based on similar freight markets. Australia, Canada, and New Zealand use the same shares as the US. The UK, Norway, Switzerland, Iceland, Japan, and Korea use EU shares. Mexico, Brazil, and Argentina use the US' shares but with 1/2 the number of electric vehicles. Turkey, Macedonia, and Taiwan use the EU's shares but with 1/2 the number of electric vehicles. All other regions use the US' shares but with no electric vehicles assumed.
5. To provide improved data pre-2020 and to improve data quality in some non-ICCT regions, we add in data from the IEA's EV outlook. We prioritise ICCT data where we have it, but when we must extrapolate or use proxies, we overwrite these estimates with the IEA's numbers. This is particularly valuable for regions like the UK, Japan, and Brazil that are not included in the ICCT dataset.
6. The IEA data does not split 'trucks' into medium and heavy duty like the ICCT does. We therefore use data on the split of medium to heavy duty electric trucks observed in the ICCT data and apply this to the IEA data.
7. The IEA data also only provides data on the market share of electric (includes battery electric, plug-in hybrid, and fuel cell) vehicles. We therefore only override estimates for these powertrains and then adjust the market shares of other technologies to ensure shares still sum to 1 within a vehicle segment (i.e., buses).

## Purchase and O&M costs
1. Purchase costs are provided for the EU, the US, and China by the ICCT. Purchase cost data for India is provided by WRI-India. Both of these datasets are incomplete and do not cover the full list of technologies in FTT-Freight.
2. Operation & maintenance costs are provided for the EU, the US, China, and India by the ICCT. Like the purchase cost data, some technologies are missing here.
3. We therefore assume that, where data is missing, purchase and O&M costs for all combustion engine vehicles are the same. Likewise for PHEVs, we assume the same costs as BEVs. While this is not an ideal assumption, PHEVs and ICE vehicles other than diesel have very low (and often zero) market shares.
4. Proxies are selected for other regions. With the exception of European countries, Japan, Korea, and Taiwan (which use EU costs), the US costs are used. 
5. Standard deviations for costs are difficult to obtain. TWVs, LCVs, and MDTs are assumed to have standard deviations of 25% of the puchase cost while HDTs and buses are assumed to have standard deviations of 30% of the purchase cost. This is broadly inline with the data on car prices in FTT-Transport where more data is available. 

## Fuel costs
1. Fuel costs are derived from the **fuel price** (USD /l) multipled by the **fuel use** (litres or kwh /km). The units of fuel costs used in FTT-Freight are therefore USD per vehicle km.
2. Fuel prices are sourced from three datasets. The ICCT data covers fuel prices for diesel, CNG, and hydrogen for EU (most member states), US, China, and India. 
3. The IEA's Transport Fuels dataset is used to enhance coverage of diesel prices in non-ICCT covered regions and to provide petrol prices. We assume these prices to be household prices (i.e., inclusive of VAT)
4. Electricity prices are enhanced using the IEA's Energy Prices Yearly dataset. This is a rich dataset, seperating costs into residential and industry prices but recent data is patchy. Where it is available, we take 2023 industry data. Otherwise, in order of preference, we use 2023 household data, 2022 industry data, or 2022 household data.
5. Both IEA datasets include average prices for the world, the OECD, and Europe. These are used to help fill regions that are still missing data.
6. Commercial vehicles are often exempt from paying VAT. Because of this, where we have used household prices, we adjust values accordingly. We apply country-specific VAT rates for China and India, assume an average EU rate of 20%, and assume 10% in all other regions. 
7. Fuel prices for CNG countries outside of the ICCTs dataset were assumed to be 25% higher than diesel costs in a given country. This was the relationship generally observed in countries with both diesel and CNG fuel data. Similarly for bioethanol, we assumed a fuel price double that of diesel.
8. Fuel use is provided by the ICCT for EU member states, the US, China, and India in units of MJ per km. We convert this to litres (or kwh for BEVs and grams for FCEV).
9. We assume that for the Adv. Petrol and Adv. Disel categories, vehicles are 10% more fuel efficient and we adjust the consumption values accordingly.
10. We also assume that TWVs (which are missing in the ICCT dataset) consume 1/2 less fuel per km than an LCV. This appears to be a sensible approximation based on comparisons of popular vans and three-wheelers in India: see [here](https://trucks.cardekho.com/en/trucks/tata/intra-v10) and [here](https://trucks.cardekho.com/en/trucks/bajaj/maxima-c). 

## Freight/passenger demand
1. Data on the average annual vehicle kilometers travelled is taken from the ICCT's Roadmap model which provides data for Canada, China, EU (at member state level), India, United Kingdom, and United States. Vehicle kilometers are consistent across powertrains (i.e., BEV HDTs travel the same annual distance as diesel HDTs).
2. For other regions, we assume Germany's average numbers with the exception of some larger countries which use the US' numbers.
3. This is supplemented with data on the average loads per vehicle. This is also provided by the ICCT for China, EU (at member state level), India, and the US. Like mileage, average loads are the same across powertrains.
4. We apply the same logic for load factor proxies as mileage.
5. For three-wheeler freight vehicles in India, we estimate that mileage is the midpoint between LCVs (vans) and two and three-wheeler personal vehicles. The reasoning is that three-wheeler freight vehicles are likely to cover more distance than personal transport but unlikely as much as LCVs. Likewise for load factors, we use an estimate of 0.4t/vehicle which is based on capacities of such vehicles observed online. This is only a very rough estimate and can be refined.

## Battery sizes and costs
1. In India, battery sizes for TWV and LCVs were assumed from sampling models from this [website](https://trucks.cardekho.com/en/trucks)
2. In China, HDTs have an average battery size of 292kwh according to this [report](https://theicct.org/publication/ze-mhdv-market-china-january-june-2024-nov24/) by the ICCT. Meanwhile, we assume an LCV battery size of 60kwh based on sampling of top selling vans in China. 
3. **To do - explain where other battery sizes are sourced from.**
4. We take battery costs ($/kwh) from the BloombergNEF Battery Survey (2023 edition) which provides a global battery cost of $139/kwh, and costs of $151, $140, and $126 for Europe, the US, and China respectively.

## Learning rate
1. We assume a battery learning rate of 25% (note: this is converted to a learning exponent in FTT). This is in-line with a number of other studies, see [Ziegler & Trancik (2021)](https://doi.org/10.1039/D0EE02681F) as well as [Lam & Mercure (2021)](https://ore.exeter.ac.uk/repository/bitstream/handle/10871/129774/Lam%20et%20al_Evidence%20for%20a%20global%20EV%20TP.pdf?sequence=1). 
2. The learning rate for the rest of a battery electric vehicle (i.e., excluding the battery pack) is assumed to be 10% based on literature including this [ICCT report](https://theicct.org/publication/purchase-cost-ze-trucks-feb22/).
3. For PHEVs, the non-battery learning rate of 6% is taken from [Weiss et al. 2019](https://doi.org/10.1016/j.jclepro.2018.12.019) and for FCEVs, the learning rate is assumed to be 18% based on [Ruffini & Wei, 2019](https://doi.org/10.1016/j.energy.2018.02.071) and this [IRENA report](https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2020/Nov/IRENA_Green_Hydrogen_breakthrough_2021.pdf).
