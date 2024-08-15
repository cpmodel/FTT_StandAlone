---

# FTT-Steel Model Python Modules

## Overview
The FTT-Steel model comprises several Python modules designed to simulate and project various aspects of steel production, including fuel consumption, sales, raw material usage, and scrap rates. The modules facilitate the calibration of historical data and the projection of future trends in the steel sector. The model also simulates the technological diffusion of residential heating technologies within the steel production sector, focusing on consumer decision-making influenced by the Levelised Cost of Steel (LCOS) and market share dynamics.

## Key Components

### Dependencies
- **NumPy**
- **Bottle**
- **Numba**
- **Openpyxl**
- **Pandas**
- **Paste**
- **tqdm**

### Installation
To install the required dependencies, use:

```bash
python -m pip install requirements.txt
```

### How to Run FTT Models
1. Make sure in the `setting.ini` file: `enable_modules = FTT-S`.
2. In a terminal, navigate to your `FTT-STANDALONE-STEEL` folder (e.g., `cd Desktop/user_name/folder_name`).
3. Run the command:

```bash
python run_file.py
```

## Modules

### 1. ftt_s_fuel_consumption.py

#### Description
This script provides a function for calculating fuel consumption and corresponding emissions for various fuel types. The function `ftt_s_fuel_consumption` computes two arrays, `sjef` and `sjco`, which represent fuel consumption and emissions respectively, based on input data arrays.

#### Function
```python
ftt_s_fuel_consumption(bstc, sewg, smed, nrti, nstti, c5ti, njti)
```

#### Parameters
- **bstc** (dict): Representing the fuel consumption data. Each entry corresponds to the amount of a specific fuel type used.
- **sewg** (dict): Representing additional energy-related data.
- **smed** (dict): Providing scaling factors for different fuel types.
- **nrti** (int): The number of regions or time intervals in the `bstc` and `sewg` arrays.
- **nstti** (int): The number of scenarios or time steps in the `bstc` and `sewg` arrays.
- **c5ti** (dict): A dictionary mapping fuel types to their indices in the `bstc` array.
- **njti** (int): The number of fuel types for which calculations are performed.

#### Returns
- **sjef** (numpy array): Representing the total fuel consumption for each fuel type across all regions/time intervals.
- **sjco** (numpy array): Representing the total emissions associated with each fuel type.

#### Function Logic

**Total Consumption of Materials (`sjef`):**
- For each fuel type, the function calculates the total consumption by multiplying the fuel amount in `BSTC` with corresponding scaling factors from `SEWG` and `SMED`, then normalizing by a constant factor (41868).

**Total Consumption of Materials for E3ME Emission Calculation (`sjco`):**
- If a specific condition where for variables in `BSTC` all the `RTI` and `STTI` having `C5TI` as BB? is 1, different scaling factors are applied for emissions calculations, each fuel type's contribution to emissions is scaled by a factor (like 0.1 or -0.9), and multiplied by the cost matrix (`BSTC`), production by technology (`SEWG`), and energy density (`SMED`). The result is then normalized by a constant (41868). Else, calculated without the scaling factors.

### 2. ftt_s_rawmaterials.py

#### Description
This Python function, `raw_material_distr`, calculates the distribution and cost of raw materials used in steel production, incorporating scrap availability, inventory management, emissions, and capital investment costs. The function takes in a data dictionary, titles, the current year, and the number of iterations per year as inputs, and modifies the data dictionary with updated values.

#### Function
```python
raw_material_distr(data, titles, year, t)
```

#### Parameters 
- **bstc** (dict): Representing the fuel consumption data. Each entry corresponds to the amount of a specific fuel type used.
- **titles** (dict): Titles is a container of all permissible dimension titles of the model.
- **year** (int): Current/active year of solution.
- **t** (int): The iteration number for the current calculation cycle.

#### Returns
- **data** (dict): Data is a container that holds all cross-sectional (of time) data for all variables. Variable names are keys and the values are 3D NumPy arrays. The values inside the container are updated and returned to the main routine.

#### Function Logic

**Initialization:**
- `metalinput`: A matrix for storing metal input values. Shape: (26, 3).
- `maxscrapdemand`: Maximum scrap demand matrix. Shape: (71, 26).
- `maxscrapdemand_p`: Partial maximum scrap demand matrix. Shape: (71, 26).
- `scraplimittrade`: Scrap trade limit matrix. Shape: (71, 26).
- `scrapcost`: Scrap cost matrix. Shape: (71, 26).

**Constants:**
- `pitcsr`: Pig iron (or DRI) to crude steel ratio. Set to 1.1, reflecting higher carbon content in pig iron/DRI.

**Scrap Trade Calculation:**

*Initial Calculations:*
- `scrapcost` is assigned from `data['BSTC']` for energy intensity.
- `maxscrapdemand` is calculated as the sum of `scrapcost` multiplied by `data['SEWG']`.
- `maxscrapdemand_p` is calculated for a subset of `scrapcost`.
- `scraplimittrade` is calculated based on a specific index of `scrapcost` and `SEWG`.

**Handling Scrap Shortages and Abundance:**
- If exogenous scrap availability (`data['SXSC']`) is less than `scraplimittrade`, calculate `scrapshortage`.
- If exogenous scrap availability is more than `maxscrapdemand`, calculate `scrapabundance`.
- If global abundance of scrap is sufficient, adjust `data['SXIM']` to cover shortages.
- If global scrap supply is insufficient, adjust `data['SXIM']` proportionally to abundance and shortage.

**Scrap Surplus Calculation:**
- `sxsc` is updated with `data['SXSC']`.
- `sxex` is recalculated to reflect the surplus of scrap.

**Metal Input Calculation:**
- For each region `r` and technology path `path`:
  - If enough scrap is available:
    - Adjust `metalinput` values based on scrap costs and availability.
  - If scrap is insufficient but meets the trade limit:
    - Adjust `metalinput` values proportionally.
  - If scrap is insufficient and below trade limits:
    - Allocate scrap to the Scrap-EAF route and adjust `metalinput`.

**Inventory and Emissions Calculation:**

*Inventory Calculation:*
- Initialize `inventory` matrix.
- Populate the `SLCI` matrix based on technology and material availability.
- Update `inventory` based on steelmaking steps and additional material requirements.

*Emission and Energy Intensity Calculation:*
- Calculate emission factors (`ef`) and biobased emissions.
- Update the `data` dictionary with calculated values:
  - Steel cost components matrix (`BSTC`).
  - Steel technology specific energy intensity factor (`STEI`).
  - Steel technology specific scrap intensity factor (`STSC`).

**CCS Technology Adjustments:**

*If Carbon Capture and Storage (CCS) technology is used (`data['BSTC'][r, a, 22] == 1`):*
- Increase fuel consumption and adjust capital investment costs.
- Update operational costs and emission factors.
- Adjust electricity consumption due to CCS.

*If CCS is not used:*
- Set capital investment costs, operational costs, emission factors, and employment based on the inventory.

---
