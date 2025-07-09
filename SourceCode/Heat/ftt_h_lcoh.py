# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:54:30 2024

@author: Alex Edwards and Femke Nijsse

=========================================
ftt_h_lcoh.py
=========================================
Domestic Heat FTT module.


This is the main file for FTT: Heat, which models technological diffusion of
domestic heat technologies due to consumer decision making. 
Consumers compare the **levelised cost of heat**, which leads to changes in the
market shares of different technologies.

The outputs of this module include levelised cost of heat technologies

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Element-wise divide which replaces divide-by-zeros with zeros


Functions included:
    - get_lcoh
        Calculate levelised cost of heating
        
variables: 
cf = capacity factor
ce = conversion efficiency


"""

# Third party imports
import numpy as np

# Local library imports
from SourceCode.support.divide import divide


def set_carbon_tax(data, c4ti):
    '''
    Convert the carbon price in REPP from euro / tC to 2020 euros / kWhUD. 
    Apply the carbon price to heat sector technologies based on their emission factors

    Returns:
        Carbon costs per country and technology (2D)
    '''
    carbon_costs = (data["REPPHX"][:, :, 0]                              # Carbon price in euro / tC
                    * data['BHTC'][:, :, c4ti['15 Emission factor']]     # kg CO2 / MWh 
                    / 3.666 / 1000 / 1000                                # Conversion from C to CO2 and MWh to kWh, kg to tonne 
                    )
    
    
    if np.isnan(carbon_costs).any():
        #print(f"Carbon price is nan in year {year}")
        print(f"The arguments of the nans are {np.argwhere(np.isnan(carbon_costs))}")
        print(f"Emissions intensity {data['BHTC'][:, :, c4ti['Emission factor']]}")
        
        raise ValueError
                       
    return carbon_costs


# %% LCOH
# --------------------------------------------------------------------------
# -------------------------- LCOH function ---------------------------------
# --------------------------------------------------------------------------

def get_lcoh(data, titles, carbon_costs):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of heat in 2020 Euros/kWh per
    boiler type. It includes intangible costs (gamma values) and together
    determines the investor preferences.
    """
    # Categories for the cost matrix (BHTC)
    c4ti = {category: index for index, category in enumerate(titles['C4TI'])}
    
    # Cost matrix
    bhtc = data['BHTC']
    
    # Heating device lifetimes and build time
    lt = bhtc[:, :, c4ti['5 Lifetime']]
    max_lt = int(np.max(lt))
    full_lt_mat = np.arange(max_lt)
    lt_mask = full_lt_mat <= (lt[..., None] - 1)
    bt_mask = full_lt_mat < np.ones_like(lt[..., None])
    
    # Capacity factor
    cf = data['BHTC'][:, :, c4ti['13 Capacity factor mean'], np.newaxis]
    dcf = data['BHTC'][:, :, c4ti['14 Capacity factor SD'], np.newaxis]
    cf[cf < 0.0001] = 0.0001
    conv_cf = 1 / (cf * 1000)

    # Conversion efficiency
    ce = data['BHTC'][:, : , c4ti['9 Conversion efficiency'], np.newaxis]
    ce[ce < 0.0001] = 0.0001
    conv_ce = 1 / ce
    
    def get_cost_component(base_cost, conversion_factor, mask):
        '''Mask costs during build or life time, and apply
        conversion to generation where appropriate'''
        cost = np.multiply(base_cost[..., None], conversion_factor)
        return np.multiply(cost, mask)
    
    # Investment cost and standard deviation
    it = get_cost_component(bhtc[:, :, c4ti['1 Inv cost mean (EUR/kW)']], conv_cf, bt_mask)
    dit = get_cost_component(bhtc[:, :, c4ti['2 Inv Cost SD']], conv_cf, bt_mask)
    
    # Operation and maintenance costs
    omt = get_cost_component(bhtc[:, :, c4ti['3 O&M mean (EUR/kW)']], conv_cf, lt_mask)
    domt = get_cost_component(bhtc[:, :, c4ti['4 O&M SD']], conv_cf, lt_mask)
    
    # Fuel costs and carbon costs
    ft = get_cost_component(bhtc[:, :, c4ti['10 Fuel cost  (EUR/kWh)']] * data['HEWP'][:, :, 0], conv_ce, lt_mask)
    dft = get_cost_component(bhtc[:, :, c4ti['11 Fuel cost SD']] * ft[:, :, 0], 1, lt_mask)
    ct = get_cost_component(carbon_costs, 1, lt_mask)
    
    # Subsidies or tax on investment costs
    st = get_cost_component(bhtc[:, :, c4ti['1 Inv cost mean (EUR/kW)']] * data['HTVS'][:, :, 0], conv_cf, bt_mask)
    
    # Subsidy or tax on fuel use
    fft = get_cost_component(data['HTRT'][:, :, 0], conv_ce, lt_mask)
    
    # Feed-in tariffs
    fit = get_cost_component(data['HEFI'][:, :, 0], 1, lt_mask)
    
    # Discount rate
    dr = data['BHTC'][:, :, c4ti['8 Discount rate'], np.newaxis]
    denominator = (1 + dr)**full_lt_mat
    
    # 1 – Expenses
    # 1.1 – Without policy costs
    npv_expenses_bare = (it + ft + omt) / denominator
    # 1.2 – Only policy costs
    npv_policy = (st + fft + ct - fit) / denominator
    # 1.3 – Both together
    npv_expenses_all = npv_expenses_bare + npv_policy
    
    # 2 – Standard deviation (propagation of error)
    variance_terms = dit**2 + dft**2 + domt**2
    summed_variance = np.sum(variance_terms / denominator**2, axis=2)
    variance_plus_dcf = summed_variance + (np.sum(npv_expenses_all, axis=2)/cf[:, :, 0]*dcf[:, :, 0])**2
    
    # 3 – Utility
    npv_utility = np.where(lt_mask, 1, 0) / denominator
    
    # 4 – levelised cost variants in Eur/kWhUD
    utility_sum = np.sum(npv_utility, axis=2)
    # 4.1 – Bare LCOH
    lcoh = np.sum(npv_expenses_bare, axis=2) / utility_sum
    # 4.2 – LCOH including policy costs
    tlcoh = np.sum(npv_expenses_all, axis=2) / utility_sum
    # 4.3 – Standard deviation of LCOH
    dlcoh = np.sqrt(variance_plus_dcf) / utility_sum
    
    # LCOH augmented with non-pecuniary costs
    tlcohg = tlcoh * (1 + data['BHTC'][:, :, c4ti['12 Gamma value']])
    
    # Pay-back thresholds
    pb = data['BHTC'][:, :, c4ti['16 Payback time, mean']]
    dpb = data['BHTC'][:, :, c4ti['17 Payback time, SD']]
    
    # Marginal costs of existing units
    tmc = ft[:, :, 0] + omt[:, :, 0] + fft[:, :, 0] - fit[:, :, 0]
    dtmc = np.sqrt(dft[:, :, 0]**2 + domt[:, :, 0]**2)

    # Total pay-back costs of potential alternatives
    tpb = tmc + (it[:, :, 0] + st[:, :, 0]) / pb
    dtpb = np.sqrt(dft[:, :, 0]**2 + domt[:, :,  0]**2
                   + divide(dit[:, :, 0]**2, pb**2)
                   + divide( it[:, :, 0]**2, pb**4)*dpb**2)
    
    # Add gamma values
    tmc = tmc * (1 + data['BHTC'][:, :, c4ti['12 Gamma value']])
    tpb = tpb * (1 + data['BHTC'][:, :, c4ti['12 Gamma value']])
    
    # Pass to variables that are stored outside.
    data['HEWC'][:, :, 0] = lcoh       # The real bare LCOH without taxes
    data['HETC'][:, :, 0] = tlcoh      # The real bare LCOH with taxes
    data['HGC1'][:, :, 0] = tlcohg     # As seen by consumer (generalised cost)
    data['HWCD'][:, :, 0] = dlcoh      # Variation on the LCOH distribution
    data['HGC2'][:, :, 0] = tmc        # Total marginal costs
    data['HGD2'][:, :, 0] = dtmc       # SD of Total marginal costs
    data['HGC3'][:, :, 0] = tpb        # Total payback costs
    data['HGD3'][:, :, 0] = dtpb       # SD of Total payback costs

    return data
