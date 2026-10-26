# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:54:3np.zeros(len(titles['STTI'])) 2024

@author: Varun

=========================================
ftt_s_lcos.py
=========================================
Domestic Heat FTT module.
####################################

This is the LOOS file for FTT: Steel, which calculates levelized cost of steel production

The outputs of this module include levelised cost of steel production technologies

Local library imports:

Support functions:

- `divide <divide.html>`__
    Bespoke element-wise divide which replaces divide-by-zeros with zeros
- `estimation <econometrics_functions.html>`__
    Predict future values according to the estimated coefficients.

Parameters
----------
- get_lcos(data, titles, spsa_dt)
- data: dictionary
    Data is a container that holds all cross-sectional (of time) for all
    variables. Variable names are keys and the values are 3D NumPy arrays.
- titles: dictionary
    Titles is a container of all permissible dimension titles of the model.
- spsa_dt: 1D NumPy array
    Steel demand by region in the current time step. Only regions with
    positive demand are updated.

Returns
----------
data: dictionary
    Data is a container that holds all cross-sectional (of time) data for
    all variables.
    Variable names are keys and the values are 3D NumPy arrays.
    The values inside the container are updated and returned to the main
    routine.

"""

# Third party imports
import numpy as np
import math

# Local library imports
from SourceCode.support.divide import divide

# %% LCOS
# --------------------------------------------------------------------------
# -------------------------- LCOS function ---------------------------------
# --------------------------------------------------------------------------

def get_lcos(data, titles, spsa_dt):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of steel production per
    integrated route, first calculating the levelised cost of ironmaking
    (used to price pig iron for the Scrap-EAF route). It includes intangible
    costs (gamma values) and together determines the investor preferences.

    The calculations are vectorised over regions, routes and years of the
    plant lifetime. Only regions with positive steel demand (spsa_dt) are
    updated.
    """
    n_mats = len(titles['SMTI'])
    spsa_dt = np.asarray(spsa_dt).reshape(-1)
    active = np.flatnonzero(spsa_dt > 0.0)
    bstc = data['BSTC']
    scmm = data['SCMM'][0]
    # Carbon price ($(2008)/tCO2) by region
    carbon_price = data['SCTX'][:, 0, 0]

    # Prices including taxes on materials, from the prices at the start of the call
    inputprices_taxes = (1 + data['STRT'][:, :, 0]) * data['SPMT'][:, :, 0]

    # %% Levelised cost of ironmaking
    # Not all integrated steelmaking routes have an Iron making step. Of the
    # plants in the ironmaking range of the SSTI classification used by a route
    # ('Conv. BF' (7) to 'HFS' (19), as in ftt_s_rawmaterials), the last one
    # determines the material/energy inventory.
    uses_plant = data['STIM'][0, :, 7:20] == 1
    has_iron = uses_plant.any(axis=1)
    iron_plant = 19 - np.argmax(uses_plant[:, ::-1], axis=1)

    # First, add IronmakingP
    inv_mat_I = scmm[:, iron_plant].T
    # Second, add sinter and pellet
    inv_mat_I = (inv_mat_I + inv_mat_I[:, 6:7] * scmm[:, 2] + inv_mat_I[:, 7:8] * scmm[:, 3]
                 + inv_mat_I[:, 8:9] * scmm[:, 4] + inv_mat_I[:, 9:10] * scmm[:, 5])
    # Third, add coke
    inv_mat_I = inv_mat_I + inv_mat_I[:, 4:5] * scmm[:, 0] + inv_mat_I[:, 5:6] * scmm[:, 1]
    # Fourth, add oxygen
    inv_mat_I = inv_mat_I + inv_mat_I[:, 10:11] * scmm[:, 6]

    # Increase fuel consumption if CCS (region x path x material)
    ccs = bstc[:, :, 21] == 1
    inv_mat_I = np.repeat(inv_mat_I[None, :, :], bstc.shape[0], axis=0)
    inv_mat_I[:, :, 11:15][ccs] *= 1.1
    inv_mat_I[:, :, 17:20][ccs] *= 1.1

    IC_I = scmm[20, iron_plant]
    OM_I = scmm[21, iron_plant]
    CF_I = 0.9

    # Emission factors (total, fossil only, biobased only) and energy intensity
    smef = data['SMEF'][0, :, 0]
    EF_I = np.sum(inv_mat_I * smef, axis=2)
    EF_I_Fossil = np.sum(inv_mat_I[:, :, :15] * smef[:15], axis=2)
    EF_I_Biobased = np.sum(inv_mat_I[:, :, 15:24] * smef[15:24], axis=2)
    SIEI = np.sum(inv_mat_I * data['SMED'][0, :, 0], axis=2)

    # Total cost and total cost including tax
    TotCost_I = np.sum(inv_mat_I * data['SPMT'][:, None, :, 0], axis=2)
    TaxTotCost_I = np.sum(inv_mat_I * inputprices_taxes[:, None, :], axis=2)

    r_I = bstc[:, :, 9]
    L_I = bstc[:, :, 5]
    B_I = bstc[:, :, 6]
    Sub_I = data['SEWT'][:, :, 0]

    # Emission factor, reduced if CCS
    SIEF = np.where(ccs, 0.1 * EF_I - EF_I_Biobased, EF_I_Fossil)

    # CO2 costs of iron production, on the (CCS-adjusted) fossil emissions
    CO2_I = SIEF * carbon_price[:, None]

    # Levelised cost: investment during the building years, then material,
    # O&M and CO2 costs during the lifetime
    n_t_I = (B_I + L_I).astype(int)
    t = np.arange(max(n_t_I.max(), 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        It_I = IC_I / (CF_I * B_I)
        St_I = Sub_I * It_I
        discount = (1 + r_I[:, :, None]) ** t
        building = t < B_I[:, :, None]
        in_life = t < n_t_I[:, :, None]
        costs = np.where(building, (It_I + St_I)[:, :, None],
                         (TaxTotCost_I + OM_I + CO2_I)[:, :, None])
        NPV1p_I = np.sum(np.where(in_life, costs / discount, 0.0), axis=2)
        # Iron is only produced after construction
        NPV2p_I = np.sum(np.where(in_life & ~building, 1 / discount, 0.0), axis=2)
        SITC = NPV1p_I / NPV2p_I

    # CO2 costs of iron production, as left by the last year of the lifetime loop
    SCOI = np.where(n_t_I - 1 >= B_I, CO2_I, np.where(n_t_I >= 1, 0.0, data['SCOI'][:, :, 0]))

    # Store results for routes with ironmaking in regions with steel demand
    data['SITC'][active] = 0.0
    data['SWGI'][active] = 0.0
    iron = np.zeros(bstc.shape[:2], dtype=bool)
    iron[active] = has_iron
    data['SITC'][:, :, 0][iron] = SITC[iron]
    data['SWGI'][:, :, 0][iron] = (data['SEWG'][:, :, 0] * CF_I)[iron]
    data['SIEF'][:, :, 0][iron] = SIEF[iron]
    data['SIEI'][:, :, 0][iron] = SIEI[iron]
    data['SCOI'][:, :, 0][iron] = SCOI[iron]

    #The average price of pig iron is the levelised costs of ironmaking times the share of each ironmaking technology
    for r in active:
        data['SIPR'][r] = 0.0
        #Total production of intermediate iron product
        data['STGI'][r] = np.sum(data['SWGI'][r, :])
        if data['STGI'][r] > 0.0:
            data['SIPR'][r] = np.sum(data['SITC'][r, :] * data['SWGI'][r, :]) / np.sum(data['SWGI'][r,:])

    #Global average price:
    mean_price_I = np.sum(data['SIPR'] * data ['STGI'] / np.sum(data['STGI']))

    #Global average EF for intermediate iron (to connect to imported iron by Scrap - EAF route)
    mean_EF_I = np.sum(data['SIEF'] * data['SWGI']) / np.sum (data['STGI'])

    # %% Price of intermediate iron and emission factor of the Scrap-EAF route
    for r in active:
        iron_demand = data['BSTC'][r, 25, 24] * data['SEWG'][r, 25, 0]
        iron_supply = 0.0

        EF_sec_route = 0.0
        # Iron used by the routes themselves
        iron_used = np.sum(data['BSTC'][r, :, 23] * data['SEWG'][r, :, 0])
        if data['STGI'][r, 0, 0] > iron_used:
            iron_supply = data['STGI'][r, 0, 0] - iron_used

        if iron_demand > 0.0:
            #The price of intermediate iron can only be used directly if there's enough iron production
            if data['SIPR'][r] > 0.0 and data['SIPR'][r] < 3 * data['SPMT'][r, 2] and iron_supply >= iron_demand:
                data['SPMT'][r, 2] = data['SIPR'][r]
            #If there isn't enough domestic supply of iron, then a part of the price has to be calculated from global price and global EF of iron production resp.
            elif data['SIPR'][r] > 0.0 and data['SIPR'][r] < 3 * data['SPMT'][r, 2] and iron_supply > 0.0 and iron_demand > iron_supply:
                data['SPMT'][r, 2] = ((iron_demand - iron_supply) * mean_price_I + iron_supply * data ['SIPR'][r]) / iron_demand
            #If there's no iron production (only MOE or Scrap-EAF present) or local price is ridiculously high, then take global average.
            else:
                data['SPMT'][r, 2] =  mean_price_I

            #Emissions related to iron production that is used in the Scrap-EAF route need to be allocated to this route
            #There's enough domestic supply to cover the domestic demand
            if iron_supply >= iron_demand:
                EF_sec_route = np.sum (data['SIEF'][r, :] * data['SWGI'][r, :] / data['STGI'][r])
            #If there's some iron supply, but not enough demand,partly use the global mean
            elif iron_supply < iron_demand and iron_supply > 0.0:
                EF_sec_route = ((iron_demand - iron_supply) * mean_EF_I + np.sum (data['SIEF'][r, :] * data['SWGI'] [r, :] / data['STGI'][r]) * iron_supply) / iron_demand
            #If there's no iron supply at all, use global mean completely.
            else:
                EF_sec_route = mean_EF_I

        #Adjust emission factor for Scrap-EAF due to emissions.
        data['STEF'][r, 25] = data['BSTC'][r, 25, 13] + data['BSTC'][r, 25, 24] * EF_sec_route

    # %% Levelised cost of steelmaking (region x path x year of lifetime)
    bstc = bstc[active]
    IC = bstc[:, :, 0]
    dIC = bstc[:, :, 1]
    OM = bstc[:, :, 2]
    dOM = bstc[:, :, 3]
    L = bstc[:, :, 5]
    B = bstc[:, :, 6]
    dr = bstc[:, :, 9]
    Gam = bstc[:, :, 10]
    CF = bstc[:, :, 11]
    EF_BSTC = bstc[:, :, 13]
    sewt = data['SEWT'][active, :, 0]
    scot = data['SCOT'][active, :, 0]

    # Material costs, using the updated price of intermediate iron
    n_cost_mats = n_mats - 4
    use = bstc[:, :, 22:22 + n_cost_mats]
    prices = data['SPMT'][active, None, :n_cost_mats, 0]
    TotCost = np.sum(use * prices, axis=2)
    TaxTotCost = np.sum(use * inputprices_taxes[active, None, :n_cost_mats], axis=2)
    dTotCost = np.sum(0.1 * use * prices, axis=2)

    n_t = (B + L).astype(int)
    t = np.arange(max(n_t.max(initial=0), 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        discount = (1 + dr[:, :, None]) ** t
        building = t < B[:, :, None]
        in_life = t < n_t[:, :, None]

        def npv(build_value, life_value):
            """Discounted sum over the lifetime, with values during and after construction"""
            value = np.where(building, build_value[:, :, None], life_value[:, :, None])
            return np.sum(np.where(in_life, value / discount, 0.0), axis=2)

        zeros = np.zeros_like(IC)
        # Investment costs are divided over each building year
        It = IC / (CF * B)
        dIt = dIC / (CF * B)
        # Tech-specific subsidy or tax
        St = sewt * It
        # CO2 costs (SCOT) after construction: we use the emission factors
        # stored in BSTC rather than the ones stored in STEF. EFs in STEF are
        # altered to take into account the emissions due to iron production
        # that is used in the scrap-EAF route. These emissions do not take place
        # directly in the Scrap-EAF route and therefore this technology does
        # not bare the costs.
        CO2t = EF_BSTC * carbon_price[active, None]
        dCO2t = 0.1 * CO2t

        NPV1 = npv(It, TotCost + OM)              #Discounted costs for the LCOS calculation
        NPV1p = npv(It + St, TaxTotCost + OM + CO2t)  #Discounted costs for the TLCOS/LTLCOS calculation
        NPV1o = npv(It + St, TaxTotCost + OM)
        dNPV = npv(np.sqrt(dIt ** 2), np.sqrt(dTotCost ** 2 + dOM ** 2))   #SD for the LCOS calculation
        dNPVp = npv(np.sqrt(dIt ** 2), np.sqrt(dTotCost ** 2 + dOM ** 2 + dCO2t ** 2))  #SD for the TLCOS/LTLCOS calculation
        ICC = npv(It + St, zeros)                 #Investment cost component of LTLCOS
        FCC = npv(zeros, TaxTotCost)              #Fuel/Material cost component of LTLCOS
        OMC = npv(zeros, OM)                      #O&M cost component of LTLCOS
        CO2C = npv(zeros, CO2t)                   #CO2 tax cost component of LTLCOS
        NPV2 = npv(zeros, zeros + 1.0)            #Denominator for the LCOS calculation: production of steel (0 during leadtime, 1 during lifetime)

        LCOS = NPV1 / NPV2
        dLCOS = dNPV / NPV2
        LCOSprice = NPV1o / NPV2
        TLCOS = NPV1p / NPV2
        dTLCOS = dNPVp / NPV2
        LTLCOS = TLCOS + Gam

        # Costs as left by the last year of the lifetime loop
        in_operation = n_t - 1 >= B
        Ft = np.where(in_operation, TotCost, 0.0)
        dOMt = np.where(in_operation, dOM, 0.0)
        scot = np.where(in_operation, CO2t, scot)

        #Variables for endogenous scrapping
        PB = bstc[:, :, 19] #payback threshold
        dPB = 0.3 * PB
        It = bstc[:, :, 0] / CF
        dIt = bstc[:, :, 1] / CF
        St = sewt * It

        #Marginal cost calculation (for endogenous scrapping) (fuel cost+OM cost+fuel and CO2 tax policies)
        TMC = Ft + OM + TaxTotCost + scot
        dTMC = np.sqrt(dTotCost ** 2 + dOMt ** 2)

        #Payback cost calculation (for endogenous scrapping) (TMC+(investment cost+investment subsidy)/payback threshold)
        TPB = TMC + (It + St) / PB
        dTPB = np.sqrt(dTotCost ** 2 + dOMt ** 2 + dIt ** 2 / PB ** 2 + (It ** 2 / PB ** 4) * dPB ** 2)

        data['SCOT'][active, :, 0] = scot
        data['SGC2'][active, :, 0] = TMC + Gam
        data['SGC3'][active, :, 0] = TPB + Gam
        data['SGD2'][active, :, 0] = dTMC
        data['SGD3'][active, :, 0] = dTPB

        #For the cost components we exclude the gamma value as we assume the gamma value is accordingly distributed over the cost components
        data['SWIC'][active, :, 0] = ICC / NPV2  # Investment cost component of TLCOS
        data['SWFC'][active, :, 0] = FCC / NPV2  # Material cost component of TLCOS
        data['SOMC'][active, :, 0] = OMC / NPV2  # O&M cost component of TLCOS
        data['SCOC'][active, :, 0] = CO2C / NPV2 # CO2 cost component of TLCOS

    data['SEWC'][active, :, 0] = LCOS
    data['SETC'][active, :, 0] = LCOSprice
    data['SGC1'][active, :, 0] = LTLCOS

    data['SDWC'][active, :, 0] = dLCOS
    data['SGD1'][active, :, 0] = dTLCOS

    # Average steel prices, weighted by market shares
    mkt_shares = data['SEWS'][active, :, 0]
    data['SPRI'][active, 0, 0] = np.sum(LTLCOS * mkt_shares, axis=1)
    data['SPRC'][active, 0, 0] = np.sum(LCOSprice * mkt_shares, axis=1)

    return data