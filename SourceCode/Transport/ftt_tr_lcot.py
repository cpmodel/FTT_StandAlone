# -*- coding: utf-8 -*-
"""
=========================================
ftt_tr_lcot.py
=========================================

Get the levelised cost of transport, in lognormal space


Local library imports:

Functions included:
    - solve
        Main solution function for the module
    - get_lcot
        Calculate levelised cost of transport
        
variables: 
cf = capacity factor
ff = filling factors
ns = number of seats
      
        
"""
import numpy as np


# %% lcot
# -----------------------------------------------------------------------------
# --------------------------- LCOT function -----------------------------------
# -----------------------------------------------------------------------------
@profile
def get_lcot(data, titles, year):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of transport in 2012$/p-km per
    vehicle type. It includes intangible costs (gamma values) and together
    determines the investor preferences.
    """

    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}

    # Taxable categories for fuel tax, CNG, EVs and H2 exempt
    taxable_fuels = np.ones([len(titles['RTI']), len(titles['VTTI']), 1])
    taxable_fuels[:, 12:15] = 0   # CNG
    taxable_fuels[:, 18:21] = 0   # EVs
    taxable_fuels[:, 24:27] = 0   # Hydrogen
    
    # Taxable categories for carbon tax: only EVs and H2 exempt
    tf_carbon = np.ones([len(titles['VTTI']), 1])
    tf_carbon[18:21] = 0   # EVs
    tf_carbon[24:27] = 0   # Hydrogen
    
    bttc = data['BTTC']
    
    # Lifetimes and build years
    lt = bttc[:, :, c3ti['8 lifetime']]
    max_lt = int(np.max(lt))
    full_lt_mat = np.arange(max_lt)
    lt_mask = full_lt_mat <= (lt[..., None] - 1)        # Life time mask
    bt_mask = full_lt_mat < np.ones_like(lt[..., None]) # Build time mask
    
    # Capacity factor
    cf = bttc[:, :, c3ti['12 Cap_F (Mpkm/kseats-y)'], np.newaxis]

    # Occupancy rates
    ff = bttc[:, :, c3ti['11 occupancy rate p/sea'], np.newaxis]

    # Number of seats
    ns = bttc[:, :, c3ti['15 Seats/Veh'], np.newaxis]

    # Energy use
    en = bttc[:, :, c3ti['9 energy use (MJ/km)'], np.newaxis]
    
    conv_full = 1 / ns / ff / cf / 1000
    conv_pkm = 1 / ns / ff
    
    
    # upfront = (bttc[:, :, c3ti['1 Prices cars (USD/veh)']] * conv_full)
    
    # upfront_pol = 
    #            * (1 + data["Base registration rate"][:, :, 0])
    #            + (data['TTVT'][:, :, 0] 
    #            + data['RTCO'][:, 0] * bttc[:, :, c3ti['14 CO2Emissions']] )
    #            * conv_full[:, :, 0] )
    
    # upfront_sd = bttc[:, :, c3ti['2 Std of price']] * conv_full
    
    # variable = (bttc[:, :, c3ti['3 fuel cost (USD/km)']] * conv_pkm
    #             + bttc[:, :, c3ti['5 O&M costs (USD/km)']] * 
    
    
    def get_cost_elem(base_cost, conversion_factor, mask):
        '''Mask costs during build or life time, and apply
        conversion to generation where appropriate'''
        cost = np.multiply(base_cost[..., None], conversion_factor)
        return np.multiply(cost, mask)
    
    it = get_cost_elem(bttc[:, :, c3ti['1 Prices cars (USD/veh)']], conv_full, bt_mask)
    dit = get_cost_elem(bttc[:, :, c3ti['2 Std of price']], conv_full, bt_mask)
    # Vehicle tax at purchase
    vtt = get_cost_elem( ( (data['TTVT'][:, :, 0] 
                             + data['RTCO'][:, 0] * bttc[:, :, c3ti['14 CO2Emissions']] )
                             * conv_full[:, :, 0]
                             + data["Base registration rate"][:, :, 0] * it[:, :, 0]
                            ), 1, bt_mask)
    ft = get_cost_elem(bttc[:, :, c3ti['3 fuel cost (USD/km)']], conv_pkm, lt_mask)
    dft = get_cost_elem(bttc[:, :, c3ti['4 std fuel cost']], conv_pkm, lt_mask)
    # Fuel tax costs
    # RTFT must be converted from $/litre to $/MJ (assuming 35 MJ/l)
    fft = get_cost_elem(data['RTFT'][:, :, 0] / 35, en / ns / ff * taxable_fuels, lt_mask )
    omt = get_cost_elem(bttc[:, :, c3ti['5 O&M costs (USD/km)']], 1 / ns / ff, lt_mask)
    domt = get_cost_elem(bttc[:, :, c3ti['6 std O&M']], 1 / ns / ff, lt_mask)
    # Yearly road tax cost
    rtt = get_cost_elem(data['TTRT'][:, :, 0], conv_full, lt_mask)
    
    # Discount rate
    dr = bttc[:, :, c3ti['7 Discount rate'], np.newaxis]
    denominator = (1+dr)**full_lt_mat
    
    # A faster way to implement this is with cumprod, but less readable. Do we want that?
    # Need to check if this is faster
    # disc_factors = 1 / (1 + dr[..., 0])
    # denominator = np.cumprod(np.repeat(disc_factors[:, :, None], max_lt, axis=2), axis=2)
    
    # 1 – Expenses
    # 1.1 – Without policy costs
    npv_expenses_bare = (it + ft + omt) / denominator
    # 1.2 – With policy costs
    npv_expenses_policy = (it + vtt + ft + fft + omt + rtt) / denominator
   
    # 2 – Utility
    npv_utility = 1 / denominator
    # Remove utility after end lifetime
    npv_utility = np.where(lt_mask, 1, 0) / denominator
    utility_sum = np.sum(npv_utility, axis=2)
    
    # 3 – Standard deviation (propagation of error)
    # Calculate variance terms and apply discounting
    variance_terms = dit**2 + dft**2 + domt**2
    summed_variance = np.sum(variance_terms/(denominator**2), axis=2)
    # Assume a 10% variation in load factors
    variance_plus_dcf = summed_variance + (np.sum(npv_expenses_policy, axis=2) * 0.1)**2
    dlcot = np.sqrt(variance_plus_dcf) / utility_sum

    # 4 – Levelised cost variants in $/pkm
    # 1.1 – Bare LCOT
    lcot = np.sum(npv_expenses_bare, axis = 2) / utility_sum
    # 1.2 – LCOT including policy costs
    tlcot = np.sum(npv_expenses_policy, axis = 2) / utility_sum
    # 1.3 – LCOT augmented with non-pecuniary costs (logtlcot used in further calculations)
    tlcotg = tlcot * (1 + bttc[:, :, c3ti['13 Gamma']])

    # 5 - Transform into lognormal space
    logtlcot = ( np.log(tlcot * tlcot / np.sqrt(dlcot * dlcot + tlcot * tlcot)) 
                + bttc[:, :, c3ti['13 Gamma']])
    dlogtlcot = np.sqrt(np.log(1.0 + dlcot * dlcot / (tlcot * tlcot)))
    
    # 6 - Pass to variables that are stored outside.
    data['TEWC'][:, :, 0] = lcot           # The real bare LCOT without taxes
    data['TETC'][:, :, 0] = tlcot          # The real bare LCOT with taxes
    data['TEGC'][:, :, 0] = tlcotg         # As seen by consumer (generalised cost)
    data['TELC'][:, :, 0] = logtlcot       # In lognormal space
    data['TECD'][:, :, 0] = dlcot          # Variation on the LCOT distribution
    data['TLCD'][:, :, 0] = dlogtlcot      # Log variation on the LCOT distribution

    return data
