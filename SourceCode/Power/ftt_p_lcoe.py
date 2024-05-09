# -*- coding: utf-8 -*-
"""
=========================================
ftt_p_lcoe.py
=========================================
Power LCOE FTT module.


Functions included:
    - get_lcoe
        Calculate levelized costs

"""

# Standard library imports
from copy import deepcopy as dc

# Third party imports
import numpy as np

# Local library imports
from SourceCode.support.divide import divide


# %% lcoe
# -----------------------------------------------------------------------------
# --------------------------- LCOE function -----------------------------------
# -----------------------------------------------------------------------------
def get_lcoe(data, titles, histend, year):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of electricity in $2013/MWh. It includes
    intangible costs (gamma values) and determines the investor preferences.

    Parameters
    -----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) for all
        variables. Variable names are keys and the values are 3D NumPy arrays.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.

    Returns
    ----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
        The values inside the container are updated and returned to the main
        routine.

    Notes
    ---------
    Additional notes.
    BCET = cost matrix 
    MEWL = Average capacity factor
    MEWT = Subsidies
    MTFT = Fuel tax
    
    """

    # Categories for the cost matrix (BCET)
    c2ti = {category: index for index, category in enumerate(titles['C2TI'])}

    for r in range(len(titles['RTI'])):
        
        # Value factor section
        if year > histend['DPVF']: 
            # Calculate generation share change compared to histend
            share_change = np.where(data['MPGS'][r,:,0]>0.0, 
                                    divide(data['MPGS'][r,:,0], data['MPGS2023'][r,:,0])-1,
                                    0.0)
            
            # Calculate value factor
            data['DPVF'][r,:,0] = data['DPVF2023'][r,:,0] + data['DVFE'][r,:,0] * share_change
            
        else:
            data['DPVF'][r,:,0] = dc(data['DPVF2023'][r,:,0])
            
        # WACC section
        if year > histend['DCOC']: 
            
            # Calculate WACC rate
            data['DCOC'][r,:,0] =  data['DCOC2015'][r,:,0] - data['RLR2015'][r,0,0] + data['RLR'][r,0,0]
            
        else:
            data['DCOC'][r,:,0] = dc(data['DCOC2015'][r,:,0])        
            

        # Cost matrix
        bcet = data['BCET'][r, :, :]

        # Plant lifetime
        lt = bcet[:, c2ti['9 Lifetime (years)']]
        bt = bcet[:, c2ti['10 Lead Time (years)']]
        max_lt = int(np.max(bt+lt))
        
        # Define (matrix) masks to turn off cost components before or after contruction 
        full_lt_mat = np.linspace(np.zeros(len(titles['T2TI'])), max_lt-1,
                                  num=max_lt, axis=1, endpoint=True)
        lt_max_mat = np.concatenate(int(max_lt) * [(lt+bt-1)[:, np.newaxis]], axis=1)
        bt_max_mat = np.concatenate(int(max_lt) * [(bt-1)[:, np.newaxis]], axis=1)
        
        bt_mask = full_lt_mat <= bt_max_mat
        bt_mask_out = full_lt_mat > bt_max_mat
        lt_mask_in = full_lt_mat <= lt_max_mat
        lt_mask = np.where(lt_mask_in == bt_mask_out, True, False)
        
        # Capacity factor of marginal unit (for decision-making)
        cf_mu = bcet[:, c2ti['11 Decision Load Factor']].copy()
        # Trap for very low CF
        cf_mu[cf_mu<0.000001] = 0.000001
        # Factor to transfer cost components in terms of capacity to generation
        conv_mu = 1/bt / cf_mu/8766*1000
        
        # Average capacity factor (for electricity price)
        cf_av = data['MEWL'][r, :, 0]
        # Trap for very low CF
        cf_av[cf_av<0.000001] = 0.000001
        # Factor to transfer cost components in terms of capacity to generation
        conv_av = 1/bt / cf_av/8766*1000        

        # Discount rate
        # dr = bcet[6]
        dr = data['DCOC'][r, :, 0, None] #bcet[:, c2ti['17 Discount Rate (%)'], np.newaxis]

        # Initialse the levelised cost components
        # Average investment cost of marginal unit
        it_mu = np.ones([len(titles['T2TI']), int(max_lt)])
        it_mu = it_mu * bcet[:, c2ti['3 Investment ($/kW)'], np.newaxis] * conv_mu[:, np.newaxis]
        it_mu = np.where(bt_mask, it_mu, 0)
        
        # Average investment costs of across all units
        it_av = np.ones([len(titles['T2TI']), int(max_lt)])
        it_av = it_av * bcet[:, c2ti['3 Investment ($/kW)'], np.newaxis] * conv_av[:, np.newaxis]
        it_av = np.where(bt_mask, it_av, 0)       

        # Standard deviation of investment cost - marginal unit
        dit_mu = np.ones([len(titles['T2TI']), int(max_lt)])
        dit_mu = dit_mu * bcet[:, c2ti['4 std ($/MWh)'], np.newaxis] * conv_mu[:, np.newaxis]
        dit_mu = np.where(bt_mask, dit_mu, 0)

        # Standard deviation of investment cost - average of all units
        dit_av = np.ones([len(titles['T2TI']), int(max_lt)])
        dit_av = dit_av * bcet[:, c2ti['4 std ($/MWh)'], np.newaxis] * conv_av[:, np.newaxis]
        dit_av = np.where(bt_mask, dit_av, 0)

        # Subsidies - only valid for marginal unit
        st = np.ones([len(titles['T2TI']), int(max_lt)])
        st = (st * bcet[:, c2ti['3 Investment ($/kW)'], np.newaxis]
              * data['MEWT'][r, :, :] * conv_mu[:, np.newaxis])
        st = np.where(bt_mask, st, 0)

        # Average fuel costs
        ft = np.ones([len(titles['T2TI']), int(max_lt)])
        ft = ft * bcet[:, c2ti['5 Fuel ($/MWh)'], np.newaxis]
        # TODO: Temporarily get MWFC from E3ME run
        # ft2 = ft * data['MWFCX'][r, :, :]
        ft = np.where(lt_mask, ft, 0)

        # Standard deviation of fuel costs
        dft = np.ones([len(titles['T2TI']), int(max_lt)])
        dft = dft * bcet[:, c2ti['6 std ($/MWh)'], np.newaxis]
        dft = np.where(lt_mask, dft, 0)

        # fuel tax/subsidies
        fft = np.ones([len(titles['T2TI']), int(max_lt)])
        fft = ft * data['MTFT'][r, :, 0, np.newaxis]
        fft = np.where(lt_mask, ft, 0)

        # Average operation & maintenance cost
        omt = np.ones([len(titles['T2TI']), int(max_lt)])
        omt = omt * bcet[:, c2ti['7 O&M ($/MWh)'], np.newaxis]
        omt = np.where(lt_mask, omt, 0)

        # Standard deviation of operation & maintenance cost
        domt = np.ones([len(titles['T2TI']), int(max_lt)])
        domt = domt * bcet[:, c2ti['8 std ($/MWh)'], np.newaxis]
        domt = np.where(lt_mask, domt, 0)

        # Carbon costs
        ct = np.ones([len(titles['T2TI']), int(max_lt)])
        ct = ct * bcet[:, c2ti['1 Carbon Costs ($/MWh)'], np.newaxis]
        ct = np.where(lt_mask, ct, 0)

        # Energy production over the lifetime (incl. buildtime)
        # No generation during the buildtime, so no benefits
        et = np.ones([len(titles['T2TI']), int(max_lt)])
        et = np.where(lt_mask, et, 0)
        
        # Grid costs
        gridcost = np.ones([len(titles['T2TI']), int(max_lt)])
        gridcost = data['DTGC'][r, :, 0, None] * conv_mu[:, np.newaxis]
        gridcost = np.where(bt_mask, gridcost, 0)

        # Storage costs and marginal costs (lifetime only)
        stor_cost = np.ones([len(titles['T2TI']), int(max_lt)])
        marg_stor_cost = np.ones([len(titles['T2TI']), int(max_lt)])

        if np.rint(data['MSAL'][r, 0, 0]) in [2]:
            stor_cost = stor_cost * (data['MSSP'][r, :, 0, np.newaxis] +
                                     data['MLSP'][r, :, 0, np.newaxis])/1000
            marg_stor_cost = marg_stor_cost * 0
        elif np.rint(data['MSAL'][r, 0, 0]) in [3, 4, 5]:
            stor_cost = stor_cost * (data['MSSP'][r, :, 0, np.newaxis] +
                                     data['MLSP'][r, :, 0, np.newaxis])/1000
            marg_stor_cost = marg_stor_cost * (data['MSSM'][r, :, 0, np.newaxis] +
                                          data['MLSM'][r, :, 0, np.newaxis])/1000
        else:
            stor_cost = stor_cost * 0
            marg_stor_cost = marg_stor_cost * 0

        stor_cost = np.where(lt_mask, stor_cost, 0)
        marg_stor_cost = np.where(lt_mask, marg_stor_cost, 0)
        
        # Debt repayment costs
        dbt = (it_av+fft+st+ft+ct+omt+stor_cost)*(data['DPDR'][r, :, :]*data['RLR'][r, 0, 0])

        # Net present value calculations
        # Discount rate
        denominator = (1+dr)**full_lt_mat
        
        # Only investment component of LCOE
        npv_investcomp_av = (it_av+st)/denominator
        npv_investcomp_mu = (it_mu+st)/denominator
        
        # Only debt cost component of LCOE
        npv_debt = dbt/denominator

        # 1-Expenses
        # 1.1-Without policy costs
        npv_expenses1 = (it_av+ft+omt+stor_cost+marg_stor_cost+gridcost)/denominator
        # 1.2-With policy costs
        # npv_expenses2 = (it+st+fft+ft+ct+omt+stor_cost+marg_stor_cost)/denominator
        npv_expenses2 = (it_av+fft+st+ft+ct+omt+stor_cost+marg_stor_cost+gridcost)/denominator
        # 1.3-Without policy, with co2p
        # TODO: marg_stor_cost?
        npv_expenses3 = (it_mu+ft+ct+omt+stor_cost+marg_stor_cost+gridcost)/denominator
        # 1.3-Only policy costs
        # npv_expenses3 = (ct+fft+st)/denominator
        # 2-Utility
        npv_utility = (et)/denominator
        
        npv_benefits = (data['DAEP'][r,0,0] * data['DPVF'][r, :, 0])/denominator
        #Remove 1s for tech with small lifetime than max
        npv_utility[npv_utility==1] = 0
        # npv_utility[:,0] = 1
        # 3-Standard deviation (propagation of error)
        npv_std = np.sqrt(dit_mu**2 + dft**2 + domt**2)/denominator
        

        # 1-levelised cost variants in $/pkm
        # 1.1-Bare LCOE
        lcoe = np.sum(npv_expenses1, axis=1)/np.sum(npv_utility, axis=1)
        # 1.2-LCOE including policy costs
        tlcoe = np.sum(npv_expenses2, axis=1)/np.sum(npv_utility, axis=1) - data['MEFI'][r, :, 0]
        # 1.3 LCOE excluding policy, including co2 price
        lcoeco2 = np.sum(npv_expenses3, axis=1)/np.sum(npv_utility, axis=1)
        # 1.3-LCOE of policy costs
        # lcoe_pol = np.sum(npv_expenses3, axis=1)/np.sum(npv_utility, axis=1)+data['MEFI'][r, :, 0]
        # Standard deviation of LCOE
        dlcoe = np.sum(npv_std, axis=1)/np.sum(npv_utility, axis=1)
        
        # Store investment component of LCOE (includes subsidies)
        data['DCPU'][r, :, 0] = np.sum(npv_investcomp_mu, axis=1)/np.sum(npv_utility, axis=1)
        data['DPCI'][r, :, 0] = divide(data['DCPU'][r, :, 0], tlcoe)
        
        # Store debt cost compoentn of LCOE (includes all policies) - $/MWh
        data['DPDC'][r, :, 0] = np.sum(npv_debt, axis=1)/np.sum(npv_utility, axis=1)

        # LCOE augmented with gamma values
        tlcoeg = tlcoe+data['MGAM'][r, :, 0]

        # Pass to variables that are stored outside.
        data['MEWC'][r, :, 0] = dc(lcoe)     # The real bare LCOE without taxes
        data['MECW'][r, :, 0] = dc(lcoeco2)  # The real bare LCOE with taxes
        data['METC'][r, :, 0] = dc(tlcoeg)   # As seen by consumer (generalised cost)
        data['MTCD'][r, :, 0] = dc(dlcoe)    # Variation on the LCOE distribution

        # data['METC'][r, :, 0] = copy.deepcopy(data['METCX'][r, :, 0])   # As seen by consumer (generalised cost)
        # data['MTCD'][r, :, 0] = copy.deepcopy(data['MTCDX'][r, :, 0])    # Variation on the LCOE distribution


        # Output variables
        data['MWIC'][r, :, 0] = dc(bcet[:, 2])
        data['MWFC'][r, :, 0] = dc(bcet[:, 4])
        data['MCOC'][r, :, 0] = dc(bcet[:, 0])
    

        if np.rint(data['MSAL'][r, 0, 0]) > 1:
            data['MWMC'][r, :, 0] = bcet[:, 0] + bcet[:, 4] + bcet[:, 6] + (data['MSSP'][r, :, 0] + data['MLSP'][r, :, 0])/1000
        else:
            data['MWMC'][r, :, 0] = bcet[:, 0] + bcet[:, 4] + bcet[:, 6]

        # TODO: Temporarily replace fuel costs with MWFCX
        # data['MWMC'][r, :, 0] = bcet[:, 0] + data['MWFCX'][r, :, 0] + bcet[:, 6]

        data['MMCD'][r, :, 0] = np.sqrt(bcet[:, 1]*bcet[:, 1] +
                                        bcet[:, 5]*bcet[:, 5] +
                                        bcet[:, 7]*bcet[:, 7])
        
        # Calculate leverage for use elsewhere
        data['DLEV'][r, :, 0] = 1+(data['DPDR'][r, :, 0]/(1 - data['DPDR'][r, :, 0]))
        
        
        # Revenue = electricity price * value factor
        data['DRPU'][r, :, 0] = data['DAEP'][r,0,0]*data['DPVF'][r,:,0]
        
        data['DRPU2'][r, :, 0] = dc(npv_benefits)
        
        # Profit per unit (includes gamma value calibration)
        data['DPPU'][r, :, 0] = data['DRPU'][r, :, 0] - data['METC'][r, :, 0]
        
        # Proft per unit as cost benefit
        data['DPPU2'][r, :, 0] = npv_benefits - npv_expenses2
        
        # Profit rate
        data['DPPR'][r, :, 0] = divide(data['DPPU'][r, :, 0] , 
                                       data['DCPU'][r, :, 0]*(1-data['DPDR'][r, :, 0]))
        
        data['DPPR2'][r, :, 0] = divide(data['DPPU2'][r, :, 0] , 
                                       data['DPPU2'][r, :, 0]*(1-data['DPDR'][r, :, 0]))
        
        # Risk adjustment factor
        data['DRAF'][r, :, 0] = np.where((1 - data['DCOC'][r, :, 0] * data['DLEV'][r, :, 0]) > 0.0,
                                         1 / (1 - data['DCOC'][r, :, 0] * data['DLEV'][r, :, 0]),
                                         1.0)
        
        # Risk adjusted profit rate
        data['DRPR'][r, :, 0] = data['DPPR'][r, :, 0] * data['DRAF'][r, :, 0]
        
        data['DRPR2'][r, :, 0] = data['DPPR2'][r, :, 0] * data['DRAF'][r, :, 0]
        
        # Rought approximation of the stand. dev. of the profit rate
        data['DPRD'][r, :, 0] = data['DPPR'][r, :, 0] * divide(data['MTCD'][r, :, 0], data['MEWC'][r, :, 0])
        
        data['DPRD2'][r, :, 0] = data['DRPR2'][r, :, 0] * divide(data['MTCD'][r, :, 0], data['MEWC'][r, :, 0])

    return data
