# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:54:30 2024

@author: AE

=========================================
ftt_h_lcoh.py
=========================================
Domestic Heat FTT module.
####################################

This is the main file for FTT: Heat, which models technological
diffusion of domestic heat technologies due to consumer decision making.
Consumers compare the **levelised cost of heat**, which leads to changes in the
market shares of different technologies.

The outputs of this module include levelised cost of heat technologies

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Bespoke element-wise divide which replaces divide-by-zeros with zeros
    - `estimation <econometrics_functions.html>`__
        Predict future values according to the estimated coefficients.

Functions included:
    - get_lcoh
        Calculate levelised cost of transport

variables:
cf = capacity factor
ce = conversion efficiency


"""

# Third party imports
import numpy as np

# Local library imports
from SourceCode.support.divide import divide

# %% LOCH
# --------------------------------------------------------------------------
# -------------------------- LCOH function ---------------------------------
# --------------------------------------------------------------------------

def get_lcoh(data, titles):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of heat in 2014 Euros/kWh per
    boiler type. It includes intangible costs (gamma values) and together
    determines the investor preferences.
    """
    # Categories for the cost matrix (BCHY)
    c7ti = {category: index for index, category in enumerate(titles['C7TI'])}

    for r in range(len(titles['RTI'])):

        # Cost matrix
        #BCHY = data['BCHY'][r, :, :]

        # Boiler lifetime
        lt = data['BCHY'][r,:, c7ti['Lifetime']]
        bt = data['BCHY'][r,:, c7ti['Buildtime']]
        max_lt = int(np.max(bt+lt))

        # Define (matrix) masks to turn off cost components before or after contruction
        full_lt_mat = np.linspace(np.zeros(len(titles['HYTI'])), max_lt-1,
                                  num=max_lt, axis=1, endpoint=True)
        lt_max_mat = np.concatenate(int(max_lt) * [(lt+bt-1)[:, np.newaxis]], axis=1)
        bt_max_mat = np.concatenate(int(max_lt) * [(bt-1)[:, np.newaxis]], axis=1)

        bt_mask = full_lt_mat <= bt_max_mat
        bt_mask_out = full_lt_mat > bt_max_mat
        lt_mask_in = full_lt_mat <= lt_max_mat
        lt_mask = np.where(lt_mask_in == bt_mask_out, True, False)

        # Capacity factor
        cf = data['BCHY'][r,:, c7ti['Capacity factor'], np.newaxis]
        conv = cf*bt[:, None]
        conv[-1, 0] = 1.0

        # Discount rate
        dr = data['BCHY'][r,:, c7ti['Discount rate'], np.newaxis]

        # Initialse the levelised cost components
        # Average investment cost
        it = np.ones([len(titles['HYTI']), int(max_lt)])
        it = it * (
            data['BCHY'][r,:, c7ti['CAPEX, mean, €/tH2 cap'],np.newaxis]/(conv) +
            data['BCHY'][r,:, c7ti['Storage CAPEX, mean, €/tH2 cap'],np.newaxis]/(conv) +
            data['BCHY'][r,:, c7ti['Onsite electricity CAPEX, mean, €/tH2 cap'],np.newaxis]/(conv))
        it = np.where(bt_mask, it, 0)

        # Standard deviation of investment cost
        dit = it * data['BCHY'][r,:, c7ti['CAPEX, std, % of mean'],np.newaxis]

        # Upfront subsidy/tax at purchase time
        st = np.zeros_like(it)

        # Average fuel costs
        ft = np.zeros([len(titles['HYTI']), int(max_lt)])
        ft = ft + (
            data['BCHY'][r,:, c7ti["Feedstock input, mean, kWh/kg H2 prod."], np.newaxis]*0.05+
            data['BCHY'][r,:, c7ti["Heat demand, mean, kWh/kg H2"], np.newaxis]*0.05+
            data['BCHY'][r,:, c7ti["Electricity demand, mean, kWh/kg H2"], np.newaxis]*0.15)
        ft = np.where(lt_mask, ft, 0)

        # Standard deviation of fuel costs
        dft = np.zeros([len(titles['HYTI']), int(max_lt)])
        dft = dft + (
            data['BCHY'][r,:, c7ti["Feedstock input, std, % of mean"], np.newaxis]*0.05+
            data['BCHY'][r,:, c7ti["Heat demand, std, % of mean"], np.newaxis]*0.05+
            data['BCHY'][r,:, c7ti["Electricity demand, % of mean"], np.newaxis]*0.15)
        dft = np.where(lt_mask, dft, 0)

        # Fixed OPEX
        opex_fix = np.zeros([len(titles['HYTI']), int(max_lt)])
        opex_fix = opex_fix + (
            data['BCHY'][r,:, c7ti['Fixed OPEX, mean, €/kg H2 cap/y'],np.newaxis]/(cf))
        opex_fix = np.where(lt_mask, opex_fix, 0)

        # st.dev fixed opex
        dopex_fix = opex_fix * data['BCHY'][r,:, c7ti['Fixed OPEX, std, % of mean'],np.newaxis]

        # Variable OPEX
        opex_var = np.zeros([len(titles['HYTI']), int(max_lt)])
        opex_var = opex_var + (
            data['BCHY'][r,:, c7ti['Variable OPEX, mean, €/kg H2 prod'],np.newaxis])
        opex_var = np.where(lt_mask, opex_var, 0)

        # st.dev variable opex
        dopex_var= opex_var * data['BCHY'][r,:, c7ti['Variable OPEX, std, % of mean'],np.newaxis]

        # Energy production
        energy_prod = np.ones([len(titles['HYTI']), int(max_lt)])
        energy_prod = np.where(lt_mask, energy_prod, 0)

        # Net present value calculations
        # Discount rate
        denominator = (1+dr)**full_lt_mat

        # 1-Expenses
        # 1.1-NPV
        npv_expenses1 = (it+st+ft+opex_fix+opex_var)/denominator

        # 2-Utility
        npv_utility = energy_prod/denominator
        npv_utility[npv_utility==1] = 0

        # 3-Standard deviation (propagation of error)
        npv_std = np.sqrt(dit**2 + dft**2 + dopex_fix**2 + + dopex_var**2)/denominator

        # 1-levelised cost variants in $/pkm
        # 1.1-Bare LCOH
        lcoh = np.sum(npv_expenses1, axis=1)/np.sum(npv_utility, axis=1)
        lcoh[np.isnan(lcoh)] == 0.0

        # Standard deviation of LCOH
        dlcoh = np.sum(npv_std, axis=1)/np.sum(npv_utility, axis=1)
        dlcoh[np.isnan(dlcoh)] == 0.0

        # Pass to variables that are stored outside.
        data['HYLC'][r, :, 0] = lcoh            # The real bare LCOH without taxes
        data['HYLD'][r, :, 0] = dlcoh          # Variation on the LCOH distribution

    return data