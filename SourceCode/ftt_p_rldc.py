# -*- coding: utf-8 -*-
"""
power_generation.py
=========================================
Power generation FTT module.

Functions included:
    - get_lcoe
        Calculate levelized costs
    - solve
        Main solution function for the module
"""

# Standard library imports
from math import sqrt
import os
import copy
import sys
import warnings

# Third party imports
import pandas as pd
import numpy as np

# Local library imports
from support.divide import divide

# %% rldc function
# -----------------------------------------------------------------------------
# -------------------------- RLDC calcultion ------------------------------
# -----------------------------------------------------------------------------
def rldc(data, time_lag, year, titles):
    """
    Calculate RLDCs.

    The function calculates the RLDCs and returns load band heights, curtailment,
    and storage information, including storage costs and marginal costs
    for wind and solar.

    Parameters
    -----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
    time_lag: dictionary
        Time_lag is a container that holds all cross-sectional (of time) data
        for all variables of the previous year.
        Variable names are keys and the values are 3D NumPy arrays.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.

    Returns
    ----------
    data: type
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
        The values inside the container are updated and returned to the main
        routine.

    Notes
    ---------
    Additional notes if required.
    """

    # Mapping of NWR = 53 world regions to 8 available RLDC regions:
    # 1 = Europe, 2 = Latin America, 3 = India, 4 = USA, 5 = Japan, 6 = Middle
    # East and North Africa, 7 = Sub-Saharan Africa, 8 = China
    rldc_regmap = np.zeros(len(titles['RTI']), dtype=int)
    rldc_regmap[0:33] = 0 # Europe
    rldc_regmap[33] = 3 # USA
    rldc_regmap[34] = 4 # Japan
    rldc_regmap[35:38] = 3 # Canada, Australia, New Zealand (USA as proxy)
    rldc_regmap[38:40] = 0  # Russia, Rest of Annex I (Europe as proxy)
    rldc_regmap[40] = 7  # China
    rldc_regmap[41] = 2  # India
    rldc_regmap[42:47] = 1  # Mexico, Brazil, Argentina, Colombia, Rest of LAM
    rldc_regmap[47:49] = 4  # Korea, Taiwan (Japan as proxy)
    rldc_regmap[49:51] = 2  # Indonesia, Rest of ASEAN (India as proxy)
    rldc_regmap[51] = 5  # OPEC excluding Venezuela (MENA as proxy)
    rldc_regmap[52] = 6  # Rest of the world (Sub-Saharan Africa as proxy)
    rldc_regmap[53] = 0  # Ukraine (Europe as proxy)
    rldc_regmap[54] = 5  # Saudi (MENA as proxy)
    rldc_regmap[55:57] = 6  # Nigeria, South Africa, Rest Africa (Africa as proxy)
    rldc_regmap[57:59] = 5  # Africa OPEC (MENA as proxy)
    rldc_regmap[59:61] = 2  # Malaysia,Kazakhstan (India as proxy)
    rldc_regmap[61:69] = 6  # Rest of African regions (Africa as proxy)
    rldc_regmap[69] = 5  # UAE (MENA as proxy)
    rldc_regmap[70] = 5  # Placeholder (MENA as proxy)

    # Define matrices with polynomial coefficients for 8 RLDC regions
    # 10 input parameters (shares of generation of wind and solar in a
    # polynomial 1 + Sw + Ss + Sw^2 + Sw*Ss + Ss^2 + Sw^3 + Sw^2*Ss + Sw*Ss^2 + Ss^3)
    # 8 output results (Curtailment, storage capacity, storage costs, 5 load
    # band heights in order H4, H3, H2, H1, Hp)
    rldc_coeff = np.empty((8,10,8))

    # Europe (RLDCreg = 0)
    rldc_coeff[0] = np.array([[0.000, 0.000, 0.000, 1.301, 1.175, 1.058, 0.871, 1.386],
                             [0.048, 0.000, 0.000,-1.066,-1.189,-1.013,-1.138,-0.588],
                             [0.017, 0.000, 0.000,-0.467,-0.806,-0.756,-1.729,-0.483],
                             [-0.220, 0.039, 0.038, 0.602, 0.783, 0.124, 0.064, 0.013],
                             [-0.191, 0.513,-0.008,-0.585, 0.402,-0.588, 1.359,-0.662],
                             [-0.046, 1.435, 1.157,-0.171, 1.013, 0.004, 1.135,-0.397],
                             [0.336,-0.020,-0.018,-0.172,-0.302, 0.024, 0.151, 0.079],
                             [0.556, 0.000, 0.163,-0.223,-0.993, 0.341,-0.281, 0.000],
                             [0.191,-0.197, 0.731, 0.346,-0.657, 0.108,-0.476, 0.255],
                             [0.309,-0.736,-0.593, 0.158,-0.578, 0.112,-0.244, 0.299]])

    # Latin America (RLDCreg = 1)
    rldc_coeff[1] = np.array([[0.000, 0.000, 0.000, 1.224, 1.160, 1.080, 0.875, 1.312],
                             [0.005, 0.000, 0.000,-0.707,-0.962,-1.014,-1.308,-0.627],
                             [0.002, 0.000, 0.000,-0.142,-0.219,-0.712,-1.893,-0.377],
                             [-0.064, 0.106, 0.026, 0.094, 0.530, 0.260, 0.367, 0.286],
                             [-0.059, 0.642, 0.599,-1.118,-0.615,-0.822, 1.703,-0.678],
                             [0.112, 1.293, 0.743,-0.615,-0.316, 0.106, 1.441,-0.593],
                             [0.247, 0.003, 0.059, 0.018,-0.257,-0.096, 0.041,-0.133],
                             [0.393,-0.379,-0.283, 0.252,-0.261, 0.346,-0.418, 0.293],
                             [0.159, 0.109, 0.403, 0.576, 0.438, 0.429,-0.638, 0.265],
                             [0.062,-0.366, 0.143, 0.162,-0.023,-0.140,-0.367, 0.278]])

    # India (RLDCreg = 2)
    rldc_coeff[2] = np.array([[0.000, 0.000, 0.000, 1.111, 1.060, 1.020, 0.960, 1.182],
                             [0.002, 0.059, 0.016,-0.614,-0.685,-0.872,-1.749,-0.517],
                             [0.002, 0.000, 0.000,-0.064,-0.085,-0.382,-2.195,-0.127],
                             [0.190,-0.056,-0.011, 0.616, 0.665, 0.705, 1.020, 0.484],
                             [-0.052, 0.368, 0.493,-0.808,-1.068,-1.165, 2.389,-0.296],
                             [0.052, 1.779, 1.008,-0.574,-0.323,-0.117, 1.791,-0.833],
                             [0.150, 0.118, 0.002,-0.292,-0.357,-0.373,-0.205,-0.195],
                             [0.306, 0.024,-0.085, 0.016,-0.045,-0.020,-0.636,-0.140],
                             [0.301,-0.311,-0.209, 0.611, 0.836, 0.836,-1.001, 0.348],
                             [0.161,-0.807,-0.051, 0.102,-0.086,-0.116,-0.487, 0.343]])

    # USA (RLDCreg = 3)
    rldc_coeff[3] = np.array([[0.000, 0.000, 0.000, 1.381, 1.176, 1.029, 0.872, 1.544],
                             [0.018, 0.001, 0.000,-0.838,-0.949,-0.957,-1.280,-0.687],
                             [0.006, 0.000, 0.000,-1.558,-0.881,-0.555,-1.601,-1.934],
                             [-0.119, 0.029, 0.036, 0.492, 0.420, 0.100, 0.238, 0.330],
                             [-0.112, 0.490, 0.235,-1.032,-0.190,-0.644, 1.588,-0.822],
                             [0.052, 1.314, 0.819, 1.853, 0.924,-0.219, 0.915, 2.325],
                             [0.263, 0.026,-0.012,-0.257,-0.206,-0.001, 0.109,-0.186],
                             [0.532,-0.314,-0.181, 0.477,-0.084, 0.358,-0.425, 0.317],
                             [0.175, 0.128, 0.506, 0.351,-0.287, 0.195,-0.527, 0.454],
                             [0.153,-0.674,-0.233,-0.963,-0.574, 0.138,-0.159,-1.148]])

    # Japan (RLDCreg = 4)
    rldc_coeff[4] = np.array([[0.000, 0.000, 0.000, 1.275, 1.147, 1.045, 0.891, 1.429],
                             [0.001, 0.000, 0.000,-0.802,-0.985,-1.030,-1.462,-0.514],
                             [0.000, 0.226, 0.000,-0.719,-0.419,-0.565,-1.813,-1.172],
                             [-0.036, 0.067, 0.037, 0.592, 0.856, 0.633, 0.607, 0.337],
                             [-0.007, 0.800, 0.853,-1.065,-0.622,-0.797, 1.854,-1.125],
                             [0.280, 0.971, 1.225, 0.369, 0.143, 0.040, 1.309, 0.901],
                             [0.330,-0.034,-0.014,-0.236,-0.397,-0.311,-0.051,-0.121],
                             [0.135, 0.000, 0.000, 0.395,-0.163, 0.031,-0.474, 0.093],
                             [0.009,-0.308,-0.328, 0.302, 0.053, 0.259,-0.670, 0.471],
                             [0.041,-0.542,-0.628,-0.107,-0.093, 0.033,-0.315,-0.287]])

    # Middle East and North Africa (RLDCreg = 5)
    rldc_coeff[5] = np.array([[0.000, 0.000, 0.000, 1.217, 1.154, 1.073, 0.885, 1.283],
                             [0.050, 0.084, 0.010,-0.997,-1.058,-1.045,-1.133,-0.795],
                             [0.050, 0.000, 0.000,-0.136,-0.387,-1.202,-1.779,-0.312],
                             [-0.185,-0.125, 0.003, 0.362, 0.252, 0.113, 0.066, 0.231],
                             [-0.351, 0.409, 0.339,-0.530,-0.661,-0.293, 1.542,-0.436],
                             [-0.206, 1.571, 0.930,-0.807,-0.276, 1.335, 1.189,-0.743],
                             [0.229, 0.062, 0.009,-0.109,-0.055,-0.004, 0.136,-0.039],
                             [0.650,-0.133,-0.185,-0.296,-0.118, 0.304,-0.294, 0.000],
                             [0.621,-0.091, 0.083, 0.448, 0.553, 0.073,-0.561, 0.168],
                             [0.367,-0.806,-0.227, 0.374, 0.058,-0.799,-0.260, 0.442]])

    # Sub-Saharan Africa (RLDCreg = 6);
    rldc_coeff[6] = np.array([[0.000, 0.000, 0.000, 1.165, 1.093, 1.043, 0.929, 1.225],
                             [0.050, 0.023, 0.000,-0.977,-0.979,-1.065,-1.206,-0.977],
                             [0.044, 0.000, 0.000, 0.000,-0.137,-0.703,-2.062,-0.084],
                             [-0.196,-0.031, 0.038, 0.409, 0.284, 0.265, 0.100, 0.742],
                             [-0.344, 0.547, 0.384,-0.714,-0.814,-0.592, 1.817,-0.426],
                             [-0.141, 1.619, 0.803,-0.827,-0.426, 0.344, 1.557,-0.985],
                             [0.258, 0.014,-0.018,-0.106,-0.077,-0.096, 0.143,-0.273],
                             [0.681,-0.019,-0.184,-0.058,-0.084, 0.206,-0.399,-0.294],
                             [0.598,-0.392, 0.239, 0.586, 0.713, 0.555,-0.688, 0.478],
                             [0.266,-0.579, 0.172, 0.236, 0.007,-0.336,-0.392, 0.410]])

    # China (RLDCreg = 7);
    rldc_coeff[7] = np.array([[0.000, 0.000, 0.000, 1.176, 1.131, 1.037, 0.908, 1.201],
                             [0.008, 0.109, 0.000,-0.778,-0.921,-0.855,-1.039,-0.447],
                             [0.004, 0.000, 0.000,-0.470,-0.658,-0.629,-1.881,-0.550],
                             [-0.073,-0.060, 0.067, 0.205, 0.313,-0.045,-0.157, 0.055],
                             [-0.087, 0.588, 0.725,-0.674,-0.052,-0.757, 1.434,-0.875],
                             [0.073, 1.571, 1.093, 0.019, 0.680, 0.239, 1.282, 0.076],
                             [0.211, 0.009,-0.034,-0.034,-0.133, 0.054, 0.233,-0.007],
                             [0.426, 0.000,-0.177,-0.107,-0.412, 0.259,-0.273, 0.189],
                             [0.252,-0.226,-0.066, 0.471,-0.200, 0.349,-0.489, 0.339],
                             [0.191,-0.806,-0.434,-0.023,-0.491,-0.174,-0.291, 0.005]])

    # Backup factor as reserve margin (% of peak load)
    backup = 1.2

    # Wind and solar shares for all regions
    Sw = np.zeros(len(titles['RTI']))
    Ss = np.zeros(len(titles['RTI']))
    Sw = np.divide(np.sum(data['MEWG'][:, [16, 17, 21], 0], axis=1),
                   np.sum(data['MEWG'][:, :, 0], axis=1),
                   where=(np.sum(data['MEWG'][:, :, 0], axis=1) != 0))
    Ss = np.divide(data['MEWG'][:, 18, 0],
                   np.sum(data['MEWG'][:, :, 0], axis=1),
                   where=(np.sum(data['MEWG'][:, :, 0], axis=1) != 0))

    # Format matrix including Sw and Ss powers and cross terms
    # template: [1 Sw Ss Sw^2 Sw*Ss Ss^2 Sw^3 Sw^2*Ss Sw*Ss^2 Ss^3]
    vre_powers = np.array([np.ones(len(Sw)), Sw, Ss, Sw**2, np.multiply(Sw, Ss), Ss**2, Sw**3, np.multiply(Sw**2, Ss), np.multiply(Sw, Ss**2), Ss**3])
    vre_powers = vre_powers.transpose()

    # What is the impact of additional wind or solar?
    # vre_gr_sol = np.ones(len(titles['RTI']))*0.005
    # vre_gr_wind = np.ones(len(titles['RTI']))*0.005
    # ind_nonzero_sol = np.where(time_lag["MEWS"][:, 18, 0]*time_lag["MEWL"][:, 18, 0] > 0.01)
    # ind_nonzero_wind =  np.where(np.sum(time_lag["MEWS"][:, 16:18, 0]*time_lag["MEWL"][:, 16:18, 0], axis=1) > 0.01)
    # vre_gr_sol[ind_nonzero_sol] = np.maximum(data["MEWS"][:, 18, 0]*data["MEWL"][:, 18, 0]/(time_lag["MEWS"][:, 18, 0]*time_lag["MEWL"][:, 18, 0] )*data["MEWS"][:, 18, 0], 0.005)                           
    
    # with np.errstate(invalid='ignore'):
    #     vre_gr_sol = np.where(time_lag["MEWS"][:, 18, 0]*time_lag["MEWL"][:, 18, 0] > 0.01, 
    #                           np.maximum(np.divide(data["MEWS"][:, 18, 0]*data["MEWL"][:, 18, 0],
    #                                    time_lag["MEWS"][:, 18, 0]*time_lag["MEWL"][:, 18, 0])
    #                           * data["MEWS"][:, 18, 0], 0.005),
    #                           0.005)
    #     vre_gr_wind = np.where(np.sum(time_lag["MEWS"][:, 16:18, 0]*time_lag["MEWL"][:, 16:18, 0], axis=1) > 0.01, 
    #                           np.maximum(np.sum(data["MEWS"][:, 16:18, 0]*data["MEWL"][:, 16:18, 0], axis=1)/np.sum(time_lag["MEWS"][:, 16:18, 0]*time_lag["MEWL"][:, 16:18, 0], axis=1)*np.sum(data["MEWS"][:, 16:18, 0], axis=1)
    #                                   , 0.005), 0.005)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        vre_gr_sol  = (data["MEWS"][:, 18, 0]*data["MEWL"][:, 18, 0]/(time_lag["MEWS"][:, 18, 0]*time_lag["MEWL"][:, 18, 0] )-1)*data["MEWS"][:, 18, 0]
        vre_gr_sol[np.isnan(vre_gr_sol) | np.isinf(vre_gr_sol)] = 0.0
        vre_gr_sol =  np.maximum(vre_gr_sol, 0.005)
        vre_gr_wind = (np.sum(data["MEWS"][:, 16:18, 0]*data["MEWL"][:, 16:18, 0], axis=1)/np.sum(time_lag["MEWS"][:, 16:18, 0]*time_lag["MEWL"][:, 16:18, 0], axis=1)-1)*np.sum(data["MEWS"][:, 16:18, 0], axis=1)
        vre_gr_wind[np.isnan(vre_gr_wind) | np.isinf(vre_gr_wind)] = 0.0
        vre_gr_wind = np.maximum(vre_gr_wind, 0.005)
                               
   
    Sw2, Ss2 = np.add(Sw, vre_gr_wind), np.add(Ss, vre_gr_sol)
    ones = np.ones(len(Sw2))
    zeros = np.zeros(len(Sw2))
    vre_powers_wind = np.array([np.ones(len(Sw2)), Sw2, Ss, Sw2**2, np.multiply(Sw2, Ss), Ss**2, Sw2**3, np.multiply(Sw2**2, Ss), np.multiply(Sw2, Ss**2), Ss**3])
    vre_powers_wind = vre_powers_wind.transpose()
    vre_powers_solar = np.array([np.ones(len(Sw)), Sw, Ss2, Sw**2, np.multiply(Sw, Ss2), Ss2**2, Sw**3, np.multiply(Sw**2, Ss2), np.multiply(Sw, Ss2**2), Ss2**3])
    vre_powers_solar = vre_powers_solar.transpose()
    # Split it to compute distribute curtailment over solar/wind depending on imbalance
    vre_powers_split_sol = np.array([np.ones(len(Sw)), zeros, Ss, zeros, zeros, Ss**2, zeros, zeros, zeros, Ss**3]) 
    vre_powers_split_sol = vre_powers_split_sol.transpose()
    vre_powers_split_wind = np.array([np.ones(len(Sw)), Sw, zeros, Sw**2, zeros, zeros, Sw**3, zeros, zeros, zeros])
    vre_powers_split_wind = vre_powers_split_wind.transpose()
    
    # Multidimensional polynomial from Ueckerdt et al. (2017)
    # Gives [Curt, Ustor, CostStor, H4, H3, H3, H1, Hp]
    rldc_prod = np.array([np.dot(vre_powers[r], rldc_coeff[rldc_regmap[r]]) for r in range(len(titles['RTI']))])
    rldc_prod_wind = np.array([np.dot(vre_powers_wind[r], rldc_coeff[rldc_regmap[r]]) for r in range(len(titles['RTI']))])
    rldc_prod_solar = np.array([np.dot(vre_powers_solar[r], rldc_coeff[rldc_regmap[r]]) for r in range(len(titles['RTI']))])
    # Split curtailment
    rldc_prod_split_wind = np.array([np.dot(vre_powers_split_wind[r], rldc_coeff[rldc_regmap[r]]) for r in range(len(titles['RTI']))])
    rldc_prod_split_solar = np.array([np.dot(vre_powers_split_sol[r], rldc_coeff[rldc_regmap[r]]) for r in range(len(titles['RTI']))])
    
    feqs = lambda a : np.max([a, 0.001])
    ratio = np.array([np.sqrt(
                    feqs(rldc_prod_split_wind[r, 0])/feqs(rldc_prod_split_solar[r, 0]) *  \
                    feqs(rldc_prod_wind[r, 0])/feqs(rldc_prod_solar[r, 0])) \
                              for r in range(len(titles['RTI']))])
    # Curtailment
    data['MCRT'][:, 0, 0] = np.array([rldc_prod[r, 0] for r in range(len(titles['RTI']))])
    
    with np.errstate(invalid='ignore'):
        cw = np.where(Sw + Ss > 0, data["MCRT"][:, 0, 0] * (Sw + Ss) / (Sw + Ss/ratio), 0.0)
        cs = np.where(Sw + Ss > 0, data["MCRT"][:, 0, 0] * (Sw + Ss) / (Sw*ratio + Ss), 0.0)
    
    # Curb values over 75% curtailment
    data["MCRT"] = np.where(data["MCRT"] > 0.75, 0.75, data["MCRT"])
    cw = np.where(cw > 0.75, 0.75, cw)
    cs = np.where(cs > 0.75, 0.75, cs)
    
    data["MCTG"] = np.zeros_like(data["MEWG"])
    
    for t in [16, 17, 21]:
        data["MCTG"][:, t, 0] = cw
    data["MCTG"][:, 18, 0] = cs
 
    
    for r in range(len(titles['RTI'])):
        output_ratio = Sw[r]*0.008512894 + Ss[r]*0.068004505 + Sw[r]*Ss[r]*0.021214422
        total_output_l = output_ratio*np.sum(data['MEWG'][r, :, 0])
        data['MLSC'][r, 0, 0] = total_output_l/140*0.1
        if 0: # Switch to change implementation (1 for old, 0 for new)
            data['MSSC'][r, 0, 0] = rldc_prod[r, 1]*np.sum(data['MEWK'][r, :, 0])
        else:
            data['MSSC'][r, 0, 0] = rldc_prod[r, 1]*np.sum(data['MEWG'][r, :, 0])*0.175e-3
            
        
        
    costs_ss = 0.15 * 1e6
    costs_ls = 0.2  * 1e6
    learning_exp_ss = -(0.342 + 0.168)/2 # Average for flow and li-ion batteries, as mid-length storage may be included here.
    learning_exp_ls = -0.194 # Both from Way paper (https://www.inet.ox.ac.uk/files/energy_transition_paper-INET-working-paper.pdf), latter less reliable due to data constraints
    if year>2020:
        # TODO: make sure this is not hard-coded
        costs_ss = costs_ss * (np.sum(data['MSSC']) / 30.51)**learning_exp_ss
        # MSSC -> 2018: 21.44, 2019: 29.84, 2020: 40.72
        costs_ls = costs_ls * (np.sum(data['MLSC']) / 7.06)**learning_exp_ls
        # MLSC -> 2018: 36.29, 2019: 43.85, 2020: 57.99
            
    # Storage. Seperate loop to account for world-wide learning        
    
    
    for r in range(len(titles['RTI'])):
        
        # LONG-TERM STORAGE
        # Simple regression
        output_ratio = Sw[r]*0.008512894 + Ss[r]*0.068004505 + Sw[r]*Ss[r]*0.021214422
        output_ratio_wind = Sw2[r]*0.008512894 + Ss[r]*0.068004505 + Sw2[r]*Ss[r]*0.021214422
        output_ratio_solar = Sw[r]*0.008512894 + Ss2[r]*0.068004505 + Sw[r]*Ss2[r]*0.021214422
        # Additional demand to account for RTE loss (assume 50% RTE)
        total_output_l = output_ratio*np.sum(data['MEWG'][r, :, 0])
        total_input_l = total_output_l/0.5
        total_output_wind_l = output_ratio_wind*np.sum(data['MEWG'][r, :, 0])
        total_input_wind_l = total_output_wind_l/0.5
        total_output_solar_l = output_ratio_solar*np.sum(data['MEWG'][r, :, 0])
        total_input_solar_l = total_output_solar_l/0.5
        # NOTE: skipped add_gen_[wind/solar] because it is not used
        # Additional electricity (GWh) that must be generated due to RTE losses
        data['MLSG'][r, 0, 0] = total_input_l - total_output_l
        # Typically, 100 MW, 70 GWh/cycle. Assume 2 cycles, so 100 MW capacity for every 140 GWh discharched


        # SHORT-TERM STORAGE
        
        # rldc_prod = np.dot(vre_powers[r], rldc_coeff[rldc_regmap[r]])
        # rldc_prod_wind = np.dot(vre_powers_wind[r], rldc_coeff[rldc_regmap[r]])
        # rldc_prod_solar = np.dot(vre_powers_solar[r], rldc_coeff[rldc_regmap[r]])
        
 
        # Short-term storage capacity
        # Older version used total capacity as upper bound for peak load, but...
        # Improve by estimating from demand in GWh
        # Best guess right now is [peak load in GW] ~= 0.175*[annual demand in TWh]
        # This is based on data from UK, US, IN, CN, EU, BR, SA
        # Coefficient might go up (cooling, EVs, etc.) or down (demand response,
        # LEDs, etc.) but has been fairly constant over time (note this
        # is directly proportional to the peak-to-average load ratio)
        
        # Assume 26.2% ratio between battery capacity and storage output
        # 0.80 is roundtrip efficiency estimate  (https://www.pnnl.gov/sites/default/files/media/file/Final%20-%20ESGC%20Cost%20Performance%20Report%2012-11-2020.pdf, average Li-ion, Vanadium)
        total_input_s = rldc_prod[r, 1]*0.262*np.sum(data['MEWG'][r, :, 0])/0.80
        total_output_s = rldc_prod[r, 1]*0.262*np.sum(data['MEWG'][r, :, 0])
        total_input_wind_s = rldc_prod_wind[r, 1]*0.262*np.sum(data['MEWG'][r, :, 0])/0.80
        total_output_wind_s = rldc_prod_wind[r, 1]*0.262*np.sum(time_lag['MEWG'][r, :, 0])
        total_input_solar_s = rldc_prod_solar[r, 1]*0.262*np.sum(data['MEWG'][r, :, 0])/0.80
        total_output_solar_s = rldc_prod_solar[r, 1]*0.262*np.sum(time_lag['MEWG'][r, :, 0])
        # Additional electricity (GWh) that must be generated due to RTE losses
        data['MSSG'][r, 0, 0] = total_input_s - total_output_s
        if data['MSSG'][r, 0, 0] < 0.0:
            data['MSSG'][r, 0, 0] = 0.0
    
    
                
        # STORAGE COSTS
        # Assume fixed short-term storage levelised cost = 0.15 EURO/kWh
        # and fixed long-term storage levelised cost = 0.20 EURO/kWh
        # TODO: Convert to US$ here?
        vre = np.sum(data['MEWG'][r, [16, 17, 18, 21], 0])
        if np.rint(data['MSAL'][r, 0, 0]) == 3 and np.sum(data['MEWG'][r, [16, 17, 18, 21], 0]) > 0:

            # Short term storage cost
            data['MSSP'][r, [16, 17, 18, 21], 0] = total_output_s*costs_ss/vre
            # Long term storage cost
            data['MLSP'][r, [16, 17, 18, 21], 0] = total_output_l*costs_ls/vre

        else:
            data['MSSP'][r, :, 0] = np.divide(total_output_s*costs_ss, np.sum(data['MEWG'][r, :, 0]), where=(np.sum(data['MEWG'][r, :, 0]) != 0))
            data['MLSP'][r, :, 0] = np.divide(total_output_l*costs_ls, np.sum(data['MEWG'][r, :, 0]), where=(np.sum(data['MEWG'][r, :, 0]) != 0))
            #data['MSSP'][r, [16, 17, 18, 21], 0] = np.divide(total_output_s*costs_ss, np.sum(data['MEWG'][r, :, 0]), where=(np.sum(data['MEWG'][r, :, 0]) != 0))
            #data['MLSP'][r, [16, 17, 18, 21], 0] = np.divide(total_output_l*costs_ls, np.sum(data['MEWG'][r, :, 0]), where=(np.sum(data['MEWG'][r, :, 0]) != 0))

        # Marginal cost due to addition of wind/solar in 2015EURO/additional GWh

        # All techs contribute to storage; only marginal costs are allocated to VRE techs
        if np.rint(data['MSAL'][r, 0, 0]) == 2 and np.sum(data['MEWG'][r, :, 0]) > 0:

            # Short term storage cost
            data['MSSM'][r, [16, 17, 21], 0] = total_output_wind_s*costs_ss/np.sum(data['MEWG'][r, :, 0]) - data['MSSP'][r, 16, 0]
            data['MSSM'][r, 18, 0] = total_output_solar_s*costs_ss/np.sum(data['MEWG'][r, :, 0]) - data['MSSP'][r, 16, 0]

            # Long term storage cost
            data['MLSM'][r, [16, 17, 21], 0] = total_output_wind_l*costs_ls/np.sum(data['MEWG'][r, :, 0]) - data['MLSP'][r, 16, 0]
            data['MLSM'][r, 18, 0] = total_output_solar_l*costs_ls/np.sum(data['MEWG'][r, :, 0]) - data['MLSP'][r, 16, 0]

        # Only VRE contribute to storage; only marginal costs are allocated to VRE techs
        elif np.rint(data['MSAL'][r, 0, 0]) == 3 and vre > 0:

            # Short term storage cost
            data['MSSM'][r, [16, 17, 21], 0] = total_output_wind_s*costs_ss/(vre + vre_gr_wind[r]*np.sum(data['MEWG'][r, :, 0])) - data['MSSP'][r, [16, 17, 21], 0]
            data['MSSM'][r, 18, 0] = total_output_solar_s*costs_ss/(vre + vre_gr_sol[r]*np.sum(data['MEWG'][r, :, 0])) - data['MSSP'][r, 18, 0]

            # Long term storage cost
            data['MLSM'][r, [16, 17, 21], 0] = total_output_wind_l*costs_ls/(vre + vre_gr_wind[r]*np.sum(data['MEWG'][r, :, 0])) - data['MLSP'][r, [16, 17, 21], 0]
            data['MLSM'][r, 18, 0] = total_output_solar_l*costs_ls/(vre + vre_gr_sol[r]*np.sum(data['MEWG'][r, :, 0])) - data['MLSP'][r, 18, 0]

        else:

            # Short term storage cost
            data['MSSM'][r, [16, 17, 21], 0] = 0.0
            data['MSSM'][r, 18, 0] = 0.0

            # Long term storage cost
            data['MLSM'][r, [16, 17, 21], 0] = 0.0
            data['MLSM'][r, 18, 0] = 0.0

        if r==40:
            x = 1+1
#            print('BE')

        # LOAD BANDS
        # Heights of load bands
        h4 = copy.deepcopy(rldc_prod[r, 3])
        h3 = copy.deepcopy(rldc_prod[r, 4])
        h2 = copy.deepcopy(rldc_prod[r, 5])
        h1 = copy.deepcopy(rldc_prod[r, 6])
        h5 = copy.deepcopy(rldc_prod[r, 7])
        # VRE CF
        if np.sum(data['MEWS'][r, [16, 17, 18, 21], 0]) > 0:
            cf_var = np.sum(np.multiply(data['MEWS'][r, [16, 17, 18, 21], 0], data['MEWL'][r, [16, 17, 18, 21], 0]))/np.sum(data['MEWS'][r, [16, 17, 18, 21], 0])
        else:
            cf_var = 1
        # CF by load band
        cflb = np.empty(6)
        cflb[0] = 7500/8766
        cflb[1] = 4400/8766
        cflb[2] = 2200/8766
        cflb[3] = 700/8766
        cflb[4] = 80/8766
        cflb[5] = copy.deepcopy(cf_var)
        # Capacity by load band (normalised so that MGLB sums to 1, usually MKLB sums to > 1)
        data['MKLB'][r, 0, 0] = 8760/7500*max(h1, 0)
        data['MKLB'][r, 1, 0] = max(h2, 0) - max(h1, 0)
        data['MKLB'][r, 2, 0] = max(h3, 0) - max(h2, 0)
        data['MKLB'][r, 3, 0] = max(h4, 0)- max(h3, 0)
        data['MKLB'][r, 4, 0] = backup*max(h5, 0) - max(h4, 0)
        # Generation in VRE
        data['MGLB'][r, 5, 0] = Ss[r] + Sw[r]
        # Shares of VRE capacity
        if cf_var > 0.0001:
            data['MKLB'][r, 5, 0] = data['MGLB'][r, 5, 0]/cf_var
        else:
            data['MKLB'][r, 5, 0] = 0
        # Generation share by load band
        data['MGLB'][r, :, 0] = np.multiply(data['MKLB'][r, :, 0], cflb)
        # Normalise MKLB such that capacity of VRE load band = sum of VRE shares
        # and MKLB sums to 1
        data['MKLB'][r, 5, 0] = np.divide(data['MGLB'][r, 5, 0]*np.sum(np.multiply(data['MEWS'][r, :, 0], data['MEWL'][r, :, 0])), cflb[5], where=(cflb[5] != 0))
        if np.sum(data['MKLB'][r, range(5), 0]) > 0:
            data['MKLB'][r, range(5), 0] = data['MKLB'][r, range(5), 0]/np.sum(data['MKLB'][r, range(5), 0])*(1-data['MKLB'][r, 5, 0])

        if r==40:
            x = 1+1

    return data