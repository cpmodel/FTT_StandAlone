# -*- coding: utf-8 -*-
"""
============================================================
ftt_fr_main.py
============================================================
Freight transport FTT module.

This is the main file for FTT: Freight, which models technological
diffusion of freight vehicle types due to simulated consumer decision making.
Consumers compare the **levelised cost of freight**, which leads to changes 
in the market shares of different technologies.

The outputs of this module include market shares, fuel use, and emissions.

Local library imports:

    FTT: Freight functions:
    - `get_lcof <ftt_fr_lcof.html>`__
        Levelised cost calculation


Functions included:
    - solve
        Main solution function for the module


"""

# Standard library imports
from math import sqrt
import copy
import warnings

# Third party imports
import numpy as np

# Local library imports
from SourceCode.Freight.ftt_fr_lcof import get_lcof, set_carbon_tax
from SourceCode.support.divide import divide
from SourceCode.Freight.ftt_fr_sales import get_sales
from SourceCode.Freight.ftt_fr_mandate import EV_truck_mandate
from SourceCode.sector_coupling.battery_lbd import battery_costs


# %% main function
# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, iter_lag, titles, histend, year, domain):
    """
    Main solution function for the module.

    This function simulates investor decision making in the freight sector.
    Levelised costs (from the get_lcof function) are taken and market shares
    for each vehicle type are simulated to ensure demand is met.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution
    time_lag: type
        Model variables from the previous year
    iter_lag: type
        Model variables from the previous year
    titles: dictionary of lists
        Dictionary containing all title classification
    histend: dict of integers
        Final year of histrorical data by variable
    year: int
        Curernt/active year of solution

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution

    """

    # Categories for the cost matrix
    c6ti = {category: index for index, category in enumerate(titles['C6TI'])}
    
    sector = 'freight'

    # Factor used to create intermediate data from annual figures
    no_it = int(data['noit'][0, 0, 0])
    dt = 1 / float(no_it)

    # Creating variables
    # Technology to fuel user conversion matrix
    zjet = copy.deepcopy(data['ZJET'][0, :, :])
    # Initialise the emission correction factor
    emis_corr = np.ones([len(titles['RTI']), len(titles['FTTI'])])

    
    if year == 2012:
        start_nonbat_cost = np.zeros([len(titles['RTI']), len(titles['FTTI']), 1])
        for veh in range(len(titles['FTTI'])):
            if veh in [12, 13, 18, 19]:
                # Starting EV cost (without battery)
                start_nonbat_cost[:, veh, 0] = (data['ZCET'][:, veh, c6ti['1 Price of vehicles (USD/vehicle)']]
                                          - time_lag['ZCET'][:, veh, c6ti['21 Battery capacity (kWh)']]
                                          * data["ZCET"][:, veh, c6ti['22 Battery cost ($/kWh)']]  )
            else:
                start_nonbat_cost[:, veh, 0] = 0
        # Save the nonbat_costs for later
        data["ZEVC"] = start_nonbat_cost
    elif year > 2020:
        # Copy over TEVC values
        data['ZEVC'] = np.copy(time_lag['ZEVC'] )
        pass



    # Initialise up to the last year of historical data
    if year <= histend["RVKZ"]:
        
        for r in range(len(titles['RTI'])):
            # Correction to market shares
            # Sometimes historical market shares do not add up to 1.0
            if (~np.isclose(np.sum(data['ZEWS'][r, :, 0]), 1.0, atol=1e-9)
                    and np.sum(data['ZEWS'][r, :, 0]) > 0.0):
                        data['ZEWS'][r, :, 0] = np.divide(data['ZEWS'][r, :, 0],
                                                np.sum(data['ZEWS'][r, :, 0]))
            
            # Find total service area
            data['ZESG'][r, :, 0] = data["RVKZ"][r, 0, 0] / data['ZLOD'][r, 0, 0]

            # ZESD is share difference between small and large trucks
            data['ZESD'][r, 0, 0] = data['ZEWS'][r, 0, 0] + data['ZEWS'][r, 2, 0] + data['ZEWS'][r, 4, 0] \
            + data['ZEWS'][r, 6, 0] + data['ZEWS'][r, 8, 0] + data['ZEWS'][r, 10, 0] + data['ZEWS'][r, 12, 0] \
            + data['ZEWS'][r, 14, 0] + data['ZEWS'][r, 16, 0] + data['ZEWS'][r, 18, 0]

            data['ZESD'][r, 1, 0] = 1 - data['ZESD'][r, 0, 0]

            # Find service area

            if data['ZESD'][r, 0, 0] > 0:
                for x in range(0, 20, 2):  
                    data['ZESA'][r, x, 0] = data['ZEWS'][r, x, 0] / data['ZESD'][r, 0, 0]
                    data['ZEVV'][r, x, 0] = data['ZESG'][r, x, 0] * data['ZESA'][r, x, 0] \
                                            / (1 - 1 / (data['ZSLR'][r, 0, 0] + 1))
            if data['ZESD'][r, 1, 0] > 0:
                for x in range(1, 21, 2):
                    data['ZESA'][r, x, 0] = data['ZEWS'][r, x, 0] / data['ZESD'][r, 1, 0]
                    data['ZEVV'][r, x, 0] = data['ZESG'][r, x, 0] * data['ZESA'][r, x, 0] \
                                            / (1 / (data['ZSLR'][r, 0, 0] + 1))
                
            for veh in range(len(titles['FTTI'])):
                for fuel in range(len(titles['JTI'])):
                    if titles['JTI'][fuel] == '11 Biofuels'  and data['ZJET'][0, veh, fuel] == 1:
                        # No biofuel blending mandate in the historical period
                        zjet[veh, fuel] = 0

            # Find fuel use
            data['ZJNJ'][r, :, 0] = (np.matmul(np.transpose(zjet), data['ZEVV'][r, :, 0] * \
                                        data['ZCET'][r, :, c6ti['9 energy use (MJ/vkm)']])) / 41.868

            # Emissions 
            data['ZEWE'][r, :, 0] = data['ZEVV'][r, :, 0] * data['ZCET'][r, :, c6ti['14 CO2Emissions (gCO2/km)']] \
                        * (1 - data['ZBFM'][r, 0, 0]) / (1e6)
        
        # Calculate number of vehicles per technology
        data['ZEWK'][:, :, 0] = data['ZEWS'][:, :, 0] \
                                * data['RFLZ'][:, np.newaxis, 0, 0]
        
        # Set cumulative capacities variable
        data['ZEWW'][0, :, 0] = data['ZCET'][0, :, c6ti['11 Cumulative seats']]

        if year == histend["RVKZ"]:
            # Calculate levelised cost
            carbon_costs = set_carbon_tax(data, c6ti)
            data = get_lcof(data, titles, carbon_costs, year)
            
            data["ZCET initial"] = np.copy(data["ZCET"])



    "Model Dynamics"

    # Endogenous calculation starts here
    if year > histend['RVKZ']:

        data_dt = {}
        data_dt['ZWIY'] = np.zeros([len(titles['RTI']), len(titles['FTTI']), 1])

        for var in time_lag.keys():

            if var.startswith("R"):

                data_dt[var] = copy.deepcopy(time_lag[var])

        for var in time_lag.keys():

            if var.startswith("Z"):

                data_dt[var] = copy.deepcopy(time_lag[var])

        # Find if there is a regulation and if it is exceeded
        division = divide((time_lag['ZEWK'][:, :, 0] - data['ZREG'][:, :, 0]),
                           data['ZREG'][:, :, 0]) # 0 when dividing by 0
        isReg = 0.5 + 0.5 * np.tanh(1.5 + 10 * division)
        isReg[data['ZREG'][:, :, 0] == 0.0] = 1.0
        isReg[data['ZREG'][:, :, 0] == -1.0] = 0.0

        
        data["ZWSA"] = EV_truck_mandate(data["EV truck mandate"], data["ZWSA"], time_lag["ZEWS"], time_lag['RFLZ'], year)
        
        for t in range(1, no_it + 1):
        
            # Interpolations to avoid staircase profile
            D = time_lag['RVKZ'][:, :, :] + (data['RVKZ'][:, :, :] - time_lag['RVKZ'][:, :, :]) * t * dt
            Utot = time_lag['RFLZ'][:, :, :] + (data['RFLZ'][:, :, :] - time_lag['RFLZ'][:, :, :]) * t * dt
            Utot_dt = time_lag['RFLZ'][:, :, :] + (data['RFLZ'][:, :, :] - time_lag['RFLZ'][:, :, :]) * (t-1) * dt

            for r in range(len(titles['RTI'])):

                if D[r] == 0.0:
                    continue

                # DSiK contains the change in shares
                dSik = np.zeros([len(titles['FTTI']), len(titles['FTTI'])])

                # F contains the preferences
                F = np.ones([len(titles['FTTI']), len(titles['FTTI'])]) * 0.5

                k_1 = np.zeros([len(titles['FTTI']), len(titles['FTTI'])])
                k_2 = np.zeros([len(titles['FTTI']), len(titles['FTTI'])])
                k_3 = np.zeros([len(titles['FTTI']), len(titles['FTTI'])])
                k_4 = np.zeros([len(titles['FTTI']), len(titles['FTTI'])])

                # -----------------------------------------------------
                # Step 1: Endogenous EOL replacements
                # -----------------------------------------------------
                for b1 in range(len(titles['FTTI'])):

                    if  not (data_dt['ZEWS'][r, b1, 0] > 0.0 and
                             data_dt['ZTLL'][r, b1, 0] != 0.0 and
                             data_dt['ZTDD'][r, b1, 0] != .0):
                        continue

                    S_i = data_dt['ZEWS'][r, b1, 0]

                    for b2 in range(b1):

                        if  not (data_dt['ZEWS'][r, b2, 0] > 0.0 and
                                 data_dt['ZTLL'][r, b2, 0] != 0.0 and
                                 data_dt['ZTDD'][r, b2, 0] != 0.0):
                            continue


                        S_k = data_dt['ZEWS'][r, b2, 0]

                        Aik = data['ZEWA'][0, b1, b2] * data['ZCET'][r, b1, c6ti['16 Turnover rate']]
                        Aki = data['ZEWA'][0, b2, b1] * data['ZCET'][r, b2, c6ti['16 Turnover rate']]

                        # Propagating width of variations in perceived costs
                        dFik = sqrt(2) * sqrt((data_dt['ZTTD'][r, b1, 0]*data_dt['ZTTD'][r, b1, 0] + data_dt['ZTTD'][r, b2, 0]*data_dt['ZTTD'][r, b2, 0]))

                        # Consumer preference incl. uncertainty
                        Fik = 0.5 * (1 + np.tanh(1.25*(data_dt['ZTTC'][r, b2, 0] - data_dt['ZTTC'][r, b1, 0]) / dFik))

                        # Preferences are then adjusted for regulations
                        F[b1, b2] = Fik*(1.0-isReg[r, b1]) * (1.0 - isReg[r, b2]) + isReg[r, b2]*(1.0-isReg[r, b1]) + 0.5*(isReg[r, b1]*isReg[r, b2])
                        F[b2, b1] = (1.0-Fik)*(1.0-isReg[r, b2]) * (1.0 - isReg[r, b1]) + isReg[r, b1]*(1.0-isReg[r, b2]) + 0.5*(isReg[r, b2]*isReg[r, b1])


                        # Runge-Kutta market share dynamiccs
                        k_1[b1, b2] = S_i*S_k * (Aik*F[b1, b2] - Aki*F[b2, b1])
                        k_2[b1, b2] = (S_i + dt*k_1[b1, b2]/2)*(S_k-dt*k_1[b1, b2]/2)* (Aik*F[b1, b2] - Aki*F[b2, b1])
                        k_3[b1, b2] = (S_i + dt*k_2[b1, b2]/2)*(S_k-dt*k_2[b1, b2]/2) * (Aik*F[b1, b2] - Aki*F[b2, b1])
                        k_4[b1, b2] = (S_i + dt*k_3[b1, b2])*(S_k-dt*k_3[b1, b2]) * (Aik*F[b1, b2] - Aki*F[b2, b1])

                        # This method currently applies RK4 to the shares, but all other components of the equation are calculated for the overall time step
                        # We must assume the the LCOE does not change significantly in a time step dt, so we can focus on the shares.

                        dSik[b1, b2] = dt*(k_1[b1, b2] + 2*k_2[b1, b2] + 2*k_3[b1, b2] + k_4[b1, b2])/6 #*data['ZCEZ'][r, 0, 0]
                        dSik[b2, b1] = -dSik[b1, b2]

                        # Market share dynamics
#                        dSik[b1, b2] = S_i*S_k* (Aik*F[b1, b2] - Aki*F[b2, b1])*dt#*data['ZCEZ'][r, 0, 0]
#                        dSik[b2, b1] = -dSik[b1, b2]

                # Calculate temporary market shares and temporary capacity from endogenous results
                endo_shares = data_dt['ZEWS'][r, :, 0] + np.sum(dSik, axis=1) 
                endo_capacity = endo_shares * Utot[r, np.newaxis]

                # Add in exogenous sales figures. These are blended with
                # endogenous result! Note that it's different from the
                # ExogSales specification!
                Utot_d = D[r, 0, 0]
                dSk = np.zeros([len(titles['FTTI'])])
                dUk = np.zeros([len(titles['FTTI'])])
                dUkTK = np.zeros([len(titles['FTTI'])])
                dUkREG = np.zeros([len(titles['FTTI'])])
                ZWSA_scalar = 1.0

                # Check that exogenous sales additions aren't too large
                # As a proxy it can't be greater than 80% of the fleet size
                # divided by 15 (the average lifetime of freight vehicles)
                if (data['ZWSA'][r, :, 0].sum() > 0.8 * Utot[r] / 15):
            
                    ZWSA_scalar = data['ZWSA'][r, :, 0].sum() \
                                  / (0.8 * Utot[r] / 15)

                # Check that exogenous capacity is smaller than regulated capacity
                # Regulations have priority over exogenous capacity
                reg_vs_exog = ((data['ZWSA'][r, :, 0] / ZWSA_scalar / no_it + endo_capacity) 
                              > data['ZREG'][r, :, 0]) & (data['ZREG'][r, :, 0] >= 0.0)
             
                # ZWSA is yearly capacity additions. We need to split it up based on the number of time steps, and also scale it if necessary.
                dUkTK =  np.where(reg_vs_exog, 0.0, data['ZWSA'][r, :, 0] \
                                  / ZWSA_scalar / no_it)

                # Correct for stretching effect in regulations. This is the difference in capacity due only to increasing fleet size.
                # This is the difference between the endogenous capacity, and what the endogenous capacity would have been
                # if rflz (i.e. total vehicles) had not grown.
                dUkREG = -(endo_capacity - endo_shares * Utot_dt[r,np.newaxis]) \
                         * isReg[r, :].reshape([len(titles['FTTI'])])
                                           
                # Sum effect of exogenous sales additions (if any) with effect of regulations. 
                dUk = dUkTK + dUkREG
                dUtot = np.sum(dUk)

                # Calaculate changes to endogenous capacity, and use to find new market shares
                # Zero capacity will result in zero shares
                # All other capacities will be streched
                data['ZEWS'][r, :, 0] = (endo_capacity + dUk)/(np.sum(endo_capacity)+dUtot)

                if ~np.isclose(np.sum(data['ZEWS'][r, :, 0]), 1.0, atol = 1e-5):
                    msg = (f"Sector: {sector} - Region: {titles['RTI'][r]} - Year: {year}"
                    "Sum of market shares do not add to 1.0 (instead: {np.sum(data['ZEWS'][r, :, 0])})")
                    warnings.warn(msg)

                if np.any(data['ZEWS'][r, :, 0] < 0.0):
                    msg = (f"Sector: {sector} - Region: {titles['RTI'][r]} - Year: {year}"
                    "Negative market shares detected! Critical error!")
                    warnings.warn(msg)
                    
                # Copy over costs that don't change
                data['ZCET'][:, :, 1:22] = data_dt['ZCET'][:, :, 1:22]

                data['ZESG'][r, :, 0] = D[r, 0, 0]/data['ZLOD'][r, 0, 0]

                # ZESD is share difference between small and large trucks
                data['ZESD'][r, 0, 0] = data['ZEWS'][r, 0, 0] + data['ZEWS'][r, 2, 0] + data['ZEWS'][r, 4, 0] + \
                data['ZEWS'][r, 6, 0] + data['ZEWS'][r, 8, 0] + data['ZEWS'][r, 10, 0] + data['ZEWS'][r, 12, 0] + \
                data['ZEWS'][r, 14, 0] + data['ZEWS'][r, 16, 0] + data['ZEWS'][r, 18, 0]

                data['ZESD'][r, 1, 0] = 1 - data['ZESD'][r, 0, 0]

                if data['ZESD'][r, 0, 0] > 0:
                    for x in range(0, 20, 2):
                        data['ZESA'][r, x, 0] = data['ZEWS'][r, x, 0]/data['ZESD'][r, 0, 0]
                        data['ZEVV'][r, x, 0] = data['ZESG'][r, x, 0]*data['ZESA'][r, x, 0]/(1-1/(data['ZSLR'][r, 0, 0] + 1))
                        data['ZEST'][r, x, 0] = data['ZEVV'][r, x, 0]*data['ZLOD'][r, 1, 0]
                
                if data['ZESD'][r, 1, 0] > 0:
                    for x in range(1, 21, 2):
                        data['ZESA'][r, x, 0] = data['ZEWS'][r, x, 0]/data['ZESD'][r, 1, 0]
                        data['ZEVV'][r, x, 0] = data['ZESG'][r, x, 0]*data['ZESA'][r, x, 0]/(1/(data['ZSLR'][r, 0, 0] + 1))
                        data['ZEST'][r, x, 0] = data['ZEVV'][r, x, 0]*data['ZLOD'][r, 1, 0]

                # This is number of trucks by technology
                data['ZEWK'][r, :, 0] = data['ZEWS'][r, :, 0] * Utot[r, 0, 0]

            # Investment (sales) = new capacity created
            # zewi_t is new additions at current timestep/iteration
            data, zewi_t = get_sales(data, data_dt, time_lag, titles, dt, c6ti, t)
            
            # Reopen country loop
            for r in range(len(titles['RTI'])):

                if D[r] == 0.0:
                    continue
                
                # Emissions
                data['ZEWE'][r, :, 0] = data['ZEVV'][r, :, 0] * data['ZCET'][r, :, c6ti['14 CO2Emissions (gCO2/km)']] \
                                        * (1 - data['ZBFM'][r, 0, 0]) / (1e6)
                zjet = copy.deepcopy(data['ZJET'][0, :, :])
                for veh in range(len(titles['FTTI'])):
                    for fuel in range(len(titles['JTI'])):
                        #  Middle distillates
                        if titles['JTI'][fuel] == '5 Middle distillates' and data['ZJET'][0, veh, fuel]  == 1:  

                            #  Mix with biofuels if there's a biofuel mandate
                            zjet[veh, fuel] = zjet[veh, fuel] * (1.0 - data['ZBFM'][r, 0, 0])

                            # Emission correction factor
                            emis_corr[r, veh] = 1.0 - data['ZBFM'][r, 0, 0]

                        elif titles['JTI'][fuel] == '11 Biofuels'  and data['ZJET'][0, veh, fuel] == 1:

                            zjet[veh, fuel] = data['ZJET'][0, veh, fuel] * data['ZBFM'][r, 0, 0]

                # Fuel use by fuel type - Convert TJ (ZCET * ZEVV) to ktoe, so divide by 41.868
                data['ZJNJ'][r, :, 0] = (np.matmul(np.transpose(zjet), data['ZEVV'][r, :, 0] * \
                                    data['ZCET'][r, :, c6ti['9 energy use (MJ/vkm)']])) / 41.868


            # Cumulative investment, not in region loop as it is global
            bi = np.zeros((len(titles['RTI']), len(titles['FTTI'])))
            for r in range(len(titles['RTI'])):
                bi[r, :] = np.matmul(data['ZEWB'][0, :, :], zewi_t[r, :, 0])

            dw = np.sum(bi, axis = 0)
            data['ZEWW'][0, :, 0] = data_dt['ZEWW'][0, :, 0] + dw
            
            
            
            quarterly_additions_freight = zewi_t[:, :, 0] * data["ZCET"][:, :, c6ti['21 Battery capacity (kWh)']]
            quarterly_additions_freight = (quarterly_additions_freight) / 1e6  # Convert kWh to GWh.
            summed_quarterly_capacity_freight = np.sum(quarterly_additions_freight)  # Summing across sectors
            
            data["Battery cap additions"][2, t-1, 0] = summed_quarterly_capacity_freight
            
            # Copy over the technology cost categories that do not change 
            # Copy over the initial cost matrix
            data["ZCET initial"] = np.copy(data_dt['ZCET initial'])
            
            # Battery learning
            battery_cost_frac = battery_costs(data, time_lag, year, titles)
            
            for tech in range(len(titles['FTTI'])):
                    
                # Only for those tech with batteries
                if np.any(data["ZCET"][:, tech, c6ti['22 Battery cost ($/kWh)']] > 0):
                    
                    data["ZCET"][:, tech, c6ti['22 Battery cost ($/kWh)']] = (
                        data['ZCET initial'][:, tech, c6ti['22 Battery cost ($/kWh)']]
                        * battery_cost_frac )
            
            # Save battery cost
            #data["ZEBC"] = np.zeros([len(titles['RTI']), len(titles['FTTI']), 1])
            #data["ZEBC"][:, :, 0] = data["ZCET"][:, :, c6ti['22 Battery cost ($/kWh)']]
            nonbat_cost = np.zeros([len(titles['RTI']), len(titles['FTTI']),1])
            nonbat_cost_dt = np.zeros([len(titles['RTI']), len(titles['FTTI']),1])
            
            # Learning-by-doing effects on investment
            for tech in range(len(titles['FTTI'])):

                if data['ZEWW'][0, tech, 0] > 0.1:
                    
                    # For EVs, add the battery costs to the non-battery costs
                    # TODO: make battery costs dt a global variable in some way. 
                    if tech in [12, 13, 18, 19]:
                        nonbat_cost_dt[:, tech, 0] = (
                                data_dt['ZCET'][:, tech, c6ti['1 Price of vehicles (USD/vehicle)']] 
                                - data_dt["ZCET"][:, tech, c6ti['22 Battery cost ($/kWh)']]  
                                * data_dt["ZCET"][:, tech, c6ti['21 Battery capacity (kWh)']]
                                )
                        nonbat_cost[:, tech, 0] = ( nonbat_cost_dt[:, tech, 0]
                                                * (1.0 + data["ZCET"][:, tech, c6ti['15 Learning exponent']]
                                                * dw[tech] / data['ZEWW'][0, tech, 0])
                                                )

                        data['ZCET'][:, tech, c6ti['1 Price of vehicles (USD/vehicle)']] =  (
                                nonbat_cost[:, tech, 0] 
                                + (data["ZCET"][:, tech, c6ti['22 Battery cost ($/kWh)']]  
                                * data["ZCET"][:, tech, c6ti['21 Battery capacity (kWh)']])
                                )
                        test = 1
                    
                    # For non-EVs, add only the non-battery costs
                    else:
                        data['ZCET'][:, tech, c6ti['1 Price of vehicles (USD/vehicle)']] =  \
                                data_dt['ZCET'][:, tech, c6ti['1 Price of vehicles (USD/vehicle)']] \
                                * (1.0 + data["ZCET"][:, tech, c6ti['15 Learning exponent']]
                                * dw[tech] / data['ZEWW'][0, tech, 0])
                        
                        # Introducing LBD on O&M costs, assuming the same learning rate. #TODO: think about this again
                        # Doesn't make too much sense for non-EVs, but does make sense for EVs
                        data['ZCET'][:, tech, c6ti['5 O&M costs (USD/km)']] =  \
                                data_dt['ZCET'][:, tech, c6ti['5 O&M costs (USD/km)']] \
                                * (1.0 + data["ZCET"][:, tech, c6ti['15 Learning exponent']]
                                * dw[tech] / data['ZEWW'][0, tech, 0])


            
            # Calculate total investment by technology in terms of truck purchases
            for r in range(len(titles['RTI'])):
                data['ZWIY'][r, :, 0] = data['ZEWI'][r, :, 0] \
                                        * data["ZCET"][r, :, c6ti['1 Price of vehicles (USD/vehicle)']]

            # Calculate levelised cost again
            carbon_costs = set_carbon_tax(data, c6ti)
            data = get_lcof(data, titles, carbon_costs, year)


            # Update time loop variables:
            for var in time_lag.keys():

                if var.startswith("R"):

                    data_dt[var] = copy.deepcopy(data[var])

            for var in time_lag.keys():

                if var.startswith("Z"):

                    data_dt[var] = copy.deepcopy(data[var])
        
        if year == 2050 and t == no_it:
            print(f"Total small electric trucks in 2050 is: {np.sum(data['ZEWK'][:, 12, 0])/10**6:.2f} M trucks")

    return data
