# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:38:50 2024

@author: Femke Nijsse
"""

import numpy as np
import copy


# %% survival function
# -----------------------------------------------------------------------------
# -------------------------- Survival calcultion ------------------------------
# -----------------------------------------------------------------------------
def survival_function(data, time_lag, histend, year, titles):
    # In this function we calculate scrappage, sales, tracking of age, and
    # average efficiency.
    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}


    # Create a generic matrix of fleet-stock by age
    # Assume uniform distribution, but only do so when we still have historical
    # market share data. Afterwards it becomes endogeous
    if year < histend['TEWS']:

        # TODO: This needs to be replaced with actual data
        correction = np.linspace(1/(3*len(titles['VYTI'])), 3/len(titles['VYTI']), len(titles['VYTI'])) * 0.6

        for age in range(len(titles['VYTI'])):

            data['RLTA'][:, :, age] = correction[age] *  data['TEWK'][:, :, 0]

    else:
        # Once we start to calculate the market shares and total fleet sizes
        # endogenously, we can update the vehicle stock by age matrix and
        # calculate scrappage, sales, average age, and average efficiency.
        for r in range(len(titles['RTI'])):

            for veh in range(len(titles['VTTI'])):

                # Move all vehicles one year up:
                # New sales will get added to the age-tracking matrix in the main
                # routine.
                data['RLTA'][r, veh, :-1] = copy.deepcopy(time_lag['RLTA'][r, veh, 1:])

                # Current age-tracking matrix:
                # Only retain the fleet that survives
                data['RLTA'][r, veh, :] = data['RLTA'][r, veh, :] * data['TESF'][r, 0, :]

                # Total amount of vehicles that survive:
                survival = np.sum(data['RLTA'][r, veh, :])

                # EoL scrappage: previous year's stock minus what survived
                if time_lag['TEWK'][r, veh, 0] > survival:

                    data['REVS'][r, veh, 0] = time_lag['TEWK'][r, veh, 0] - survival

                elif time_lag['TEWK'][r, veh, 0] < survival:
                    if year > 2016:
                        msg = """
                        Erronous outcome!
                        Check year {}, region {} - {}, vehicle {} - {}
                        Vehicles that survived are greater than what was in the fleet before:
                        {} versus {}
                        """.format(year, r, titles['RTI'][r], veh,
                                   titles['VTTI'][veh], time_lag['TEWK'][r, veh, 0], survival)
#                        print(msg)

    # calculate fleet size
    return data