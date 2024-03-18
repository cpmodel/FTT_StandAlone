# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:38:50 2024
TODO: use pre-simulation TEWK values to improve the age distribution before simulation starts
"""



import numpy as np
import copy


def get_survival_ratio(survival_function):
    """Transform survival function
    into a year-on-year ratio of survival.
    
    The survival ratio is reshapen to work with RLTA"""
    
    survival_ratio = survival_function[:, :-1, :] / survival_function[:, 1:, :] 
    survival_ratio = survival_ratio.reshape(71, 1, 22)
    return survival_ratio

def add_new_cars_age_matrix(capacity, lagged_capacity, scrappage):
    """Add new cars to the age matrix.
    Add the growth in capacity (pos or neg) to the scrappage    
    
    New car additions are set to zero if calculation is negative (f.i. with regulation)
    
    """
    capacity_growth = capacity - lagged_capacity
    new_cars = capacity_growth[:, :, 0] + scrappage[:, :, 0]
    # Set new cars to zero
    new_cars = np.where(new_cars < 0, 0, new_cars)
        
    
    return new_cars
    

def survival_function(data, time_lag, histend, year, titles):
    """
    Survival function for each car type
    
    We calculate the number of cars by age bracket (RLTA) and scrappage (REVS), 
    Ultimately, we want to include sales and average efficiency in this function too.
        
    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution
    """
    
    # TODO: Implement survival function to get a more accurate depiction of
    # vehicles being phased out and to be able to track the age of the fleet.
    # This means that a new variable will need to be implemented which is
    # basically TP_VFLT with a third dimension (vehicle age in years- up to 23y)
    # Reduced efficiences can then be tracked properly as well.
    
    # Categories for the cost matrix (BTTC)
    c3ti = {category: index for index, category in enumerate(titles['C3TI'])}
    survival_ratio = get_survival_ratio(data['TESF'])
    
    # Create a generic matrix of fleet-stock by age
    # Assume uniform distribution, but only do so when we still have historical
    # market share data. Afterwards it becomes endogeous
    if year <= np.min(data["TDA1"][:, 0, 0]):
        
        # VYTI has 23 year categories (number 0-22)
        # TODO: This needs to be replaced with actual data
        correction = np.linspace(1/(3*len(titles['VYTI'])), 3/len(titles['VYTI']), len(titles['VYTI'])) * 0.6

        for age in range(len(titles['VYTI'])):
            # RLTA -> age-tracking matrix of cars
            data['RLTA'][:, :, age] = correction[age] *  data['TEWK'][:, :, 0]
        
        # Sum over the age dimension
        survival = np.sum(data['RLTA'][:, :, 1:] * survival_ratio[:, :, :], axis=2)
        data['REVS'][:, :, 0] = time_lag["TEWK"][:, :, 0] - survival

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
                # Only retain the fleet that survives; TESF is the survival function
                data['RLTA'][r, veh, :-1] = data['RLTA'][r, veh, :-1] * survival_ratio[r, :, :]

                # Total amount of vehicles that survive:
                survival = np.sum(data['RLTA'][r, veh, :])

                # EoL vehicle scrappage: previous year's stock minus what survived
                if time_lag['TEWK'][r, veh, 0] > survival:

                    data['REVS'][r, veh, 0] = time_lag['TEWK'][r, veh, 0] - survival

                elif time_lag['TEWK'][r, veh, 0] < survival:
                    if year > 2016:
                        msg = (
                            f"Error! \n"
                            f"Check year {year}, region {r} - {titles['RTI'][r]}, vehicle {veh} - {titles['VTTI'][veh]}\n"
                            "Vehicles that survived are greater than what was in the fleet before:\n"
                            f"{time_lag['TEWK'][r, veh, 0]} versus {survival}"
                            )
                        print(msg)
        test = 1

    # calculate fleet size
    test=1
    return data