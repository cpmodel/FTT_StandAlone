# -*- coding: utf-8 -*-


import numpy as np

from SourceCode.ftt_core.ftt_mandate import get_new_sales_under_mandate, get_mandate_share


green_indices = range(30, 35)  # Indices for green technologies
MANDATE_START_YEAR = 2025
N_YEARS = 16


def implement_seeding(cap, seeding, cum_sales_in, sales_in, n_veh_classes, year):    
    '''Implement mandate: linearly increasing sales. First recalculate sales, then 
    recalculate capacity''' 
    
    mandate_end_year = 2030

    # If there are no mandates, immediately return inputs
    if seeding == 0 or year not in range(2025, mandate_end_year + 1):
        return cum_sales_in, sales_in, cap
    
    cum_sales_after_mandate = np.copy(cum_sales_in)
    sales_after_mandate = np.copy(sales_in)
            
        
    for veh_class in range(n_veh_classes):
        
        sales_in_class = sales_in[:, veh_class::n_veh_classes, :]
        green_indices_class = [6]
        
        # Step 4: Apply mandate adjustments with global shares and strict enforcement
        green_share = np.sum(sales_in_class[:, green_indices_class])/np.sum(sales_in_class)
        mandate_share = get_mandate_share(year, MANDATE_START_YEAR, mandate_end_year) * 0.15 * green_share
        
        # If the mandate is turned off that specific year, go to next vehicle class
        if np.sum(mandate_share) == 0:
            continue
        
        # Recompute sales, after implementation of mandate
        sales_after_mandate_class = get_new_sales_under_mandate(sales_in_class, mandate_share,
                                                                green_indices_class)
        sales_after_mandate[:, veh_class::n_veh_classes] = sales_after_mandate_class

        # Step 5: Update capacity
        sales_difference = sales_after_mandate_class - sales_in_class
        cap[:, veh_class::n_veh_classes, :] = cap[:, veh_class::n_veh_classes, :] + sales_difference
        cap[:, veh_class::n_veh_classes, 0] = np.maximum(cap[:, veh_class::n_veh_classes, 0], 0)
        
        # Step 6: Update cumulative sales
        cum_sales_after_mandate[:, veh_class::n_veh_classes, 0] += sales_difference[:, :, 0]
    
    return cum_sales_after_mandate, sales_after_mandate, cap
    



def implement_mandate(cap, EV_truck_mandate, cum_sales_in, sales_in, n_veh_classes, year):    
    '''Implement mandate: linearly increasing sales. First recalculate sales, then 
    recalculate capacity''' 
    
    # If there are no mandates, immediately return inputs
    if np.all(EV_truck_mandate[:, 0, 0] == 0):
        return cum_sales_in, sales_in, cap
    
    cum_sales_after_mandate = np.copy(cum_sales_in)
    sales_after_mandate = np.copy(sales_in)
    mandate_end_year = MANDATE_START_YEAR + N_YEARS
    
    if EV_truck_mandate[0, 0, 0] in range(2010, 2040) and year > EV_truck_mandate[0, 0, 0]:
        # For the sequencing, I'm changing the end year
        mandate_end_year = EV_truck_mandate[0, 0, 0]
    
    if EV_truck_mandate[0, 0, 0] in range(2040, 2060):
        # For the sectoral interactions, I'm simply trying to halve the mandate by stretching it out
        mandate_end_year = EV_truck_mandate[0, 0, 0]
        
        
    # Step 4: Apply mandate adjustments with global shares and strict enforcement
    mandate_share = get_mandate_share(year, MANDATE_START_YEAR, mandate_end_year)
    
    # If the mandate is turned off that specific year, also return inputs
    if np.sum(mandate_share) == 0:
        return cum_sales_in, sales_in, cap
    
    # Select countries for which the mandate is turned on
    regions = np.where(EV_truck_mandate != 0)[0]

    if EV_truck_mandate[0, 0, 0] not in [-1, 0] and np.sum(mandate_share) > 0:
        
        for veh_class in range(n_veh_classes):
            sales_in_class = sales_in[:, veh_class::n_veh_classes, :]
            green_indices_class = [6]
            
            # Recompute sales, after implementation of mandate
            sales_after_mandate_class = get_new_sales_under_mandate(sales_in_class, mandate_share,
                                                                    green_indices_class, regions=regions)
            sales_after_mandate[:, veh_class::n_veh_classes] = sales_after_mandate_class
    
            # Step 5: Update capacity
            sales_difference = sales_after_mandate_class - sales_in_class
            cap[:, veh_class::n_veh_classes, :] = cap[:, veh_class::n_veh_classes, :] + sales_difference
            cap[:, veh_class::n_veh_classes, 0] = np.maximum(cap[:, veh_class::n_veh_classes, 0], 0)
            
            # Step 6: Update cumulative sales
            cum_sales_after_mandate[:, veh_class::n_veh_classes, 0] += sales_difference[:, :, 0]
        
        return cum_sales_after_mandate, sales_after_mandate, cap
    
    



