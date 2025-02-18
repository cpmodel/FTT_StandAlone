# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:03:40 2025

@author: User
"""

import numpy as np
from SourceCode.ftt_core.ftt_mandate import get_new_sales_under_mandate, get_mandate_share

MANDATE_START_YEAR = 2025
N_YEARS = 16

def ensure_minimum_bev_share(sales_in_class, emissions_intensity, min_share=0.03):
    """Ensure BEV sales meet minimum share (used for emissions regulation)"""
    sales = np.copy(sales_in_class)
    total_sales = np.sum(sales)
    
    if total_sales == 0:
        return sales
        
    bev_index = np.where(emissions_intensity == 0)[0][0]
    current_bev_share = sales[bev_index] / total_sales
    
    if current_bev_share < min_share:
        required_bev_sales = total_sales * min_share
        bev_sales_increase = required_bev_sales - sales[bev_index]
        
        non_bev_indices = [i for i in range(len(sales)) if i != bev_index]
        non_bev_sales = sales[non_bev_indices]
        total_non_bev = np.sum(non_bev_sales)
        
        if total_non_bev > 0:
            reduction_factor = (total_non_bev - bev_sales_increase) / total_non_bev
            for idx in non_bev_indices:
                sales[idx] *= reduction_factor
        
        sales[bev_index] = required_bev_sales
    
    return sales

def calculate_target_emissions(year, veh_class, start_year=2025, end_year=2040):
    """Calculate target emissions for emissions regulation"""
    start_emissions = {
        0: float('inf'),  # TWV
        1: 327,    # LCV 
        2: 879,    # MDT
        3: 1354,   # HDT
        4: float('inf')  # Bus
    }
    
    if year < start_year:
        return float('inf')
    elif year >= end_year:
        return 0.0
    
    total_years = end_year - start_year
    years_in = year - start_year
    reduction_factor = 1 - (years_in / total_years)
    return start_emissions[veh_class] * reduction_factor

def get_fleet_emissions(sales, emissions_intensity):
    """Calculate fleet emissions for emissions regulation"""
    if np.sum(sales) == 0:
        return 0.0
    
    total_emissions = np.sum(sales * emissions_intensity)
    return total_emissions / np.sum(sales)

def find_advanced_counterpart(tech_idx, emissions_intensity):
    """
    Find the advanced version of a technology based on emissions patterns.
    Returns None if no advanced version exists or if already advanced.
    """
    base_emission = emissions_intensity[tech_idx]
    
    # Check next technology to see if it's the advanced version
    if tech_idx + 1 < len(emissions_intensity):
        next_emission = emissions_intensity[tech_idx + 1]
        # If next technology has lower emissions and isn't zero-emission
        if 0 < next_emission < base_emission:
            return tech_idx + 1
    
    return None

def redistribute_sales_by_emissions(sales_in_class, emissions_intensity, target_emissions):
    """
    Redistribute sales to meet emissions target.
    Now tries advanced ICE versions before jumping to BEV.
    """
    sales = np.copy(sales_in_class)
    total_sales = np.sum(sales)
    
    if total_sales == 0:
        return sales
        
    # Find BEV index (technology with zero emissions)
    bev_index = np.where(emissions_intensity == 0)[0][0]
    
    # Sort non-BEV technologies by emissions intensity
    ice_indices = [i for i in range(len(sales)) if emissions_intensity[i] > 0]
    ice_indices.sort(key=lambda x: emissions_intensity[x], reverse=True)
    
    current_emissions = get_fleet_emissions(sales, emissions_intensity)
    
    while current_emissions > target_emissions and current_emissions > 0:
        for ice_idx in ice_indices:
            if sales[ice_idx] > 0:
                # First try to shift to advanced version if available
                adv_idx = find_advanced_counterpart(ice_idx, emissions_intensity)
                
                # Calculate how much to transfer (limit to 50% of total sales)
                transfer_amount = min(sales[ice_idx], total_sales * 0.25)
                
                if adv_idx is not None and emissions_intensity[adv_idx] > 0:
                    # Transfer to advanced ICE version
                    sales[ice_idx] -= transfer_amount
                    sales[adv_idx] += transfer_amount
                else:
                    # If no advanced version or still too high emissions, shift to BEV
                    sales[ice_idx] -= transfer_amount
                    sales[bev_index] += transfer_amount
                
                # Recalculate fleet emissions
                current_emissions = get_fleet_emissions(sales, emissions_intensity)
                
                if current_emissions <= target_emissions:
                    break
    
    return sales

def implement_mandate(cap, EV_truck_mandate, EV_truck_kickstarter, emissions_regulation_active,
                     cum_sales_in, sales_in, n_veh_classes, year, emissions_intensity=None):
    """
    Implement either standard mandate, kickstarter, or emissions regulation based on active flag.
    Only one policy can be active at a time.
    """
    cum_sales_after = np.copy(cum_sales_in)
    sales_after = np.copy(sales_in)
    
    # Standard EV Mandate
    if EV_truck_mandate[0, 0, 0] not in [-1, 0]:
        mandate_end_year = MANDATE_START_YEAR + N_YEARS
        if EV_truck_mandate[0, 0, 0] in range(2010, 2040) and year > EV_truck_mandate[0, 0, 0]:
            mandate_end_year = EV_truck_mandate[0, 0, 0]
            
        mandate_share = get_mandate_share(year, MANDATE_START_YEAR, mandate_end_year)
        
        if np.sum(mandate_share) > 0:
            for veh_class in range(n_veh_classes):
                sales_in_class = sales_in[:, veh_class::n_veh_classes, :]
                green_indices_class = [6]
                
                sales_after_mandate_class = get_new_sales_under_mandate(
                    sales_in_class, mandate_share, green_indices_class)
                sales_after[:, veh_class::n_veh_classes] = sales_after_mandate_class
                
                # Update capacity and sales
                sales_difference = sales_after_mandate_class - sales_in_class
                cap[:, veh_class::n_veh_classes, :] += sales_difference
                cap[:, veh_class::n_veh_classes, 0] = np.maximum(
                    cap[:, veh_class::n_veh_classes, 0], 0)
                cum_sales_after[:, veh_class::n_veh_classes, 0] += sales_difference[:, :, 0]
    
    # Kickstarter Policy
    elif EV_truck_kickstarter[0, 0, 0] != 0:
        target_share = 0.10 if year == 2024 else 0.20 if year == 2025 else 0.03
        
        for veh_class in range(n_veh_classes):
            sales_in_class = sales_in[:, veh_class::n_veh_classes, :]
            bev_idx = 6
            
            # Process each region
            for r in range(sales_in_class.shape[0]):
                total_sales = np.sum(sales_in_class[r, :, 0])
                if total_sales > 0:
                    current_bev_sales = sales_in_class[r, bev_idx, 0]
                    current_bev_share = current_bev_sales / total_sales
                    
                    if current_bev_share < target_share:
                        target_bev_sales = total_sales * target_share
                        sales_after[r, veh_class + bev_idx * n_veh_classes, 0] = target_bev_sales
                        
                        # Reduce other vehicles proportionally
                        non_bev_indices = [i for i in range(sales_in_class.shape[1]) if i != bev_idx]
                        total_non_bev = np.sum(sales_in_class[r, non_bev_indices, 0])
                        if total_non_bev > 0:
                            reduction_factor = (total_sales - target_bev_sales) / total_non_bev
                            for idx in non_bev_indices:
                                sales_after[r, veh_class + idx * n_veh_classes, 0] = (
                                    sales_in_class[r, idx, 0] * reduction_factor)
            
            # Update capacity and sales
            sales_difference = (
                sales_after[:, veh_class::n_veh_classes, :] - 
                sales_in[:, veh_class::n_veh_classes, :])
            cap[:, veh_class::n_veh_classes, :] += sales_difference
            cap[:, veh_class::n_veh_classes, 0] = np.maximum(
                cap[:, veh_class::n_veh_classes, 0], 0)
            cum_sales_after[:, veh_class::n_veh_classes, 0] += sales_difference[:, :, 0]
    
    # Emissions Regulation
    elif emissions_regulation_active[0, 0, 0] != 0 and emissions_intensity is not None:
        for veh_class in range(n_veh_classes):
            sales_in_class = sales_in[:, veh_class::n_veh_classes, :]
            emissions_for_class = emissions_intensity[:, veh_class::n_veh_classes]
            
            for r in range(sales_in_class.shape[0]):
                # First ensure minimum BEV share
                new_sales = ensure_minimum_bev_share(
                    sales_in_class[r, :, 0],
                    emissions_for_class[r, :]
                )
                
                # Then apply emissions target
                target_emissions = calculate_target_emissions(year, veh_class)
                new_sales = redistribute_sales_by_emissions(
                    new_sales,
                    emissions_for_class[r, :],
                    target_emissions
                )
                
                sales_after[r, veh_class::n_veh_classes, 0] = new_sales
            
            # Update capacity and sales
            sales_difference = (
                sales_after[:, veh_class::n_veh_classes, :] - 
                sales_in[:, veh_class::n_veh_classes, :])
            cap[:, veh_class::n_veh_classes, :] += sales_difference
            cap[:, veh_class::n_veh_classes, 0] = np.maximum(
                cap[:, veh_class::n_veh_classes, 0], 0)
            cum_sales_after[:, veh_class::n_veh_classes, 0] += sales_difference[:, :, 0]
    
    # No active policy - maintain minimum 3% share
    else:
        min_share = 0.03
        for veh_class in range(n_veh_classes):
            sales_in_class = sales_in[:, veh_class::n_veh_classes, :]
            green_indices_class = [6]
            
            sales_after_mandate_class = get_new_sales_under_mandate(
                sales_in_class, min_share, green_indices_class)
            sales_after[:, veh_class::n_veh_classes] = sales_after_mandate_class
            
            # Update capacity and sales
            sales_difference = sales_after_mandate_class - sales_in_class
            cap[:, veh_class::n_veh_classes, :] += sales_difference
            cap[:, veh_class::n_veh_classes, 0] = np.maximum(
                cap[:, veh_class::n_veh_classes, 0], 0)
            cum_sales_after[:, veh_class::n_veh_classes, 0] += sales_difference[:, :, 0]
    
    return cum_sales_after, sales_after, cap