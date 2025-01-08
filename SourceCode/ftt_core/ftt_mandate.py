# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:34:51 2024

@author: Amir Akther (refactored by Femke)
"""

import numpy as np



def get_mandate_share(year, mandate_start_year, mandate_end_year):
    """Calculate the mandate share based on the year."""
    if year < mandate_start_year:
        return 0.0
    elif year >= mandate_end_year:
        return 0.0
    else:
        # Linear increase from 0 to 1 between start and end years
        return (year + 1 - mandate_start_year) / (mandate_end_year - mandate_start_year)
    


def get_new_sales_under_mandate(sales_in, mandate_share, green_indices):
    '''Using sales_dt (RTI x techs x 1), the indices of green techs, and the
    share of sales that need to be green, compute the new sales'''
    
    sales_after_mandate = np.copy(sales_in)
    
    for r in range(sales_in.shape[0]):
        total_sales = np.sum(sales_in[r, :, 0])
        if total_sales > 0:
            current_green = np.sum(sales_in[r, green_indices, 0])
            current_share = current_green / total_sales
            
            if current_share < mandate_share:
                target_green = total_sales * mandate_share
                
                # First, use local proportions if there are sales locally
                if current_green > 0:
                    scale_factor = target_green / current_green
                    sales_after_mandate[r, green_indices, 0] *= scale_factor
                else:
                    # Use global shares for distribution
                    global_green_sales = np.sum(sales_in[:, green_indices, 0], axis=0)
                    if np.sum(global_green_sales) > 0:
                        global_shares = global_green_sales / np.sum(global_green_sales)
                        sales_after_mandate[r, green_indices, 0] = target_green * global_shares
                    else:
                        sales_after_mandate[r, green_indices, 0] = target_green / len(green_indices)

                    
                # Adjust non-green sales to maintain total
                non_green_indices = [i for i in range(sales_in.shape[1]) if i not in green_indices]
                remaining_sales = total_sales - target_green
                current_non_green = np.sum(sales_in[r, non_green_indices, 0])
                if current_non_green > 0:
                    scale_factor = remaining_sales / current_non_green
                    sales_after_mandate[r, non_green_indices, 0] *= scale_factor
                elif len(non_green_indices) > 0:
                    sales_after_mandate[r, non_green_indices, 0] = remaining_sales / len(non_green_indices)
                    
    return sales_after_mandate