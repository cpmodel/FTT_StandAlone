# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 08:46:30 2025

@author: Femke
"""

import numpy as np

def check_market_shares(shares, titles, sector, year):
    '''
    Checks if the region sum of market shares is one and whether there are
    regions and technologies with negative shares.
    
    Raises ValueErrors if there are problems'''
    
    # Sum market shares for each region. 
    total_shares = shares[:, :, 0].sum(axis=1)
    
    # Using a lax threshold (1e-4) because FTT:Tr pseudoshares in empty regions may not sum exactly to 1.
    if sector != 'freight':
        invalid = np.abs(total_shares - 1.0) > 1e-4
    else: # Freight: total per region depends on number of vehicle classes (3, 4, or 5)
        allowed = np.array([3.0, 4.0, 5.0])
        # Invalid if not close to any allowed total
        invalid = (np.abs(total_shares[:, None] - allowed) > 1e-4).all(axis=1)
    
    # Raise error listing regions with invalid totals
    if np.any(invalid):
        regions = [titles['RTI'][r] for r in np.where(invalid)[0]]
        shares = [f"{total_shares[r]:.4f}" for r in np.where(invalid)[0]]
        messages = [f"{region} (sum={share})" for region, share in zip(regions, shares)]
        raise ValueError(
            f"Sector: {sector} - Year: {year} - Invalid market share sums in regions: "
            + ", ".join(messages)
        )
    
    # Check for negative shares and raise error if found
    negatives = (shares[:, :, 0] < 0.0).any(axis=1)
    if np.any(negatives):
        regions = [titles['RTI'][r] for r in np.where(negatives)[0]]
        raise ValueError(
            f"Sector: {sector} - Year: {year} - Negative market shares detected in regions: "
            + ", ".join(regions)
        )