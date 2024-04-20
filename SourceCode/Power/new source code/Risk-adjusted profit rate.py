# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:52:49 2024

@author: dumor
"""

from SourceCode.Power.new.ftt_p_revu import get_revu


def get_rapr(data, titles, year, histend):
    """
    calculate risk-adjsuted profit rate for each region and technology
        
    """

    for r in range(len(titles['RTI'])): #loop for regions
            for tech in range(len(titles['T2TI'])): #loop for technologies
                levelized_revenue = get_revu(data,titles,year,histend)
                levelized_cost = get_lcoe(data, titles)
                capital_unit = data['DPCC'][r,tech,0]
                
                
                profit = levelized_revenue - levelized_cost
                profit_rate = profit/capital_unit

                data['DPPR'][r,tech,0 ] = profit_rate

##risk-adjustment##
#input for risk-adjustment
#            wacc = get_wacc(data, titles,year,histend)
#            debt_rate = data['DPDR'][r,tech,0]
                
#            leverage form debt rate
#            leverage = 1 + (debt_rate/(1-debt_rate))
            
#risk-adjusted profit rate calculation
            
#            risk-adjusted_profit_rate = profit_rate*(1/(1 + wacc*leverage))

#data['DRPR'][r,tech,0 ] = risk-adjusted_profit_rate



return data
