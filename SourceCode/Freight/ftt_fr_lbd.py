# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:00:33 2024

@author: Rishi
"""


from SourceCode.support.learning_by_doing import generalized_learning_by_doing

def learning_by_doing_fr(data, data_dt, titles, c6ti, dt, dw):
    return generalized_learning_by_doing('freight', data, data_dt, titles=titles, c6ti=c6ti, dt=dt, dw=dw)

"""
import numpy as np
import copy
from SourceCode.Freight.ftt_fr_lcof import get_lcof

def learning_by_doing_fr(data, data_dt, titles, c6ti, dt, dw):
    for tech in range(len(titles['FTTI'])):
        if data['ZEWW'][0, tech, 0] > 0.1:
            data['ZCET'][:, tech, c6ti['1 Price of vehicles (USD/vehicle)']] =  \
                    data_dt['ZCET'][:, tech, c6ti['1 Price of vehicles (USD/vehicle)']] * \
                    (1.0 + data['ZLER'][tech] * dw[tech]/data['ZEWW'][0, tech, 0])

    # Calculate total investment by technology in terms of truck purchases
    for r in range(len(titles['RTI'])):
        data['ZWIY'][r, :, 0] = data_dt['ZWIY'][r, :, 0] + \
        data['ZEWY'][r, :, 0]*dt*data['ZCET'][r, :, c6ti['1 Price of vehicles (USD/vehicle)']]*1.263

    # Calculate levelised cost again
    data = get_lcof(data, titles)

    return data
"""