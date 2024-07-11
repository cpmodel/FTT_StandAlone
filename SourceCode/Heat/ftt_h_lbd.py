# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:54:08 2024

@author: Rishi
"""

from SourceCode.support.learning_by_doing import generalized_learning_by_doing

def learning_by_doing_h(data, data_dt, time_lag, titles, c4ti, dt, hewi_t):
    return generalized_learning_by_doing('heat', data, data_dt, time_lag=time_lag, titles=titles, c4ti=c4ti, dt=dt, hewi_t=hewi_t)

"""
def learning_by_doing_h(data, data_dt, time_lag, titles, c4ti, dt, hewi_t):
    bi = np.zeros((len(titles['RTI']),len(titles['HTTI'])))
    for r in range(len(titles['RTI'])):
        bi[r,:] = np.matmul(data['HEWB'][0, :, :],hewi_t[r, :, 0])
    dw = np.sum(bi, axis=0)

    # Cumulative capacity incl. learning spill-over effects
    data['HEWW'][0, :, 0] = data_dt['HEWW'][0, :, 0] + dw

    # Copy over the technology cost categories that do not change (all except prices which are updated through learning-by-doing below)
    data['BHTC'] = copy.deepcopy(data_dt['BHTC'])

    # Learning-by-doing effects on investment and efficiency
    for b in range(len(titles['HTTI'])):

        if data['HEWW'][0, b, 0] > 0.0001:

            data['BHTC'][:, b, c4ti['1 Inv cost mean (EUR/Kw)']] = (data_dt['BHTC'][:, b, c4ti['1 Inv cost mean (EUR/Kw)']]  \
                                                                     *(1.0 + data['BHTC'][:, b, c4ti['7 Investment LR']] * dw[b]/data['HEWW'][0, b, 0]))
            data['BHTC'][:, b, c4ti['2 Inv Cost SD']] = (data_dt['BHTC'][:, b, c4ti['2 Inv Cost SD']]  \
                                                                     *(1.0 + data['BHTC'][:, b, c4ti['7 Investment LR']] * dw[b]/data['HEWW'][0, b, 0]))
            data['BHTC'][:, b, c4ti['9 Conversion efficiency']] = (data_dt['BHTC'][:, b, c4ti['9 Conversion efficiency']] \
                                                                    * 1.0 / (1.0 + data['BHTC'][:, b, c4ti['20 Efficiency LR']] * dw[b]/data['HEWW'][0, b, 0]))


    #Total investment in new capacity in a year (m 2014 euros):
      #HEWI is the continuous time amount of new capacity built per unit time dI/dt (GW/y)
      #BHTC(:,:,1) are the investment costs (2014Euro/kW)
    data['HWIY'][:,:,0] = data['HWIY'][:,:,0] + data['HEWI'][:,:,0]*dt*data['BHTC'][:,:,0]/data['PRSC14'][:,0,0,np.newaxis]
    # Save investment cost for front end
    data["HWIC"][:, :, 0] = data["BHTC"][:, :, c4ti['1 Inv cost mean (EUR/Kw)']]
    # Save efficiency for front end
    data["HEFF"][:, :, 0] = data["BHTC"][:, :, c4ti['9 Conversion efficiency']]
    
    return data
"""