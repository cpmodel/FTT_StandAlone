# -*- coding: utf-8 -*-
"""
=========================================
ftt_fr_lcof.py
=========================================
Freight LCOF FTT module.
########################

Local library imports:

    Support functions:

    - `divide <divide.html>`__
        Bespoke element-wise divide which replaces divide-by-zeros with zeros

Functions included:
    - get_lcof
        Calculate levelized costs

"""

from math import sqrt

# Third party imports
import numpy as np

# Local library imports
from SourceCode.support.divide import divide


def get_lcof(data, titles):
    """
    Calculate levelized costs.

    The function calculates the levelised cost of freight transport in 2012$/t-km per
    vehicle type. These costs are then converted into 2010 Euros/t-km per vehicle type.
    It includes intangible costs (gamma values) and together
    determines the investor preferences.

    Parameters
    -----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) for all
        variables. Variable names are keys and the values are 3D NumPy arrays.
    titles: dictionary
        Titles is a container of all permissible dimension titles of the model.

    Returns
    ----------
    data: dictionary
        Data is a container that holds all cross-sectional (of time) data for
        all variables.
        Variable names are keys and the values are 3D NumPy arrays.
        The values inside the container are updated and returned to the main
        routine.

    Notes
    ---------
    Additional notes if required.
    """
    # Categories for the cost matrix (ZCET)
    c6ti = {category: index for index, category in enumerate(titles['C6TI'])}

    NTT=20 # number of technologies

    for r in range(len(titles['RTI'])):

        # Defining and Initialising Variables

        #Cost matrix
        zcet = data['ZCET'][r, :, :]

        #First, mask for lifetime
        LF = zcet[:, c6ti['8 service lifetime (y)']]
        max_LF = int(np.max(LF))
        LF_mat = np.linspace(np.zeros(len(titles['FTTI'])), max_LF-1,
                             num=max_LF, axis=1, endpoint=True)
        LF_max_mat = np.concatenate(int(max_LF)*[LF[:, np.newaxis]], axis=1)
        mask = LF_mat < LF_max_mat
        LF_mask = np.where(mask, LF_mat, 0)

        # Discount rate
        rM = zcet[:,c6ti['7 Discount rate'], np.newaxis]
        # For NPV calculations
        denominator = (1+rM)**LF_mat


        # Costs of trucks, paid once in a lifetime
        It = np.ones([len(titles['FTTI']), int(max_LF)])
        It = It * zcet[:, c6ti['1 Price of vehicles (USD/vehicle)'], np.newaxis]
        It = It / zcet[:, c6ti['18 Average Mileage'], np.newaxis]
        It[:,1:] = 0

        # Standard deviation of costs of trucks
        dIt = np.ones([len(titles['FTTI']), int(max_LF)])
        dIt = dIt * zcet[:, c6ti['2 Std of price  (USD/vehicle)'], np.newaxis]
        dIt[:,1:] = 0

        # Reg tax based on carbon price, RTCOt = ($/tCO2/km)/(tCO2/km)
        RZCOt = np.ones([len(titles['FTTI']), int(max_LF)])
        RZCOt = (RZCOt * zcet[:, c6ti['14 CO2Emissions (gCO2/km)'], np.newaxis]
              * data['RZCO'][r,0,0])
        RZCOt[:,1:] = 0

        # Registration Taxes, ZTVT is vehicle tax
        ItVT = np.ones([len(titles['FTTI']), int(max_LF)])
        ItVT = ItVT * (zcet[:, c6ti['14 CO2Emissions (gCO2/km)'], np.newaxis] + data['ZTVT'][r,:,0, np.newaxis])
        ItVT = ItVT / zcet[:, c6ti['18 Average Mileage'], np.newaxis]
        ItVT[:,1:] = 0

        # Fuel Cost
        FT = np.ones([len(titles['FTTI']), int(max_LF)])
        FT = FT * zcet[:, c6ti['3 fuel cost (USD/km)'], np.newaxis]
        FT = np.where(mask, FT, 0)

        # Standard deviation of fuel costs
        dFT = np.ones([len(titles['FTTI']), int(max_LF)])
        dFT = dFT * zcet[:, c6ti['4 std fuel cost (USD/km)'], np.newaxis]
        dFT = np.where(mask, dFT, 0)

       # fuel tax/subsidies
        fft = np.ones([len(titles['FTTI']), int(max_LF)])
        fft = fft * data['ZTFT'][r, :, 0, np.newaxis]
        fft = np.where(mask, fft, 0)

        # Lifetime vector of fuel taxes to be added
        #FtFT = np.ones([len(titles['FTTI']), int(max_LF)])
        #FtFT = FtFT * zcet[:, c6ti['9 energy use (MJ/vkm)'], np.newaxis] * FT
        #FtFT = np.where(mask, FtFT, 0)

        # O&M costs
        OMt = np.ones([len(titles['FTTI']), int(max_LF)])
        OMt = OMt * zcet[:, c6ti['5 O&M costs (USD/km)'], np.newaxis]
        OMt = np.where(mask, OMt, 0)

        # Standard deviation of O&M costs
        dOMt = np.ones([len(titles['FTTI']), int(max_LF)])
        dOMt = dOMt * zcet[:, c6ti['6 std O&M  (USD/km)'], np.newaxis]
        dOMt = np.where(mask, dOMt, 0)

        # Capacity factors
        Lfactor = np.ones([len(titles['FTTI']), int(max_LF)])
        Lfactor = Lfactor * zcet[:, c6ti['10 Loads (t/V)'], np.newaxis]

        # Road Tax
        RT = np.ones([len(titles['FTTI']), int(max_LF)])
        RT = RT * data['ZTRT'][r, :, 0, np.newaxis]
        RT = np.where(mask, RT, 0)

        # Gamma values
        Gam = np.ones([len(titles['FTTI']), int(max_LF)])
        Gam = Gam * zcet[:, c6ti['13 Gam (USD/pkm)'], np.newaxis]
        
        # Calculate LCOF without policy, and find standard deviation
        npv_expenses1 = (It+FT+OMt)/Lfactor
        npv_expenses1 = (npv_expenses1/denominator)
        npv_utility = 1/denominator
        
        # Remove 1s for tech with small lifetime than max
        npv_utility[npv_utility==1] = 0
        npv_utility[:,0] = 1
        LCOF = np.sum(npv_expenses1, axis =1)/np.sum(npv_utility, axis=1)

        dnpv_expenses1 = np.sqrt(((dIt**2)/(Lfactor**2)) + ((dFT**2)/(Lfactor**2))
        + ((dOMt**2)/(Lfactor**2)))/denominator
        dLCOF = np.sum(dnpv_expenses1, axis=1)/np.sum(npv_utility, axis=1)

        # Calculate LCOF with policy, and find standard deviation
        npv_expenses2 = (It+ItVT+FT+fft+OMt+RT)/Lfactor
        npv_expenses2 = npv_expenses2/denominator
        TLCOF = np.sum(npv_expenses2, axis=1)/np.sum(npv_utility, axis=1)
        dTLCOF=dLCOF

        # Introduce Gamma Values

        # Convert costs into logarithmic space - applying a log-normal distribution
        LTLCOF = np.log10((TLCOF**2)/np.sqrt((dTLCOF**2)+(TLCOF**2))) + Gam[:,0]

        dLTLCOF = np.sqrt(np.log10(1+(dTLCOF**2)/(TLCOF**2)))

        # Convert LCOF from 2012 dollars to 2010 Euros (factor of 1.33/1.0529)

        data['ZTLC'][r, :, 0] = LCOF*1.263
        data['ZTLD'][r, :, 0] = dLCOF*1.263
        data['ZTTC'][r, :, 0] = TLCOF*1.263
        data['ZTTD'][r, :, 0] = dTLCOF*1.263
        data['ZTLL'][r, :, 0] = LTLCOF*1.263
        data['ZTDD'][r, :, 0] = dLTLCOF*1.263

    return data
