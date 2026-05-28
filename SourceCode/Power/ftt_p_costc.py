'''
=========================================
ftt_p_costc.py
=========================================
Power cost-supply curves module.


Cost-supply curves give the cost of a resource as a function of its quantity, and
hence provide the **marginal cost** of each resource. For example, if a fossil fuel
resource is running out, the fuel cost increases. For e.g. hydropower, if there are
less accessible places to build, then the investment cost goes up. For variable
renewables, the load factor (the number of hours it delivers to the grid, divide by the
number of hours in a year).


Functions included:
    - marginal_costs_nonrenewables
        Calculates marginal cost of production of non renewable resources
    - cost_curves
        Calculates cost-supply curves for the power sector


'''
<<<<<<< HEAD

=======
>>>>>>> origin/main
# Third party imports
import numpy as np

<<<<<<< HEAD
# Local library imports
from SourceCode.support.divide import divide

# Make NumPy raise an error on overflow
np.seterr(over='raise', under='raise')

# %% interpolation function
# -----------------------------------------------------------------------------
# -------------------------- Interpolate ------------------------------
# -----------------------------------------------------------------------------

def interp(X, Y, X0, L):
    '''

    Linear interpolation function which estimates data points between start points
    and end points.


    Parameters
    -----------
    X: List
        X vector for interpolation
    Y: List
        Y vector for interpolation
    X0: float
        Value at which to interpolate
    L: int
        Length of data set


    Returns
    ----------
    Y0: float
        Interpolated value
    I: int
        Index of the position of the interpolated value


    '''
    # So that we don't change the incoming L
    LL = L/2
    # X Data spacing: assumes homogenous spacing
    D = abs(X[1] - X[0])
    # We do a table lookup algorithm
    I = int(LL) - 1
    while abs(X[I] - X0) > D:
        LL = LL/2
        if X[I] > X0:
            I = I - max(int(LL), 1)    # int() truncates decimals to int
        else:
            I = I + max(int(LL), 1)
        # These conditionals refer to cases where X0 falls outside of X
        if I == 0:
            I = 1
            break
        if I == L-1:
            I = L - 1
            break

#    if I > X.shape[0]:
#        print("stop")
    if(X0 < X[I] and I > 1):
        X1 = X[I - 1]
        Y1 = Y[I - 1]
        X2 = X[I]
        Y2 = Y[I]
        # Interpolate linearly between (X1, Y1) and (X2, Y2) at X0
        Y0 = Y1 + (Y2 - Y1)*(X0 - X1)/(X2 - X1)
    elif (X0 >= X[I] and I < L - 1):
        X1 = X[I]
        Y1 = Y[I]
        X2 = X[I + 1]
        Y2 = Y[I + 1]
        # Interpolate linearly between (X1, Y1) and (X2, Y2) at X0
        Y0 = Y1 + (Y2 - Y1)*(X0 - X1)/(X2 - X1)
    elif(X0 < X[I] and I == 1):  # If X0 is below the range we take the last value
        Y0 = Y[1]
    elif(X0 > X[I] and I == L - 1):  # If X0 is above the range
        Y0 = Y[L - 1]

    return Y0, I
=======
>>>>>>> origin/main

# %% marginal cost of production of non-renewable resources
# -----------------------------------------------------------------------------
<<<<<<< HEAD
# -------------------------- marginal calculation -----------------------------
# -----------------------------------------------------------------------------

#@njit(fastmath=True) # Doesn't work either
def marginal_function(MEPD, RERY, MPTR, BCSC, HistC, MRCL, MERC, MRED, MRES, dt):
=======
# ----------------------- fuel costs non-renewables ---------------------------
# -----------------------------------------------------------------------------

def marginal_costs_nonrenewables(MEPD, RERY, MPTR, BCSC, HistC, MRCL, MERC, dt,
                      num_regions, num_resources):
>>>>>>> origin/main
    '''
    Marginal cost of production of non-renewable resources.

    Parameters
    -----------
      MPTR: NumPy array
        Values of nu by region (Production to reserve ratio) 
        See paper Mercure & Salas arXiv:1209.0708 for data
      HistC: NumPy array
        Cost axis C
      MRCL: NumPy array
        Previous marginal cost value
      MEPD: NumPy array
        Total resource demand
      MERC: NumPy array
        Current (new) marginal cost value
      MRED: NumPy array
        Resources left
      MRES: NumPy array
        reserves
      RERY: NumPy array
        supply in GJ of fossil fuel resource (only I = 2 to 4 used)
      BCSC: NumPy array
        Dataset for natural resources
      num_regions, num_resources: int
        Number of regions and resources
      

    Returns
    ----------
      RERY
      BCSC
      MERC
      MRED
      MRES
      
    Notes
    -------------
    Takes about 16s to compile, so probably not worth it unless running
    a lot of scenarios. 
    
    Scipy.fsolve might be a better strategy for speed ups.. 

    '''
    
    MRED = np.zeros((num_regions, num_resources, 1))
    MRES = np.zeros((num_regions, num_resources, 1))
    P = np.zeros((4))

    # See paper Mercure & Salas arXiv:1209.0708 for data (Energy Policy 2013)
   
    # Width of the F function is resource-dependent (except coal where cost data is coarse)
    sig = np.zeros((4))
    sig[0] = 1       # Uranium
    sig[1] = 1       # Oil
    sig[2] = 1       # Coal
    sig[3] = 1       # Gas

    # First 4 elements are the non-renewable resources

<<<<<<< HEAD
    P[:4] = np.copy(MRCL[1, :4, 0])     # All marginal costs of non-renewable resources are identical (global), we use Belgium
    MEPD_sum = np.sum(MEPD[:, :, 0], axis=0)  # Sum over regions
    demand_non_renewables = np.copy(MEPD_sum[:4])       # Global demand for non-renewable resources
=======
    P[:4] = MRCL[0, :4, 0]     # Marginal costs of non-renewable resources are identical globally, use Belgium
    MEPD_sum = np.sum(MEPD[:, :, 0], axis=0)   # Sum over regions
    demand_non_renewables = MEPD_sum[:4]       # Global demand for non-renewable resources
>>>>>>> origin/main

    # We search for the value of P that enables enough total production to supply demand
    # i.e. the value of P that minimises the difference between dFdt and demand_non_renewables
    for j in range(4): # j is the non-renewable resource
       
        # Cost interval
        dC = HistC[j, 1] - HistC[j, 0]
        dFdt = 0
        count = 0
        const = 1.25 * 2 * sig[j] 
        
        while abs((dFdt - demand_non_renewables[j]) / demand_non_renewables[j]) > 0.01  and count < 20:
            
            # Sum total supply dFdt from all extraction cost ranges below marginal cost P 
            # 1 (or close to 1) when P(rice) > HistC, 0 otherwise (tc in FORTRAN)
            costs_below_price = \
                0.5 - 0.5 * np.tanh(const * (HistC[j, :] - P[j]) /  P[j] )
            
            # The supply dFdt is determined from:
            # the regional production-to-reserve ratio MPTR,
            # the BCSC contains (sparse) regional matrix of reserves at cost level
            # and where_costs_below_price selects costs hist under the currrent cost guess P, with smoothing
            dFdt = np.sum(MPTR[:, j] * BCSC[: , j, 4:] * costs_below_price[np.newaxis, :] * dC)

            # Work out price
            # Difference between supply and demand.
            # As we usually go up, the step in that direction is larger
            if dFdt < demand_non_renewables[j]:
                # Increase the marginal cost
                P[j] = P[j] * \
                    (1.0 + np.abs(dFdt - demand_non_renewables[j]) / demand_non_renewables[j] / 8)
            else:
                # Decrease the marginal cost
                P[j] = ( P[j] / (1.0 + np.abs(dFdt - demand_non_renewables[j]) / demand_non_renewables[j] / 5) )
            count = count + 1
        
        # Remove used resources from the regional histograms (uranium, oil, coal and gas only)
        BCSC[:, j, 4:] = BCSC[:, j, 4:] - (MPTR[:, j] * BCSC[:, j, 4:] * (0.5 - 0.5 * np.tanh(const * (HistC[j, :] - P[j]) /  P[j]))) * dt
        RERY[:, j, 0] = np.sum((MPTR[:, j] * BCSC[:, j, 4:]
                                * (0.5 - 0.5 * np.tanh(const * (HistC[j, :] - P[j]) /  P[j]))) * dC)

        # Write back new marginal cost values (same value for all regions)
<<<<<<< HEAD
        MERC[:, j, 0] = np.copy(P[j])
=======
        MERC[:, j, 0] = P[j]
>>>>>>> origin/main
        
        # How much non-renewable resources are left in the cost curves
        HistCSUM = np.sum(HistC, axis=1)
        MRED[:, j, 0] = MRED[:, j, 0] + np.sum(BCSC[:, j, 4:], axis=1) * (
                    BCSC[:, j, 2] - BCSC[:, j, 1])/(BCSC[:, j, 3] - 1)
        MRES[:, j, 0] = MRES[:, j, 0] + np.sum(BCSC[:, j, 4:], axis=1) * (
                    BCSC[:, j, 2] - BCSC[:, j, 1])/(BCSC[:, j, 3] - 1) * (0.5 - 0.5 * np.tanh(1.25 * 2 * (HistCSUM[j] - P[j]) / P[j]))


    return RERY, BCSC, MERC, MRED, MRES


# %% cost curves function
# -----------------------------------------------------------------------------
# -------------------------- Cost Curves ------------------------------
# -----------------------------------------------------------------------------

<<<<<<< HEAD
#@njit(fastmath=True) ## Doesn't work!
def cost_curves(BCET, BCSC, MEWD, MEWG, MEWL, MEPD, MERC, MRCL, RERY, MPTR, MRED, MRES, rti, t2ti, erti, year, dt):
=======

def update_fuel_costs(BCET, MERC, MRCL, tech_to_resource, fuel_type_mask):
    """For nuclear, oil, coal and gas, update fuel costs based on changes to MERC
    compared to the previous time step
    """
        
    # Get indices where we need to update
    regions, techs = np.where(fuel_type_mask)
    resource_idx = np.array([tech_to_resource[j] for j in techs])
    
    # Vectorized calculation for all fuel technologies at once
    marginal_cost_changes = (MERC[regions, resource_idx, 0] - 
                           MRCL[regions, resource_idx, 0])
    
    # Resource use efficiencies
    efficiencies = BCET[regions, techs, 13]
    
    # Add cost changes
    BCET[regions, techs, 4] += marginal_cost_changes * 3.6 / efficiencies
        
    return BCET


def update_capacity_factors(BCET, BCSC, MEWL, MERC, MEPD, tech_to_resource, loadfactor_type_mask):
    """New capacity factor method. Much less strict that the previous version
    based on work by Rishi Sahastrabuddhe."""
    
    k = 5 # Constant that determined how fast the load factors go up close to technical potential
    
    # Select applicable techs and resources
    regions, techs = np.where(loadfactor_type_mask)
    resource_idx = np.array([tech_to_resource[j] for j in techs])
    
    # How much generation in each region/tech combo in right units
    X0s = MEPD[regions, resource_idx, 0] / 3.6  # PJ -> TWh
    
    # Share of technical potential used
    share_TP_used = X0s / (BCSC[regions, resource_idx, 2] + 1e-6)
    starting_CF = 1.0 / BCSC[regions, resource_idx, 4]
    
    # Initially, little effect of cost-supply curves. CFs go down close to technical potential
    BCET[regions, techs, 10] = starting_CF * (1 - np.exp(-k*(1-share_TP_used)))
    MERC[regions, resource_idx, 0] = starting_CF * (1 - np.exp(-k*(1-share_TP_used)))
    # Integrate to find the average capacity factor
    MEWL[regions, techs, 0] = (starting_CF / share_TP_used
                               * (share_TP_used - (np.exp(k*share_TP_used) - 1) / (k*np.exp(k)) ) )
    
    # Catch extreme values
    BCET[regions, techs, 10] = np.where(share_TP_used > 0.98,
                                        starting_CF * 0.1,
                                        BCET[regions, techs, 10])
    MEWL[regions, techs, 0] = np.where(share_TP_used > 0.98,
                                       starting_CF * 0.8,
                                       MEWL[regions, techs, 0])
    
    # CSP is twice as efficient as solar power typically
    BCET[:, 19, 10] = BCET[:, 18, 10] * 2.0
    MEWL[:, 19, 0] = MEWL[:, 18, 0] * 2.0
    
    return BCET, MEWL, MERC

def update_capacity_factors_original(BCET, MEWG, MEWL, BCSC, CSC_Q, MEPD, MERC,
                                     tech_to_resource, num_regions, num_techs, ):
    # Type 2: Original variable renewables cost curves (reduced capacity factors)
    CFvar = np.zeros([990])
   
    # Update costs in the technology cost matrix BCET (BCET(:, :, 11) is the type of cost curve)
    for r in range(num_regions):           # Loop over region
        for j in range(num_techs):      # Loop over technology 
            if(MEPD[r, tech_to_resource[j]] > 0.0):

               
                # For variable renewables (e.g. wind, solar, wave)
                # the overall (average) capacity factor decreases as new units have lower and lower CFs
                if(BCET[r, j, 11] == 0):

                    X = CSC_Q[r, tech_to_resource[j], :]
                    Y = BCSC[r, tech_to_resource[j], 4:]
    
                    # Note: the curve is in the form of an inverse capacity factor in BCSC
                    X0 = MEPD[r, tech_to_resource[j], 0]/3.6 #PJ -> TWh
                    if X0 > 0.0:
                        Y0 = np.interp(X0, X, Y)
                        ind = np.searchsorted(X, X0)  # Find corresponding index
                    MERC[r, tech_to_resource[j], 0] = 1.0/(Y0 + 0.000001)
                    BCET[r, j, 10] = 1.0/(Y0 + 0.000001)         # We use an inverse here
                    
                    
                    # CSP is more efficient than PV by a factor 2
                    if j == 19 :
                        BCET[r, j, 10] = 1.0/(Y0 + 0.0000001) * 2.0
    
                    if(MEWG[r, j, 0] > 0.01 and X0 > 0):
    
                        # Average capacity factor costs up to the point of use (integrate CSCurve divided by total use)
                        CFvar = np.sum(1.0 / (Y[:ind + 1] + 1e-6) / (ind + 1))
                        
                        if CFvar > 0:
                            MEWL[r, j, 0] = CFvar
                     
                        # CSP is more efficient than PV by a factor 2
                        if j == 19 :
                            MEWL[r, j, 0] = CFvar * 2.0
        
        return BCET, MEWL, MERC


def update_investment_cost(BCET, BCSC, CSC_Q, MEPD, MERC, tech_to_resource, investment_type_mask):
    '''Increasing investment term type of limit, for technologies such as
    hydropower and geothermal. Unrealistically strict, so turned off'''
    
    regions, techs = np.where(investment_type_mask)
    resource_idx = np.array([tech_to_resource[j] for j in techs])
    X0s = MEPD[regions, resource_idx, 0] / 3.6  # PJ -> TWh
    
    X = CSC_Q[regions, resource_idx, :]
    Y = BCSC[regions, resource_idx, 4:]
    
    Y0s = np.zeros_like(X0s)
    for ri, r in enumerate(regions):
        Y0s[ri] = np.interp(X0s[ri], X[ri], Y[ri])
    
    MERC[regions, resource_idx, 0] = Y0s
    BCET[regions, techs, 2] = Y0s
    
    return MERC, BCET


def cost_curves(BCET, BCSC, MEWD, MEWG, MEWL, MEPD, MERC, MRCL, RERY, MPTR, MRED, MRES,
                num_regions, num_techs, num_resources, year, dt):
>>>>>>> origin/main
    '''
    FTT: Power cost-supply curves routine.
    This calculates the cost of resources given the available supply.
    '''

<<<<<<< HEAD
    L = 990
    lmo = np.arange(L)          # This will have length 990, from 0 to 989
    CSC_Q = np.zeros([71, 14, L])
    HistC = np.zeros([14, L])
    #HistQ = np.zeros([14, 990])
    X = np.zeros([990])
    Y = np.zeros([990])
    Ind = 0
    CFvar = np.zeros([990])

    # Resources classification:
    # Correspondence vector between NT2 and NER (Technologies and resources: if I = Tech, II(I) = resource)
    # Setting storage to biomass, as it's turned off
    tech_to_resource = [0, 1, 2, 2, 5, 5, 3, 3, 3, 3, 4, 4, 8, 8, 12, 7, 9, 10, 11, 11, 4, 4]

    # BCSC is natural resource data with dimensions NER NR and length of cost axis k

    # Unpack histograms
    # First 4 values in each BCSC(I, J, :) vectors are:
    # Parameters: (1) Type (2) Min (3) Max (4) Number of data points
    # Resource data type: (0) Capacity Factor reduction (1) Histogram, (2) Fuel cost, (3) Investment cost 

    for i in range(len(erti)): # Resource classification
        # if the data type contained in k=0 (k=1 in fortran) is a histogram (same for all regions)
        if BCSC[0, i, 0] == 1:

            HistC[i, :] = BCSC[0, i, 1]  \
                        + lmo * (BCSC[0, i, 2] - BCSC[0, i, 1]) / (BCSC[0, i, 3]-1)
            # BCSC goes up to L+4, Hist goes up to L
            #HistQ[j, :] = HistQ[j, :] + sum(BCSC[:, j, 4:], axis=0) #actual histograms, adds up regions, after 4 k is PJ/bin size (PJ/dollar), units sorted into bins are PJs of energy

        # resource type not histogram
        else:

            for j in range(len(rti)):

                #QuantityAxis(K) = min(Q) + (K-1) * (max(Q)-min(Q))/(N data points -1) # We don't need this loop either I believe
                CSC_Q[j, i, :] = BCSC[j, i, 1] \
                            + lmo * (BCSC[j, i, 2] - BCSC[j, i, 1]) / (BCSC[j, i, 3] - 1)


    # Calculate non-renewable resource use
    # Non renewable resource use (treated global <=> identical for all regions)

=======
>>>>>>> origin/main
    # Sum electricity generation for all regions, transform into PJ (resource histograms in PJ)
    # MEWD is the non-power demand for resources
    # RERY is set in FTTinvP for fossil fuels

    # Uranium
    MEPD[:, 0, 0] = MEWG[:, 0, 0] / BCET[:, 0, 13] * 3.6e-3
    # Oil
    MEPD[:, 1, 0] = MEWD[:, 2, 0] + MEWD[:, 3, 0] + MEWD[:, 4, 0]
    # Coal
    MEPD[:, 2, 0] = MEWD[:, 0, 0] + MEWD[:, 1, 0]
    # Gas
<<<<<<< HEAD
    MEPD[:, 3, 0] = np.copy(MEWD[:, 6, 0])
=======
    MEPD[:, 3, 0] = MEWD[:, 6, 0]
>>>>>>> origin/main

    # Renewable resource use is local, so equal to total resource demand MEPD
    # We assume RERY equal to MEPD regionally, although it is globally
    # Biomass (the cost curve is in PJ)
<<<<<<< HEAD
    MEPD[:, 4, 0] =  (divide(MEWG[:, 10, 0], BCET[:, 10, 13])  \
                    + divide(MEWG[:, 11, 0], BCET[:, 11, 13])) \
                    * 3.6/1000 # +  MEWD[11, :]   +   MEWD[10, :]
    RERY[:, 4, 0] = np.copy(MEPD[:, 4, 0]) #in PJ
    # Biogas (From here onwards cost curves are in TWh)
    MEPD[:, 5, 0] = (MEWG[:, 12, 0] \
                   + MEWG[:, 13, 0] * divide(BCET[:, 13, 13], BCET[:, 12, 13])) \
                    * 3.6/1000
    RERY[:, 5, 0] = np.copy(MEPD[:, 5, 0]) # in PJ
    # Biogas + CCS
    MEPD[:, 6, 0] = (MEWG[:, 12, 0] \
                   + MEWG[:, 13, 0] * divide(BCET[:, 13, 13], BCET[:, 12, 13])) \
                    * 3.6/1000
    RERY[:, 6, 0] = np.copy(MEPD[:, 6, 0]) # in PJ
    # Marine
    MEPD[:, 7, 0] = MEWG[:, 15, 0] * 3.6/1000
    RERY[:, 7, 0] = np.copy(MEPD[:, 7, 0]) # in PJ
    # Hydro
    MEPD[:, 8, 0] = (MEWG[:, 12, 0] + MEWG[:, 13, 0]) * 3.6/1000
    RERY[:, 8, 0] = np.copy(MEPD[:, 8, 0]) # in PJ
    # Onshore
    MEPD[:, 9, 0] = MEWG[:, 16, 0] * 3.6/1000
    RERY[:, 9, 0] = np.copy(MEPD[:, 9, 0]) # in PJ
    # Offshore
    MEPD[:, 10, 0] = MEWG[:, 17, 0] * 3.6/1000
    RERY[:, 10, 0] = np.copy(MEPD[:, 10, 0]) # in PJ
    # Solar (PV + CSP)
    MEPD[:, 11, 0] = (divide(MEWG[:, 18, 0], BCET[:, 18, 13])  +  divide(MEWG[:, 19, 0], BCET[:, 19, 13])) * 3.6/1000
    RERY[:, 11, 0] = np.copy(MEPD[:, 11, 0]) # in PJ
    # Geothermal
    MEPD[:, 12, 0] = MEWG[:, 14, 0] * 3.6/1000
    RERY[:, 12, 0] = np.copy(MEPD[:, 12, 0]) # in PJ
    

    # All regions have the same information
    MERC[:, 0, 0] = np.copy(MRCL[:, 0, 0])
    MERC[:, 1, 0] = np.copy(MRCL[:, 1, 0])
    MERC[:, 2, 0] = np.copy(MRCL[:, 2, 0])
    MERC[:, 3, 0] = np.copy(MRCL[:, 3, 0])
    MERC[:, 4, 0] = np.copy(MRCL[:, 4, 0])
=======
    MEPD[:, 4, 0] = (MEWG[:, 10, 0] / BCET[:, 10, 13]
                    + MEWG[:, 11, 0] / BCET[:, 11, 13] ) \
                    * 3.6e-3 # +  MEWD[11, :]   +   MEWD[10, :]
    RERY[:, 4, 0] = MEPD[:, 4, 0]       # In PJ
    # Biogas (which comes mostly from waste) (From here onwards cost curves are in TWh)
    MEPD[:, 5, 0] = (MEWG[:, 4, 0] / BCET[:, 4, 13] 
                     + MEWG[:, 5, 0] / BCET[:, 5, 13] ) * 3.6e-3
    RERY[:, 5, 0] = MEPD[:, 5, 0]        # In PJ

    # Tidal
    MEPD[:, 7, 0] = MEWG[:, 15, 0] * 3.6e-3
    RERY[:, 7, 0] = MEPD[:, 7, 0] # in PJ
    # Hydro
    MEPD[:, 8, 0] = (MEWG[:, 12, 0] + MEWG[:, 13, 0]) * 3.6e-3
    RERY[:, 8, 0] = MEPD[:, 8, 0] # in PJ
    # Onshore
    MEPD[:, 9, 0] = MEWG[:, 16, 0] * 3.6e-3
    RERY[:, 9, 0] = MEPD[:, 9, 0] # in PJ
    # Offshore
    MEPD[:, 10, 0] = MEWG[:, 17, 0] * 3.6e-3
    RERY[:, 10, 0] = MEPD[:, 10, 0] # in PJ
    # Solar (PV + CSP)
    MEPD[:, 11, 0] = (MEWG[:, 18, 0] / BCET[:, 18, 13] +  MEWG[:, 19, 0] / BCET[:, 19, 13]) * 3.6e-3
    RERY[:, 11, 0] = MEPD[:, 11, 0] # in PJ
    # Geothermal
    MEPD[:, 12, 0] = MEWG[:, 14, 0] * 3.6e-3
    RERY[:, 12, 0] = MEPD[:, 12, 0] # in PJ


    # All regions have the same information
    MERC[:, 0, 0] = MRCL[:, 0, 0]
    MERC[:, 1, 0] = MRCL[:, 1, 0]
    MERC[:, 2, 0] = MRCL[:, 2, 0]
    MERC[:, 3, 0] = MRCL[:, 3, 0]
    MERC[:, 4, 0] = MRCL[:, 4, 0]
>>>>>>> origin/main

    
    # ========================================================================
    # Go down the cost-supply curves
    # ========================================================================

    L = 990
    lmo = np.arange(L)          # This will have length 990, from 0 to 989
    HistC = np.zeros([num_resources, L])

    # Resources classification:
    # Correspondence vector between techs and resources
    tech_to_resource = [0, 1, 2, 2, 5, 5, 3, 3, 3, 3, 4, 4, 8, 8, 12, 7, 9, 10, 11, 11, 6, 6]

    # BCSC is natural resource data with dimensions num_resources x num_techs x length of cost axis L

    # Unpack histograms
    # First 4 values in each BCSC[r, j, :] vectors are:
    # Parameters: (1) Type (2) Min (3) Max (4) Number of data points
    # Resource type: (0) Capacity Factor reduction (1) Histogram fuel cost (2) Fuel cost, (3) Investment cost 

    for i in range(num_resources): 
        # if the data type contained in k=0 is a histogram (same for all regions)
        if BCSC[0, i, 0] == 1:

            HistC[i, :] = BCSC[0, i, 1]  \
                        + lmo * (BCSC[0, i, 2] - BCSC[0, i, 1]) / (BCSC[0, i, 3]-1)

    # Before 2017 do not overwrite RERY for fossil fuels, and no need to calculate CSCurves
    if year <= 2013:
        return BCET, BCSC, MEWL, MEPD, MERC, RERY, MRED, MRES

    # Calculate the marginal cost of production of non-renewable resources
    
    # Non renewable resource use (treated global <=> identical for all regions)
    if year >= 2017:

<<<<<<< HEAD
        RERY, BCSC, HistC, MERC, MRED, MRES = \
                marginal_function(
                MEPD, RERY, MPTR, BCSC, HistC, MRCL, MERC, MRED, MRES, dt
                )


    # Update costs in the technology cost matrix BCET (BCET(:, :, 11) is the type of cost curve)
    for r in range(len(rti)):           # Loop over region
        for j in range(len(t2ti)):      # Loop over technology 
            if(MEPD[r, tech_to_resource[j]] > 0.0):

                # Non-renewable resources fuel costs (histograms)
                if(BCET[r, j, 11]==1):   # BCET 11 contains the type of resource
                    
                    BCET[r, j, 4] = \
                        BCET[r, j, 4] + \
                        (MERC[r, tech_to_resource[j], 0] - MRCL[r, tech_to_resource[j], 0]) * 3.6 / BCET[r, j, 13]
                
                # For renewable resources: interpolate MEPD into the cost curves.
                # Decreasing capacity factor type of limit
                # Replaced with a much simpler exponential, as this is overly punishing to wind
                elif(BCET[r, j, 11] == 0):
                    
                    X0 = MEPD[r, tech_to_resource[j], 0] / 3.6 #PJ -> TWh
                    
                    # X = np.copy(CSC_Q[r, tech_to_resource[j], :])
                    # Y = np.copy(BCSC[r, tech_to_resource[j], 4:])

                    # # Note: the curve is in the form of an inverse capacity factor in BCSC
                    
                    # if X0 > 0.0:
                    #     Y0, Ind = interp(X, Y, X0, L)
                    # MERC[r, tech_to_resource[j], 0] = 1.0/(Y0 + 0.000001)
                    # BCET[r, j, 10] = 1.0/(Y0 + 0.000001)         # We use an inverse here
                    # # For variable renewables (e.g. wind, solar, wave)
                    # # the overall (average) capacity factor decreases as new units have lower and lower CFs

                    # if(MEWG[r, j, 0] > 0.01 and Ind >= 1 and X0 > 0):

                    #     # Average capacity factor costs up to the point of use (integrate CSCurve divided by total use)
                    #     CFvar[1:Ind + 1] = 1.0 / (Y[1:Ind + 1] + 0.000001) / (Ind)
                    #     CFvar2 = sum(CFvar[1:Ind + 1])
                    #     if CFvar2 > 0:

                    #         MEWL[r, j, 0] = CFvar2
                    
                    k = 5
                    share_TP_used = X0 / (BCSC[r, tech_to_resource[j], 2] + 0.000001)
                    starting_CF = 1 / BCSC[r, tech_to_resource[j], 4]
                    try:
                        BCET[r, j, 10] = starting_CF * (1 - np.exp(-k*(1-share_TP_used)))
                        # Integrate to find the average capacity factor
                        MEWL[r, j, 0] = starting_CF/share_TP_used * (share_TP_used - (np.exp(k*share_TP_used) - 1)/ (k*np.exp(k)) )
                    except FloatingPointError as e:
                        print(f"In region {r} and for technology {j}, share_TP_used is {share_TP_used}")
                        print("Floating-point error detected:", e)
                        raise e
                    # Fix: CSP is more efficient than PV by a factor 2
                    if j == 19 :
                        BCET[r, j, 10] = BCET[r, 18, 10] * 2.0
                        
                    if share_TP_used > 0.98:
                        BCET[r, j, 10] = starting_CF * 0.1
                        MEWL[r, j, 0] = starting_CF * 0.8
                        
                # Turned off, as the initial numbers don't align with BCET, and it's too strict.
                # Increasing investment term type of limit
                # elif(BCET[r, j, 11] == 3):

                #     #------> Insert an interpolation here TODO this is from the fortran, do I need to do this?
                #     X = np.copy(CSC_Q[r, tech_to_resource[j], :])
                #     Y = np.copy(BCSC[r, tech_to_resource[j], 4:])

                #     X0 = MEPD[r, tech_to_resource[j], 0] / 3.6 #PJ -> TWh
                #     if X0> 0.0:
                #         Y0, I = interp(X, Y, X0, L)
                #     MERC[r, tech_to_resource[j], 0] = Y0
                #     BCET[r, j, 2] = Y0

    # Add REN resources left in MRED, MRES
    
    # Total technical potential r>=4 (j>4 in python)
    MRED[:, 4:, 0] = np.copy(BCSC[:, 4:, 2]) * 3.6
=======
        RERY, BCSC, MERC, MRED, MRES = marginal_costs_nonrenewables(
                MEPD, RERY, MPTR, BCSC, HistC, MRCL, MERC, dt,
                num_regions, num_resources
                )
       
    # Select the Update fuel costs (type 1), capacity factors (type 0) or investment costs (type 3)
    
    resource_in_use_mask = (MEPD[:, tech_to_resource, 0] > 0.0)
    fuel_type_mask = (BCET[:, :, 11] == 1) & resource_in_use_mask  # Type 1: Fuel type resource
    loadfactor_type_mask = (BCET[:, :, 11] == 0) & resource_in_use_mask
    # investment_type_mask = (BCET[:, :, 11] == 3) & resource_in_use_mask
    
    # Type 1: Fossil fuel and nuclear cost curves (increased fossil fuel costs)
    BCET = update_fuel_costs(
        BCET, MERC, MRCL, tech_to_resource, fuel_type_mask)

    
    # Type 2: Variable renewables cost curves (reduced capacity factors)
    # The x-axis written out for the old capacity factor cost-supply curves
    techs = np.where(BCSC[0, range(num_resources), 0] == 0)[0]
    CSC_Q = np.zeros([num_regions, num_resources, L])
    CSC_Q[:, techs, :] = (BCSC[:, techs, 1, None]
                          + lmo * (BCSC[:, techs, 2, None] - BCSC[:, techs, 1, None]) / (BCSC[:, techs, 3, None] - 1))
        
    # Type 2: New variable renewables cost curves (reduced capacity factors)
    BCET, MEWL, MERC = update_capacity_factors(
        BCET, BCSC, MEWL, MERC, MEPD, tech_to_resource, loadfactor_type_mask
        )
    
    # Type 3: Investment cost type of cost-supply curve
    # If you turn this one on again, make sure CSC_Q are defined for these techs too
    # MERC, BCET = update_investment_cost(
    #     BCET, BCSC, CSC_Q, MEPD, MERC, tech_to_resource, investment_type_mask
    #     )
           

                        
    
# %%    
    # Add REN resources (resource # >4) in MRED, and remaining resources in MRES    
    MRED[:, 4:, 0] = BCSC[:, 4:, 2] * 3.6

>>>>>>> origin/main
    # Remaining technical potential
    MRES[:, 4:, 0] = BCSC[:, 4:, 2] * 3.6 - MEPD[:, 4:, 0]

    return BCET, BCSC, MEWL, MEPD, MERC, RERY, MRED, MRES
