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
# Standard library imports
from pathlib import Path

# Third party imports
import numpy as np
import pandas as pd

# Local library imports
from ftt_source.paths import get_utilities_path


# %% tech-to-resource mapping and related lookups
# -----------------------------------------------------------------------------

def get_tech_to_resource(titles):
    """Read converters.csv and return the tech-to-resource index mapping.

    Returns a list where entry j is the 0-based index into titles['ERTI'] of the
    resource used by technology j in titles['T2TI'].  Adding or removing rows in
    converters.csv automatically updates the mapping.
    """
    csv_path = get_utilities_path() / 'titles' / 'converters.csv'
    df = pd.read_csv(csv_path, keep_default_na=False)

    t2ti = list(titles['T2TI'])
    erti = list(titles['ERTI'])
    tech_resource_map = dict(zip(df['T2TI'], df['ERTI']))

    return [erti.index(tech_resource_map[tech]) for tech in t2ti]


def get_erti_jti_map(titles):
    """Read ERTI_JTI_map.csv and return a dict of erti_idx -> [jti_idx, ...].

    Used to sum non-power fuel demand (MEWD) into resource demand (MEPD) for
    fossil-fuel ERTI entries without hardcoding JTI integer indices.
    """
    csv_path = get_utilities_path() / 'titles' / 'ERTI_JTI_map.csv'
    df = pd.read_csv(csv_path, keep_default_na=False)

    erti = list(titles['ERTI'])
    jti = list(titles['JTI'])
    result = {}
    for _, row in df.iterrows():
        erti_idx = erti.index(row['ERTI'])
        jti_labels = [label.strip() for label in row['JTI'].split(',')]
        result[erti_idx] = [jti.index(label) for label in jti_labels]
    return result


def get_cf_multipliers(titles):
    """Read T2TI_CF_multiplier.csv and return a dict of tech_idx -> multiplier.

    Technologies absent from the CSV are not included (treat as multiplier=1).
    Used to apply technology-specific corrections to capacity factors computed
    from cost-supply curves (e.g. CSP is twice as efficient as Solar PV).
    """
    csv_path = get_utilities_path() / 'titles' / 'T2TI_CF_multiplier.csv'
    df = pd.read_csv(csv_path, keep_default_na=False)

    t2ti = list(titles['T2TI'])
    return {t2ti.index(row['T2TI']): float(row['CF_multiplier'])
            for _, row in df.iterrows()}


def _detect_power_schema(t2ti):
    """Return '22tech' or '12tech' by probing a discriminating T2TI label.

    22-tech uses '11 Biomass'; 12-tech uses '7 Bioenergy'.  These labels are
    mutually exclusive across the two schemas so a single membership test is
    sufficient.  Raises ValueError if neither sentinel is found.
    """
    if '11 Biomass' in t2ti:
        return '22tech'
    if '7 Bioenergy' in t2ti:
        return '12tech'
    raise ValueError(
        "Cannot detect FTT:Power T2TI schema: expected '11 Biomass' (22-tech) "
        f"or '7 Bioenergy' (12-tech) in T2TI. Got: {t2ti}"
    )


def get_gen_tech_indices(titles):
    """Return T2TI and ERTI index pairs used in the generation-based primary energy demand block.

    Each entry maps a resource name to (t2ti_index_or_list, erti_index).
    The schema (22-tech or 12-tech) is detected automatically from titles['T2TI'].

    22-tech: separate entries for Biomass/Biomass+CCS, Waste/Waste+CCS, Marine,
             Geothermal, Onshore, Offshore, Solar PV, CSP, Large Hydro, Pumped Hydro.
    12-tech: Bioenergy and Bioenergy+CCS replace Biomass+Waste; biogas gets an empty
             list (zero demand). Marine and Geothermal are merged into Other Renewable;
             Onshore and Offshore both map to Wind (approximation — both ERTI slots
             receive the full aggregate Wind generation). CSP is merged into Solar.
    """
    t2ti = list(titles['T2TI'])
    erti = list(titles['ERTI'])
    schema = _detect_power_schema(t2ti)

    if schema == '22tech':
        return {
            'nuclear':    (t2ti.index('1 Nuclear'),
                           erti.index('1 Nuclear')),
            'biomass':    ([t2ti.index('11 Biomass'), t2ti.index('12 Biomass + CCS')],
                           erti.index('5 Biomass')),
            'biogas':     ([t2ti.index('5 Waste'), t2ti.index('6 Waste + CCS')],
                           erti.index('6 Biogas')),
            'tidal':      (t2ti.index('16 Marine'),
                           erti.index('8 Tidal')),
            'hydro':      ([t2ti.index('13 Large Hydro'), t2ti.index('14 Pumped Hydro')],
                           erti.index('9 Large Hydro')),
            'onshore':    (t2ti.index('17 Onshore'),
                           erti.index('10 Onshore')),
            'offshore':   (t2ti.index('18 Offshore'),
                           erti.index('11 Offshore')),
            'solar':      ([t2ti.index('19 Solar PV'), t2ti.index('20 CSP')],
                           erti.index('12 Solar PV')),
            'geothermal': (t2ti.index('15 Geothermal'),
                           erti.index('13 Geothermal')),
        }

    # 12-tech schema
    return {
        'nuclear':    (t2ti.index('1 Nuclear'),
                       erti.index('1 Nuclear')),
        'biomass':    ([t2ti.index('7 Bioenergy'), t2ti.index('8 Bioenergy + CCS')],
                       erti.index('5 Biomass')),
        'biogas':     ([],
                       erti.index('6 Biogas')),
        'tidal':      (t2ti.index('12 Other Renewable'),
                       erti.index('8 Tidal')),
        'hydro':      ([t2ti.index('9 Hydro')],
                       erti.index('9 Large Hydro')),
        'onshore':    (t2ti.index('11 Wind'),
                       erti.index('10 Onshore')),
        'offshore':   (t2ti.index('11 Wind'),
                       erti.index('11 Offshore')),
        'solar':      ([t2ti.index('10 Solar')],
                       erti.index('12 Solar PV')),
        'geothermal': (t2ti.index('12 Other Renewable'),
                       erti.index('13 Geothermal')),
    }


# %% marginal cost of production of non renewable resources
# -----------------------------------------------------------------------------
# ----------------------- fuel costs non-renewables ---------------------------
# -----------------------------------------------------------------------------

def marginal_costs_nonrenewables(MEPD, RERY, MPTR, BCSC, HistC, MRCL, MERC, dt,
                      num_regions, num_resources):
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

    P[:4] = MRCL[0, :4, 0]     # Marginal costs of non-renewable resources are identical globally, use Belgium
    MEPD_sum = np.sum(MEPD[:, :, 0], axis=0)   # Sum over regions
    demand_non_renewables = MEPD_sum[:4]       # Global demand for non-renewable resources

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
        MERC[:, j, 0] = P[j]
        
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


def update_capacity_factors(BCET, BCSC, MEWL, MERC, MEPD, tech_to_resource,
                            loadfactor_type_mask, cf_multipliers):
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

    # Apply per-technology CF multipliers (e.g. CSP is more efficient than Solar PV)
    for j, multiplier in cf_multipliers.items():
        BCET[:, j, 10] *= multiplier
        MEWL[:, j, 0] *= multiplier

    return BCET, MEWL, MERC

def update_capacity_factors_original(BCET, MEWG, MEWL, BCSC, CSC_Q, MEPD, MERC,
                                     tech_to_resource, num_regions, num_techs, cf_multipliers):
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

                    if j in cf_multipliers:
                        BCET[r, j, 10] *= cf_multipliers[j]

                    if(MEWG[r, j, 0] > 0.01 and X0 > 0):

                        # Average capacity factor costs up to the point of use (integrate CSCurve divided by total use)
                        CFvar = np.sum(1.0 / (Y[:ind + 1] + 1e-6) / (ind + 1))

                        if CFvar > 0:
                            MEWL[r, j, 0] = CFvar

                        if j in cf_multipliers:
                            MEWL[r, j, 0] *= cf_multipliers[j]

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
                num_regions, num_techs, num_resources, year, dt, tech_to_resource,
                erti_jti_map, cf_multipliers, gen_tech_indices):
    '''
    FTT: Power cost-supply curves routine.
    This calculates the cost of resources given the available supply.
    '''
    # Sum electricity generation for all regions, transform into PJ (resource histograms in PJ)
    # MEWD is the non-power demand for resources
    # RERY is set in FTTinvP for fossil fuels

    # Uranium
    nuclear_tech_idx, uranium_res_idx = gen_tech_indices['nuclear']
    MEPD[:, uranium_res_idx, 0] = MEWG[:, nuclear_tech_idx, 0] / BCET[:, nuclear_tech_idx, 13] * 3.6e-3
    # Fossil fuels: sum MEWD carrier demands into resource demand using CSV-driven mapping
    for erti_idx, jti_indices in erti_jti_map.items():
        MEPD[:, erti_idx, 0] = sum(MEWD[:, j, 0] for j in jti_indices)

    # Renewable resource use is local, so equal to total resource demand MEPD
    # We assume RERY equal to MEPD regionally, although it is globally
    # Biomass (the cost curve is in PJ)
    biomass_tech_idxs, biomass_res_idx = gen_tech_indices['biomass']
    MEPD[:, biomass_res_idx, 0] = sum(MEWG[:, t, 0] / BCET[:, t, 13] for t in biomass_tech_idxs) * 3.6e-3
    RERY[:, biomass_res_idx, 0] = MEPD[:, biomass_res_idx, 0]
    # Biogas (which comes mostly from waste) (From here onwards cost curves are in TWh)
    biogas_tech_idxs, biogas_res_idx = gen_tech_indices['biogas']
    MEPD[:, biogas_res_idx, 0] = sum(MEWG[:, t, 0] / BCET[:, t, 13] for t in biogas_tech_idxs) * 3.6e-3
    RERY[:, biogas_res_idx, 0] = MEPD[:, biogas_res_idx, 0]

    # Tidal
    tidal_tech_idx, tidal_res_idx = gen_tech_indices['tidal']
    MEPD[:, tidal_res_idx, 0] = MEWG[:, tidal_tech_idx, 0] * 3.6e-3
    RERY[:, tidal_res_idx, 0] = MEPD[:, tidal_res_idx, 0]
    # Hydro
    hydro_tech_idxs, hydro_res_idx = gen_tech_indices['hydro']
    MEPD[:, hydro_res_idx, 0] = sum(MEWG[:, t, 0] for t in hydro_tech_idxs) * 3.6e-3
    RERY[:, hydro_res_idx, 0] = MEPD[:, hydro_res_idx, 0]
    # Onshore
    onshore_tech_idx, onshore_res_idx = gen_tech_indices['onshore']
    MEPD[:, onshore_res_idx, 0] = MEWG[:, onshore_tech_idx, 0] * 3.6e-3
    RERY[:, onshore_res_idx, 0] = MEPD[:, onshore_res_idx, 0]
    # Offshore
    offshore_tech_idx, offshore_res_idx = gen_tech_indices['offshore']
    MEPD[:, offshore_res_idx, 0] = MEWG[:, offshore_tech_idx, 0] * 3.6e-3
    RERY[:, offshore_res_idx, 0] = MEPD[:, offshore_res_idx, 0]
    # Solar (PV + CSP)
    solar_tech_idxs, solar_res_idx = gen_tech_indices['solar']
    MEPD[:, solar_res_idx, 0] = sum(MEWG[:, t, 0] / BCET[:, t, 13] for t in solar_tech_idxs) * 3.6e-3
    RERY[:, solar_res_idx, 0] = MEPD[:, solar_res_idx, 0]
    # Geothermal
    geothermal_tech_idx, geothermal_res_idx = gen_tech_indices['geothermal']
    MEPD[:, geothermal_res_idx, 0] = MEWG[:, geothermal_tech_idx, 0] * 3.6e-3
    RERY[:, geothermal_res_idx, 0] = MEPD[:, geothermal_res_idx, 0]


    # All regions have the same information
    MERC[:, 0, 0] = MRCL[:, 0, 0]
    MERC[:, 1, 0] = MRCL[:, 1, 0]
    MERC[:, 2, 0] = MRCL[:, 2, 0]
    MERC[:, 3, 0] = MRCL[:, 3, 0]
    MERC[:, 4, 0] = MRCL[:, 4, 0]

    
    # ========================================================================
    # Go down the cost-supply curves
    # ========================================================================

    L = 990
    lmo = np.arange(L)          # This will have length 990, from 0 to 989
    HistC = np.zeros([num_resources, L])

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
        BCET, BCSC, MEWL, MERC, MEPD, tech_to_resource, loadfactor_type_mask, cf_multipliers
        )
    
    # Type 3: Investment cost type of cost-supply curve
    # If you turn this one on again, make sure CSC_Q are defined for these techs too
    # MERC, BCET = update_investment_cost(
    #     BCET, BCSC, CSC_Q, MEPD, MERC, tech_to_resource, investment_type_mask
    #     )
           

                        
    
# %%    
    # Add REN resources (resource # >4) in MRED, and remaining resources in MRES    
    MRED[:, 4:, 0] = BCSC[:, 4:, 2] * 3.6

    # Remaining technical potential
    MRES[:, 4:, 0] = BCSC[:, 4:, 2] * 3.6 - MEPD[:, 4:, 0]

    return BCET, BCSC, MEWL, MEPD, MERC, RERY, MRED, MRES
