# -*- coding: utf-8 -*-
"""
=========================================
run_file.py
=========================================
Run file for FTT Stand alone.
#############################


Programme calls the FTT stand-alone model run class, and executes model run.
Call this file from the command line (or terminal) to run FTT Stand Alone.

Local library imports:

    Model Class:

    - `ModelRun <model_class.html>`__
        Creates a new instance of the ModelRun class


"""


# Local library imports
from SourceCode.model_class import ModelRun
import numpy as np
import pandas as pd
import copy
import pickle
from pathlib import Path

# Instantiate the run
model = ModelRun()
# model.scenarios = ['S{}'.format(i) for i in [0,3]]

# Fetch ModelRun attributes, for examination
# Titles of the model
titles = model.titles
# Dimensions of model variables
dims = model.dims
# Model inputs
inputs = model.input
# Metadata for inputs of the model
histend = model.histend
# Domains to which variables belong
domain = model.domain
tl = model.timeline
scens = model.scenarios

# scen_dict = dict(zip(model.scenarios, ['REF', 'CP', 'MD', 'CP+MD']))
# %%
# Call the 'run' method of the ModelRun class to solve the model
# model.run()

# Fetch ModelRun attributes, for examination
# Output of the model
# output_all = model.output

# %%

with open(Path('.') / 'Output' / 'Results.pickle', 'rb') as f:
    output_all = pickle.load(f)
# # output_all2 = pickle.load("Output\Results.pickle")


# %% Graph init
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as mticker
import os
from SourceCode.support.divide import divide
params = {'legend.fontsize': 9,
          'figure.figsize': (5.5, 9),
          'axes.labelsize': 8,
          'axes.titlesize': 9,
          'xtick.labelsize':7,
          'ytick.labelsize':7}
pylab.rcParams.update(params)

SAVE_GRAPHS = False
FORMAT = 'svg' # png, jpeg
VERSION = 1


# %% Setup converters and colour maps

# Regions
# Global, Russia, USA, Canada, China, India, Indonesia, Saudi Arabia, LATAM, 
# EU, MENA, Japan & Korea, Australia & New Zealand, Rest of the World

region_titles = titles['RTI_short']
technology_titles = titles['HYTI']

# Region maps
region_mapping = {'Global' :                    list(region_titles),
                  'USA':                        [region_titles[33]],
                   # 'Canada':                     [region_titles[35]],
                  'China':                      [region_titles[40]],
                  'India':                      [region_titles[41]],
                  'KSA':               [region_titles[54]],
                  'MENA':                       [region_titles[idx] for idx in [51, 57, 61, 66, 69]],
                    'Brazil':                     [region_titles[43]],
                   'LATAM':                      [region_titles[idx] for idx in[42, 44, 45, 46]],
                  'EU27+UK':                    [region_titles[idx] for idx in (list(np.arange(0, 27)) + [30])],
                  'Japan + Korea':              [region_titles[idx] for idx in [34, 47]],
                  'Canada':                     [region_titles[35]],
                  'Australia':                  [region_titles[36]],
                  # 'Rest of the World':          [region_titles[idx] for idx in 
                  #                                [27, 28, 29, 31, 32, 39, 48, 50,
                  #                                52, 53, 55, 56, 58, 59, 60, 62,
                  #                                63, 64, 65, 67, 68, 70]]
                  } 

# Add all regions that are not explicitly in the mapping to the RoW category

countries_included = [reg for regagg, reglist in region_mapping.items() for reg in reglist if regagg != 'Global']
missing_countries_idx = [region_titles.index(reg) for reg in region_titles if reg not in countries_included]
region_mapping['RoW'] = [region_titles[idx] for idx in missing_countries_idx]
region_mapping_wo_glo = {reg_agg: reg for reg_agg, reg in region_mapping.items() if reg_agg != 'Global'}


region_col_map = {'Global' :                    'lime',
                  'USA':                        'blue',
                  # 'Canada':                     'orangered',
                  'China':                      'red',
                  'India':                      'darkorange',
                  # 'Russia':                     'darkgreen',
                  # 'Indonesia':                  'purple',
                  'KSA':               'saddlebrown',
                  'MENA':                       'salmon',
                   'Brazil':                     'teal',
                  'LATAM':                      'cadetblue',
                  'EU27+UK':                    'crimson',
                  'Japan + Korea':              'violet',
                  'Canada':                     'goldenrod',
                  'Australia':                  'darkgreen',
                  'RoW':                        'darkgray'
                  } 

reg_conv = pd.DataFrame(0.0, index=region_mapping.keys(), columns=region_titles)
for reg_agg, reg_orig in region_mapping.items():
    
    for r in reg_orig:
        
        reg_conv.loc[reg_agg, r] = 1.0

reg_aggs = list(region_mapping.keys())
reg_aggs_wo_glo = [reg for reg in reg_aggs if reg != 'Global']

# Technology maps
tech_mapping = {'Grey':                         [technology_titles[0]],
                'Brown':                        [technology_titles[2]],
                'Blue':                         [technology_titles[idx] for idx in [1, 3]],
                'Turquoise':                    [technology_titles[4]],
                'Yellow':                       [technology_titles[idx] for idx in [5, 6, 7]],
                'Green':                        [technology_titles[idx] for idx in [8, 9, 10]],
                }

tech_conv = pd.DataFrame(0.0, index=tech_mapping.keys(), columns=technology_titles)
for tech_agg, techs_orig in tech_mapping.items():
    
    for t in techs_orig:
        
        tech_conv.loc[tech_agg, t] = 1.0

tech_col_map = {'Grey':                         'darkgray',
                'Brown':                        'sienna',
                'Blue':                         'blue',
                'Turquoise':                    'turquoise',
                'Yellow':                       'goldenrod',
                'Green':                        'green',
                }

# Scenarios
assumptions = ['Default', 'Optimistic', 'Pessimistic']
market_segments = ['Default', 'Mandated', 'Total']

scen_main_map = {'Baseline':                'S0',
                 'Carbon price':            'S1',
                 'EU CBAM':                 'S2',
                 'Mandate':                 'S3'}

scen_opt_map = {'Baseline (optimistic sensitivity)':                'S4',
                'Carbon price (optimistic sensitivity)':     'S5',
                'EU CBAM (optimistic sensitivity)':                 'S6',
                'Mandate (optimistic sensitivity)':          'S7'}

scen_pes_map = {'Baseline (pessimistic sensitivity)':                'S8',
                'Carbon price (pessimistic sensitivity)':     'S9',
                'EU CBAM (pessimistic sensitivity)':                 'S10',
                'Mandate (pessimistic sensitivity)':          'S11'}

scen_all_map = scen_main_map | scen_opt_map | scen_pes_map

# # %% Convert variables to new classifications

# vars_supply_maps = ['NH3SMLVL', 'NH3SMSHAR', 'NH3TCCout', 'NH3BILACOST', 
#                     'NH3TRANSPORTFUELCOST', 'NH3TRANSPORTEMISSIONCOST',
#                     'NH3SHIPPINGCOSTout']

# vars_currency_conversion = ['PRSC', 'EX', 'REX']

# vars_flows = ['NH3DEM', 'NH3PROD', 'NH3IMP', 'NH3EXP']

# vars_h2_techs = ['HYG1', 'WBWG', 'WGWG']

# HYLC

# NH3LC

# %% Functions to assist the calculation of price averages based on volumes

# --- Your aggregate -> original dict ---
agg_to_orig = region_mapping_wo_glo

# --- Your original axis order (exporters/importers) ---
original_order = region_titles

# --- Helpers to build mappings and membership matrix ---

def make_region_mapping_ordered(agg_to_orig: dict, original_order: list):
    """
    Build label and integer id mappings aligned to original_order, preserving the
    *insertion order* of agg_to_orig keys for group IDs.
    Returns:
        map_labels: length-N array of aggregate names aligned to original_order
        map_ids:    length-N int array (0..G-1) aligned to original_order
        agg_order:  list of aggregate names in the preserved order
    """
    # Invert mapping: orig -> aggregate
    orig_to_agg = {}
    for agg, codes in agg_to_orig.items():
        for c in codes:
            if c in orig_to_agg and orig_to_agg[c] != agg:
                raise ValueError(f"Code {c} assigned to two aggregates: {orig_to_agg[c]} and {agg}")
            orig_to_agg[c] = agg

    # Ensure all original codes are covered
    missing = [c for c in original_order if c not in orig_to_agg]
    if missing:
        raise ValueError(f"Missing codes in agg_to_orig: {missing}")

    # Preserve dictionary order for aggregates
    agg_order = list(agg_to_orig.keys())
    agg_index = {agg: i for i, agg in enumerate(agg_order)}

    # Map each original code (by order) to aggregate label and id
    map_labels = np.array([orig_to_agg[c] for c in original_order], dtype=object)
    map_ids = np.array([agg_index[a] for a in map_labels], dtype=int)

    return map_labels, map_ids, agg_order



def membership_matrix_from_ids(map_ids: np.ndarray, n_groups: int = None):
    """Create one-hot membership matrix G (N x G) from integer codes 0..G-1."""
    n = map_ids.shape[0]
    G = int(map_ids.max()) + 1 if n_groups is None else n_groups
    M = np.zeros((n, G), dtype=float)
    M[np.arange(n), map_ids] = 1.0
    return M


# --- Build exporter/importer mappings (same order on both axes) ---
map_exp_labels, map_exp_ids, agg_order = make_region_mapping_ordered(agg_to_orig, list(original_order))
map_imp_labels, map_imp_ids = map_exp_labels, map_exp_ids  # same axis order

Gexp = membership_matrix_from_ids(map_exp_ids,  n_groups=len(agg_order))  # shape (71, 12)
Gimp = membership_matrix_from_ids(map_imp_ids,  n_groups=len(agg_order))  # shape (71, 12)


# -------------------------------------------------------
# Core aggregator for a single market-year 2D slice (71x71)
# -------------------------------------------------------
def aggregate_slice_to_df(
    P2: np.ndarray,  # prices
    Q2: np.ndarray,  # quantities (weights)
    *,
    drop_missing_prices_from_weights: bool = True,
    exclude_same_country: bool = False,
    exclude_within_aggregate: bool = False,
    return_value_qty: bool = False
):
    """
    Aggregate a (71x71) slice to a (12x12) labeled DataFrame of unit-value prices.
    Options:
      - drop_missing_prices_from_weights: drop cells with NaN/inf prices from weights (recommended)
      - exclude_same_country: zero out i==j flows before aggregation
      - exclude_within_aggregate: zero out flows where exporter and importer belong to same aggregate
      - return_value_qty: also return DataFrames of summed values and quantities
    """
    P = np.array(P2, dtype=float, copy=True)
    Q = np.array(Q2, dtype=float, copy=True)
    Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)

    if exclude_same_country:
        np.fill_diagonal(Q, 0.0)

    if exclude_within_aggregate:
        same_block = map_exp_ids[:, None] == map_imp_ids[None, :]
        Q[same_block] = 0.0

    # Effective weights: drop cells with missing/invalid prices if requested
    price_ok = np.isfinite(P)
    Q_eff = np.where(price_ok, Q, 0.0) if drop_missing_prices_from_weights else Q
    V = np.where(price_ok, P * Q_eff, 0.0)

    # Block sums via membership matrices
    V_agg = Gexp.T @ V @ Gimp  # (12, 12)
    Q_agg = Gexp.T @ Q_eff @ Gimp
    P_agg = np.divide(V_agg, Q_agg, out=np.full_like(V_agg, np.nan), where=Q_agg > 0)

    # Labeled DataFrames in your preserved order
    idx = pd.Index(agg_order, name="Exporter (agg)")
    col = pd.Index(agg_order, name="Importer (agg)")

    P_df = pd.DataFrame(P_agg, index=idx, columns=col)
    if not return_value_qty:
        return P_df

    V_df = pd.DataFrame(V_agg, index=idx, columns=col)
    Q_df = pd.DataFrame(Q_agg, index=idx, columns=col)
    return P_df, V_df, Q_df



# %% Convert variables

var_conv = {}

for scen_name, scen in scen_all_map.items():
    
    msg = "Converting variables for the {} scenario".format(scen_name)
    
    var_conv[scen] = {}
    
    # H2 production by market segment (also relates to NH3)
    h2_prod_green = output_all[scen]['WGWG'][:, :, 0, :].sum(axis=1)        
    h2_prod_default = output_all[scen]['WBWG'][:, :, 0, :].sum(axis=1)
    h2_prod_total = h2_prod_green + h2_prod_default

    # Convert country-specific values from 2010 Euros to 2024 USD
    # Get price indices
    prsc_2024 = output_all[scen]['PRSC'][:, 0, 0, 2024-tl[0]-1]
    ex_2024 = output_all[scen]['EX'][:, 0, 0, 2024-tl[0]-1]
    
    # =============== Region conversion first =====================
    
    # Convert accounting data to new regional aggregation
    for var in ['NH3DEM', 'NH3PROD', 'NH3IMP', 'NH3EXP']:
        
        var_conv[scen][var] = {}
    
        # Default market
        var_conv[scen][var]['Default'] = pd.DataFrame(reg_conv.values[1:, :] @ output_all[scen][var][:, 1, 0, :],
                                                      index=reg_aggs_wo_glo,
                                                      columns=tl)
        
        # Mandated market
        var_conv[scen][var]['Mandated'] = pd.DataFrame(reg_conv.values[1:, :] @ output_all[scen][var][:, 0, 0, :],
                                                      index=reg_aggs_wo_glo,
                                                      columns=tl)
        
        # Total market
        var_conv[scen][var]['Total'] = var_conv[scen][var]['Default'] + var_conv[scen][var]['Mandated']
        
    # Convert supply maps
    for var in ['NH3SMLVL']:
        
        var_conv[scen][var] = {}
        
        var_conv[scen][var]['Default'] = {}
        var_conv[scen][var]['Mandated'] = {}
        var_conv[scen][var]['Total'] = {}
        
        for y, year in enumerate(tl):
            
            # Default market
            double_conv = reg_conv.values[1:, :] @ output_all[scen][var][:, :, 1, y] @ reg_conv.T.values[:, 1:]
            var_conv[scen][var]['Default'][year] = pd.DataFrame(double_conv, 
                                                                index=reg_aggs_wo_glo,
                                                                columns=reg_aggs_wo_glo)
            
            # Mandated market
            double_conv = reg_conv.values[1:, :] @ output_all[scen][var][:, :, 0, y] @ reg_conv.T.values[:, 1:]
            var_conv[scen][var]['Mandated'][year] = pd.DataFrame(double_conv, 
                                                                index=reg_aggs_wo_glo,
                                                                columns=reg_aggs_wo_glo)
            
            # Total market
            var_conv[scen][var]['Total'][year] = var_conv[scen][var]['Default'][year] + var_conv[scen][var]['Mandated'][year] 
    
    # Convert H2 technology variables, region and technology conversion
    for var in ['HYG1', 'WBWG', 'WGWG', 'HYIY', 'HYIT', 'HYMT']:
        
        var_conv[scen][var] = {}
        
        for reg_agg, regs_orig in region_mapping.items():
            
            indices = [region_titles.index(reg) for reg in regs_orig]
            
            if var not in ['HYIY', 'HYIT']:
                # These variables are in units of kt H2
                var_conv[scen][var][reg_agg] = pd.DataFrame(tech_conv.values @ output_all[scen][var][indices, :, 0, :].sum(axis=0),
                                                            index=tech_mapping.keys(),
                                                            columns=tl)   
                # Convert to NH3 production
                var_conv[scen][var][reg_agg] /= 0.179
            else:

                var_conv[scen][var][reg_agg] = pd.DataFrame(tech_conv.values @ (output_all[scen][var][indices, :, 0, :]
                                                                                * 1.2 
                                                                                * 1.18).sum(axis=0),
                                                            index=tech_mapping.keys(),
                                                            columns=tl)
            
    for var in ['NH3LC',  'WPPR', 'NH3LCexclH2']:
        
        var_conv[scen][var] = {}
        
        # Mandated
        price_mandated = output_all[scen][var][:, 0, 0, :] * 1.2 * 1.18
        
        # Default
        price_default = output_all[scen][var][:, 1, 0, :] * 1.2 * 1.18
        
        # Total
        mandated_share = np.divide(h2_prod_green,
                                   h2_prod_total,
                                   where=h2_prod_total > 0.0,
                                  )
        
        # h2_prod_green/h2_prod_total
        mandated_share[np.isnan(mandated_share)] = 0.0
        price_average = mandated_share * price_mandated + (1-mandated_share) * price_default
        
        # Set up variables
        var_conv[scen][var]['Mandated']  = pd.DataFrame(0.0, index=region_mapping.keys(), columns=tl)
        var_conv[scen][var]['Default']  = pd.DataFrame(0.0, index=region_mapping.keys(), columns=tl)
        var_conv[scen][var]['Total']  = pd.DataFrame(0.0, index=region_mapping.keys(), columns=tl)
        
        for reg_agg in region_mapping.keys():
            
            mask = reg_conv.loc[reg_agg, :].values
            
            # Mandated
            var_conv[scen][var]['Mandated'].loc[reg_agg, :] = np.sum((price_mandated*h2_prod_green*mask[:, None]) /
                                                               np.sum(h2_prod_green*mask[:, None])
                                                               )
            
            # Default
            var_conv[scen][var]['Default'].loc[reg_agg, :] = np.sum((price_default*h2_prod_default*mask[:, None]) /
                                                               np.sum(h2_prod_default*mask[:, None])
                                                               )
            
            # Green
            var_conv[scen][var]['Total'].loc[reg_agg, :] = np.sum((price_average*h2_prod_total*mask[:, None]) /
                                                               np.sum(h2_prod_total*mask[:, None])
                                                               )

            

    for var in ['NH3CBAM', 'NH3DELIVCOST', 'NH3TCCout']:
        
        var_conv[scen][var] = {}
        
        var_conv[scen][var]['Default'] = {}
        var_conv[scen][var]['Mandated'] = {}
        var_conv[scen][var]['Total'] = {}
        
        
        # Convert prices to 2024 USD
        if var != 'NH3TCCout':
            
            # Individual markets
            bila_cost_mandated_conv = output_all[scen][var][:, :, 0, :] * 1.2 * 1.18
            bila_cost_default_conv = output_all[scen][var][:, :, 1, :] *  1.2 * 1.18
            
            # Combined
            numer = (bila_cost_mandated_conv * output_all[scen]['NH3SMLVL'][:, :, 0, :]
                     + bila_cost_default_conv * output_all[scen]['NH3SMLVL'][:, :, 1, :])
            denom = output_all[scen]['NH3SMLVL'][:, :, :, :].sum(axis=2)
            bila_cost_total_conv = np.divide(numer, denom, where=denom>0.0)
            
            # bila_cost_total_conv = np.divide((bila_cost_mandated_conv * output_all[scen]['NH3SMLVL'][:, :, 0, :]
            #                                   + bila_cost_default_conv * output_all[scen]['NH3SMLVL'][:, :, 1, :]),
            #                                  output_all[scen]['NH3SMLVL'][:, :, :, :].sum(axis=2),
            #                                  where=output_all[scen]['NH3SMLVL'][:, :, :, :].sum(axis=2))
                
        
        else:
            
            bila_cost_mandated_conv = output_all[scen][var][:, :, 0, :] * 1.2 * 1.18
            bila_cost_default_conv = output_all[scen][var][:, :, 0, :] * 1.2 * 1.18
            bila_cost_total_conv = output_all[scen][var][:, :, 0, :] * 1.2 * 1.18
            
        for y, year in enumerate(tl):
            
            # Mandated market
            prices = bila_cost_mandated_conv[:, :, y]
            volumes = output_all[scen]['NH3SMLVL'][:, :, 0, y]
            prices_agg, values_agg, quantities_agg = aggregate_slice_to_df(prices, volumes, return_value_qty=True)
            # P_agg_df, V_agg_df, Q_agg_df = aggregate_slice_to_df(P2, Q2, return_value_qty=True)
            var_conv[scen][var]['Mandated'][year] = prices_agg.copy()
            
            # Default market
            prices = bila_cost_default_conv[:, :, y]
            volumes = output_all[scen]['NH3SMLVL'][:, :, 1, y]
            prices_agg, values_agg, quantities_agg = aggregate_slice_to_df(prices, volumes, return_value_qty=True)
            # P_agg_df, V_agg_df, Q_agg_df = aggregate_slice_to_df(P2, Q2, return_value_qty=True)         
            var_conv[scen][var]['Default'][year] = prices_agg.copy()
           
            # Total market
            prices = bila_cost_total_conv[:, :, y]
            volumes = output_all[scen]['NH3SMLVL'][:, :, :, y].sum(axis=2)
            prices_agg, values_agg, quantities_agg = aggregate_slice_to_df(prices, volumes, return_value_qty=True)
            # P_agg_df, V_agg_df, Q_agg_df = aggregate_slice_to_df(P2, Q2, return_value_qty=True)
            var_conv[scen][var]['Total'][year] = prices_agg.copy()
            
    
    for var in ['HYLC']:
        
        var_conv[scen][var] = {}
        
        # Convert to 2024 USD
        var_conv_curr = output_all[scen][var][:, :, 0, :] * 1.2 * 1.18
        
        # Cost * volume
        cost_volumes = var_conv_curr * output_all[scen]['HYG1'][:, :, 0, :]
        
        for reg_agg, regs_orig in region_mapping.items():
            
            var_conv[scen][var][reg_agg] = pd.DataFrame(0.0, index=tech_mapping.keys(), columns=tl)
            
            indices = [region_titles.index(reg) for reg in regs_orig]
            
            # These variables are in units of kt H2
            cost_volumes_reg_tech = pd.DataFrame(tech_conv.values @ cost_volumes[indices, :, :].sum(axis=0),
                                                        index=tech_mapping.keys(),
                                                        columns=tl)
            
            # Divide by volume of aggregate 
            cost_volumes_reg_tech = cost_volumes_reg_tech.mask(cost_volumes_reg_tech>0.0,
                                                                other=cost_volumes_reg_tech/(var_conv[scen]['HYG1'][reg_agg] * 0.179))
            
            # Get unweighted average in case there is no production
            unweighted_average = pd.DataFrame(0.0, index=tech_mapping.keys(), columns=tl)
            for tech_agg, techs_orig in tech_mapping.items():
            
                tech_indices = [technology_titles.index(tech) for tech in techs_orig]
                
                reg_sum = var_conv_curr[indices, :, :].sum(axis=0)
                reg_tech_sum = reg_sum[tech_indices, :].sum(axis=0)
                unweighted_average.loc[tech_agg, :] = reg_tech_sum / (len(indices) + len(tech_indices))
            
            # Fil NaNs with unweighted averages
            cost_volumes_reg_tech = cost_volumes_reg_tech.fillna(unweighted_average)
            cost_volumes_reg_tech[np.isclose(cost_volumes_reg_tech, 0.0)] = unweighted_average
            
            var_conv[scen][var][reg_agg] = copy.deepcopy(cost_volumes_reg_tech)
            
            # Now divide cost * volume by aggregate volume to get the weighted
            # average cost
            # var_conv[scen][var][reg_agg] = var_conv[scen][var][reg_agg].where(
            #     (var_conv[scen]['HYG1'][reg_agg] * 0.179) > 0.0,
            #     other=var_conv[scen][var][reg_agg] / (var_conv[scen]['HYG1'][reg_agg] * 0.179))
            
    for var in ['HYEF', 'HYEFINDIRECT']:
        
        var_conv[scen][var] = {}
        
        # Default market
        tot_reg_emis = np.sum(output_all[scen][var][:, :, 0, :] * output_all[scen]["WBWG"][:, :, 0, :], axis=1)
        var_conv[scen][var]['Default'] = pd.DataFrame(reg_conv.values[1:, :] @ tot_reg_emis,
                                                      index=reg_aggs_wo_glo,
                                                      columns=tl)
        
        # Mandated market
        tot_reg_emis = np.sum(output_all[scen][var][:, :, 0, :] * output_all[scen]["WGWG"][:, :, 0, :], axis=1)
        var_conv[scen][var]['Mandated'] = pd.DataFrame(reg_conv.values[1:, :] @ tot_reg_emis,
                                                      index=reg_aggs_wo_glo,
                                                      columns=tl)
        
        # Total market
        var_conv[scen][var]['Total'] = var_conv[scen][var]['Default'] + var_conv[scen][var]['Mandated']   
        
    for var in ['NH3EFINDIRECT']:
        
        var_conv[scen][var] = {}
        
        # Default market
        tot_reg_emis = output_all[scen][var][:, 0, 0, :] * output_all[scen]['NH3PROD'][:, 1, 0, :]
        var_conv[scen][var]['Default'] = pd.DataFrame(reg_conv.values[1:, :] @ tot_reg_emis,
                                                      index=reg_aggs_wo_glo,
                                                      columns=tl)
        
        # Mandated market
        tot_reg_emis = output_all[scen][var][:, 0, 0, :] * output_all[scen]['NH3PROD'][:, 0, 0, :]
        var_conv[scen][var]['Mandated'] = pd.DataFrame(reg_conv.values[1:, :] @ tot_reg_emis,
                                                      index=reg_aggs_wo_glo,
                                                      columns=tl)
        
        # Total market
        var_conv[scen][var]['Total'] = var_conv[scen][var]['Default'] + var_conv[scen][var]['Mandated']  

    for var in ['NH3TRANSPORTEMISSIONFACTOR']:
        
        var_conv[scen][var] = {}
        
        var_conv[scen][var]['Default'] = {}
        var_conv[scen][var]['Mandated'] = {}
        var_conv[scen][var]['Total'] = {}
        
        for y, year in enumerate(tl):
        
            # Default market
            tot_bila_emis = output_all[scen][var][:, :, 0, :] * output_all[scen]['NH3SMLVL'][:, :, 1, :]
            double_conv = reg_conv.values[1:, :] @ tot_bila_emis[:, :, y] @ reg_conv.T.values[:, 1:]
            var_conv[scen][var]['Default'][year] = pd.DataFrame(double_conv, 
                                                                index=reg_aggs_wo_glo,
                                                                columns=reg_aggs_wo_glo)        
        
            # Mandated market
            tot_bila_emis = output_all[scen][var][:, :, 0, :] * output_all[scen]['NH3SMLVL'][:, :, 0, :]
            double_conv = reg_conv.values[1:, :] @ tot_bila_emis[:, :, y] @ reg_conv.T.values[:, 1:]
            var_conv[scen][var]['Mandated'][year] = pd.DataFrame(double_conv, 
                                                                index=reg_aggs_wo_glo,
                                                                columns=reg_aggs_wo_glo) 
            
            # Total market
            var_conv[scen][var]['Total'][year]  = var_conv[scen][var]['Default'][year]  + var_conv[scen][var]['Mandated'][year]  
                    

# %% Graph 1 - Production by scenario and region (group) 2023 and 2050

for assump in assumptions:
    
    for segment in market_segments:
    
        if assump == 'Default':
            
            scen_map = scen_main_map
            scen_ref = 'S0'
            
        elif assump == 'Optimistic':
            
            scen_map = scen_opt_map
            scen_ref = 'S4'
        
        elif assump == 'Pessimistic':
            
            scen_map = scen_pes_map
            scen_ref = 'S8'
            
        if segment == 'Default':
            
            prod_var = 'WBWG'
            
        elif segment == 'Mandated':
            
            prod_var = 'WGWG'
            
        else:
            
            prod_var = 'HYG1'
            
        # file path
        fp = os.path.join('Graphs', 'fig1_production_v{}_{}_{}.{}'.format(VERSION, segment, assump, FORMAT))
        
        # Figure params
        figsize = (7.5, 20)
        # # Create subplot    
        fig, axes = plt.subplots(nrows=int(len(reg_aggs_wo_glo)/3),
                                 ncols=3,
                                 figsize=figsize,
                                 sharex=True,
                                 sharey=True)
        
        axes_flat = axes.flatten()
        # axes_flat[-1].set_visible(False)
        
        col_names = ['2023']
        col_names += ['{} 2035'.format(scen) for scen in scen_main_map.keys()]
        col_names += ['{} 2050'.format(scen) for scen in scen_main_map.keys()]
        
        small_gap = 1
        large_gap = small_gap * 1.5
        bar_width = small_gap*0.8
        x_steps = np.asarray([large_gap, small_gap, small_gap, small_gap, large_gap, small_gap, small_gap, small_gap])
        x_positions = np.asarray([0.0] + list(x_steps.cumsum()))
        
        
        for r, reg in enumerate(reg_aggs_wo_glo):
            
            plot_data = pd.DataFrame(0.0, index=list(tech_mapping.keys()), columns=col_names)
            plot_data.loc[:, '2023'] = var_conv['S0'][prod_var][reg].loc[:, 2023]
            
            for scen, scen_short in scen_map.items(): 
                
                scen_out = scen.split(' (')[0]
                
                plot_data.loc[:, '{} 2035'.format(scen_out)] = var_conv[scen_short][prod_var][reg].loc[:, 2035]
                plot_data.loc[:, '{} 2050'.format(scen_out)] = var_conv[scen_short][prod_var][reg].loc[:, 2050]
                
            # Convert data to Mt NH3
            plot_data *= 0.001
            
            bottom = np.zeros(len(plot_data.columns))
            for tech, colour in tech_col_map.items():
                
                axes_flat[r].bar(x_positions, plot_data.loc[tech, :].values, width=bar_width, bottom=bottom,
                            color=colour, label=tech, edgecolor='white', linewidth=0.2)
                
                bottom += plot_data.loc[tech, :].values
                
        
            # X-axis setup
            axes_flat[r].set_xticks(x_positions)
            axes_flat[r].set_xticklabels(col_names, fontsize=8, rotation=90)
            axes_flat[r].set_xlim(x_positions[0] - 0.5, x_positions[-1] + bar_width + 0.5)
            
        
            # Labels
            axes_flat[r].set_title(reg)
            axes_flat[r].set_ylabel('Mt NH$_3$')
            # axes_flat[r].label_outer()
            
        
            # Optional: visual separators for the large gaps
            for i in [0, 4]:
                separator_x = x_positions[i] + large_gap/2
                axes_flat[r].axvline(separator_x, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        
            
        
        # Legend
        h1, l1 = axes_flat[0].get_legend_handles_labels()
        
        fig.legend(handles=h1,
                   labels=l1, 
                   loc='lower center',
                   bbox_to_anchor=(0.5, 0.05),
                   frameon=False,
                   borderaxespad=0.,
                   ncol=3,
                   title="Technologies",
                   fontsize=8)
        
        fig.subplots_adjust(hspace=0.5, wspace=0.22, right=0.97, bottom=0.22, left=0.1, top=0.95)
        
        
            
        plt.show()
        plt.savefig(fp)
    
    

# %% Graph 2 - Demand by scenario and region (group) 2023 and 2050



for assump in assumptions:
    
    for segment in market_segments:
    
        if assump == 'Default':
            
            scen_map = scen_main_map
            scen_ref = 'S0'
            
        elif assump == 'Optimistic':
            
            scen_map = scen_opt_map
            scen_ref = 'S4'
        
        elif assump == 'Pessimistic':
            
            scen_map = scen_pes_map
            scen_ref = 'S8'
        
            
        # file path
        fp = os.path.join('Graphs', 'fig2_demand_comp_v{}_{}_{}.{}'.format(VERSION, segment, assump, FORMAT))
        
        # Figure params
        figsize = (7.5, 20)
        # # Create subplot    
        fig, axes = plt.subplots(nrows=int(len(reg_aggs_wo_glo)/3),
                                 ncols=3,
                                 figsize=figsize,
                                 sharex=True,
                                 sharey=True)
        
        axes_flat = axes.flatten()
        # axes_flat[-1].set_visible(False)
        
        col_names = ['2023']
        col_names += ['{} 2035'.format(scen.split(' (')[0]) for scen in scen_map.keys()]
        col_names += ['{} 2050'.format(scen.split(' (')[0]) for scen in scen_map.keys()]
        
        row_names = ['Production', 'Imports', 'Exports']
        
        small_gap = 1
        large_gap = small_gap * 1.5
        bar_width = small_gap*0.8
        x_steps = np.asarray([large_gap, small_gap, small_gap, small_gap, large_gap, small_gap, small_gap, small_gap])
        x_positions = np.asarray([0.0] + list(x_steps.cumsum()))
        
        colour_map = dict(zip(row_names, ['teal', 'burlywood', 'tomato']))
        
        for r, reg in enumerate(reg_aggs_wo_glo):
            
            # Bar chart data
            plot_data = pd.DataFrame(0.0, index=row_names, columns=col_names)
            plot_data.loc['Production', '2023'] = var_conv[scen_ref]['NH3PROD'][segment].loc[reg, 2023]
            plot_data.loc['Exports', '2023'] = -var_conv[scen_ref]['NH3EXP'][segment].loc[reg, 2023]
            plot_data.loc['Imports', '2023'] = var_conv[scen_ref]['NH3IMP'][segment].loc[reg, 2023]
            
            plot_demand = pd.Series(0.0, index=col_names)
            plot_demand.loc['2023'] = var_conv[scen_ref]['NH3DEM'][segment].loc[reg, 2023]
            
            for scen, scen_short in scen_map.items(): 
                
                scen_out = scen.split(' (')[0]
                
                plot_data.loc['Production', '{} 2035'.format(scen_out)] = var_conv[scen_short]['NH3PROD'][segment].loc[reg, 2035]
                plot_data.loc['Exports', '{} 2035'.format(scen_out)] = -var_conv[scen_short]['NH3EXP'][segment].loc[reg, 2035]
                plot_data.loc['Imports', '{} 2035'.format(scen_out)] = var_conv[scen_short]['NH3IMP'][segment].loc[reg, 2035]
                
                plot_data.loc['Production', '{} 2050'.format(scen_out)] = var_conv[scen_short]['NH3PROD'][segment].loc[reg, 2050]
                plot_data.loc['Exports', '{} 2050'.format(scen_out)] = -var_conv[scen_short]['NH3EXP'][segment].loc[reg, 2050]
                plot_data.loc['Imports', '{} 2050'.format(scen_out)] = var_conv[scen_short]['NH3IMP'][segment].loc[reg, 2050]
                
                plot_demand.loc['{} 2035'.format(scen_out)] = var_conv[scen_short]['NH3DEM'][segment].loc[reg, 2035]
                plot_demand.loc['{} 2050'.format(scen_out)] = var_conv[scen_short]['NH3DEM'][segment].loc[reg, 2050]
                
               
            # Convert data to Mt NH3
            plot_data *= 0.001
            plot_demand *= 0.001
            
            bottom = np.zeros(len(plot_data.columns))
            for var, colour in colour_map.items():
                
                if var != 'Exports':
                
                    axes_flat[r].bar(x_positions, plot_data.loc[var, :].values, width=bar_width, bottom=bottom,
                                color=colour, label=var, edgecolor='white', linewidth=0.2)
                    
                    bottom += plot_data.loc[var, :].values
                    
                else:
                    
                    axes_flat[r].bar(x_positions, plot_data.loc[var, :].values, width=bar_width, bottom=np.zeros(len(plot_data.columns)),
                                color=colour, label=var, edgecolor='white', linewidth=0.2)   
                    
            # Add demand
            axes_flat[r].scatter(x_positions, plot_demand.values, color='black', label='Demand')
                    
            # axes_flat[r].scatter(x_positions, scatter_y, s=60, color='#2F4B7C', zorder=3, label='Scatter metric')
        
            # X-axis setup
            axes_flat[r].set_xticks(x_positions)
            axes_flat[r].set_xticklabels(col_names, fontsize=8, rotation=90)
            axes_flat[r].set_xlim(x_positions[0] - 0.5, x_positions[-1] + bar_width + 0.5)
            
        
            # Labels
            axes_flat[r].set_title(reg)
            axes_flat[r].set_ylabel('Mt NH$_3$')
            # axes_flat[r].label_outer()
            
        
            # axes_flat[r].yaxis.set_minor_locator(mticker.FixedLocator([0]))
            # axes_flat[r].grid(which='minor', axis='y', linestyle='--', linewidth=1.2, color='#666666', alpha=0.8, zorder=1)
        
        
            # axes_flat[r].grid(which='major', axis='y', color='#DDDDDD', linewidth=0.8, linestyle='-')
            axes_flat[r].axhline(0, color='#666666', linewidth=1.2, linestyle='--', alpha=0.9, zorder=1)
        
            
            # Optional: visual separators for the large gaps
            for i in [0, 4]:
                separator_x = x_positions[i] + large_gap/2
                axes_flat[r].axvline(separator_x, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
            
        
        # Legend
        h1, l1 = axes_flat[0].get_legend_handles_labels()
        h1_adj = h1[1:] + [h1[0]]
        l1_adj = l1[1:] + [l1[0]]
        
        fig.legend(handles=h1_adj,
                   labels=l1_adj, 
                   loc='lower center',
                   bbox_to_anchor=(0.5, 0.08),
                   frameon=False,
                   borderaxespad=0.,
                   ncol=4,
                   title="NH$_3$ flow",
                   fontsize=8)
        
        fig.subplots_adjust(hspace=0.5, wspace=0.22, right=0.97, bottom=0.22, left=0.1, top=0.95)
        
        
            
        plt.show()
        plt.savefig(fp)

# %% Graph 3 - Sankey Diagrams of trade flows 2023, 2035, 2050 each scenario

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kaleido



for assump in assumptions:
    
    for segment in market_segments:
    
        if assump == 'Default':
            
            scen_map = scen_main_map
            scen_ref = 'S0'
            
        elif assump == 'Optimistic':
            
            scen_map = scen_opt_map
            scen_ref = 'S4'
        
        elif assump == 'Pessimistic':
            
            scen_map = scen_pes_map
            scen_ref = 'S8'
    
        # file path
        fp = os.path.join('Graphs', 'fig3_sankey_diagram_of_flows_v{}_{}_{}.{}'.format(VERSION, segment, assump, FORMAT))
        
        
        # ----------------------------
        # Configuration
        # ----------------------------
        num_countries = len(reg_aggs_wo_glo)
        years = [2030, 2040, 2050]
        scenarios = scen_map.keys()
        
        # Node labels (left: origins, right: targets)
        labels = reg_aggs_wo_glo + reg_aggs_wo_glo  
        
        # Colors: origins (blue), targets (orange)
        country_colours = [colour for reg, colour in region_col_map.items() if reg != 'Global']
        node_colors = country_colours + country_colours
        
        
        
        
        # Fixed horizontal positions: left for origins, right for targets
        x_orig = [0.01] * num_countries
        x_targ = [0.99] * num_countries
        
        
        
        
        # ----------------------------
        # Helper: convert matrix to Sankey link arrays
        # ----------------------------
        def matrix_to_links(M):
            source, target, value = [], [], []
            for i in range(M.shape[0]):
                for j in range(M.shape[0]):
                    v = M[i][j]
                    if v and v > 0:
                        source.append(i)              # origins indexed 0..13
                        target.append(M.shape[0] + j)          # targets indexed 14..27
                        value.append(v)
            return source, target, value
        
        
        
        # ----------------------------
        # Create subplots (3 rows x 4 cols)
        # ----------------------------
        subplot_titles = [f"{y} | {s}" for y in years for s in scenarios]
        fig = make_subplots(rows=len(years), cols=len(scenarios),
                            specs=[[{"type": "domain"} for _ in scenarios] for _ in years],
                            subplot_titles=subplot_titles,
                            horizontal_spacing=0.05,  # default ~0.2; smaller = closer
                            vertical_spacing=0.03     # default ~0.3; smaller = closer
        
                            )
        
        
        # Add Sankey traces per panel
        for r, year in enumerate(years, start=1):
            for c, scenario in enumerate(scenarios, start=1):
                
                scen_short = scen_map[scenario]
                supply_map = np.copy(var_conv[scen_short]['NH3SMLVL'][segment][year])
                
                production_shares = (var_conv[scen_short]['NH3SMLVL'][segment][year].sum(axis=1) / 
                                     var_conv[scen_short]['NH3SMLVL'][segment][year].sum().sum()
                                     )
                production_shares.iloc[0] = 0.02
                production_shares.iloc[1:] = 0.02 + production_shares.iloc[:-1].values
                production_shares /= production_shares.sum() /0.98
                
                demand_shares = (var_conv[scen_short]['NH3SMLVL'][segment][year].sum(axis=0) / 
                                 var_conv[scen_short]['NH3SMLVL'][segment][year].sum().sum()
                                 )
                demand_shares.iloc[0] = 0.02
                demand_shares.iloc[1:] = 0.02 + demand_shares.iloc[:-1].values
                demand_shares /= demand_shares.sum() / 0.98       
                
                
                
                # Fixed vertical order list (top-to-bottom). Example: O1..O14 and T1..T14
                # Define evenly spaced y-positions for origins and targets:
                y_orig = production_shares.cumsum().to_list()   # origins top→bottom
                y_targ = demand_shares.cumsum().to_list()   # targets top→bottom
                
                # Combine into full arrays aligned with your labels = origins + targets
                x_all = x_orig + x_targ
                y_all = y_orig + y_targ
                
                src, tgt, val = matrix_to_links(supply_map)
                
                # Quick adjustment
                # src = np.where(np.isclose(src, 0),
                #                0.0001,
                #                src)
                # tgt = np.where(np.isclose(tgt, 0),
                #                0.0001,
                #                tgt)    
                # src = np.where(np.isclose(src, num_countries),
                #                num_countries-0.0001,
                #                src)
                # tgt = np.where(np.isclose(tgt, num_countries),
                #                num_countries - 0.0001,
                #                tgt)            
                
        
                # Link colours
                link_colors = [country_colours[int(r)] for r in src]
                
                sankey = go.Sankey(
                    node=dict(
                        #pad=10,
                        thickness=15,
                        line=dict(color="black", width=0.5),
                        label=labels,
                        color=node_colors,
                        # x=x_all,                 # fixed positions
                        # y=y_all,                 # fixed positions
                    ),
                    link=dict(
                        source=src,
                        target=tgt,
                        value=val,
                        color=link_colors
                    ),
                    hoverlabel=dict(bgcolor="white")
                )
                fig.add_trace(sankey, row=r, col=c)
        
        fig.update_layout(
            font_size=10,
            width=1600,
            height=2000,
            margin=dict(l=3, r=3, t=30, b=10)
        )
        
        
        # Save outputs
        # fig.write_html("sankey_subplots_3x4.html")
        fig.write_image("sankey_subplots_3x4.png")
        fig.write_image(fp)
        
        
        fig.show()

# %% Graph 4 - Delivery costs from top 5 exporters to top 5 imports - v1


# main_importers = ['China', 'India', 'MENA', 'EU27+UK', 'USA']
main_importers = list(var_conv['S0']["NH3IMP"]['Default'][2050].nlargest(6).index)
if "RoW" in main_importers:
    main_importers = [reg2 for reg2 in main_importers if reg2 != 'RoW']
elif reg in main_importers:
    main_importers = main_importers[:5]

for assump in assumptions:
    
    if assump == 'Default':
        
        scen_map = scen_main_map
        scen_ref = 'S0'
        
    elif assump == 'Optimistic':
        
        scen_map = scen_opt_map
        scen_ref = 'S4'
    
    elif assump == 'Pessimistic':
        
        scen_map = scen_pes_map
        scen_ref = 'S8'
        
    # file path
    fp = os.path.join('Graphs', 'fig4_delivery_costs_top5_demanders_v{}_{}.{}'.format(VERSION, assump, FORMAT))
    
    # Get the top 5 exporters in each scenario by 2050
    
    top_exporters = pd.DataFrame(index=["Top {}".format(i) for i in range(1,11)], columns=scen_map.keys())
    
    for scen, scen_short in scen_map.items():
        
        top_exporters.loc[:, scen] = list(var_conv[scen_short]["NH3EXP"]['Total'].loc[:, 2050].nlargest(10).index)
        
    selected_top_exporters = ['Saudi Arabia', 'China', 'MENA', 'EU + UK']
    
    # Figure params
    figsize = (7.5, 16)
    # # Create subplot    
    fig, axes = plt.subplots(nrows=len(scen_map.keys()) + 1,
                             ncols=len(main_importers),
                             figsize=figsize,
                             sharex=False,
                             sharey=True
                             )
    
    deliv_cost_comps = ['Haber Bosch', 'Hydrogen production', 'Transportation costs', 'CBAM penalty']
    
    colour_map = dict(zip(deliv_cost_comps, ['green', 'blue', 'purple', 'firebrick']))
    
    
    
    for r, reg in enumerate(main_importers):
        
        for s, scen in enumerate(scen_map.keys()):
            scen_short = scen_map[scen]
            
            axes2 = axes[s, r].twinx()
            
            segment = 'Default'
            # Determine the top 3 targets
            bila_trade_proxy = copy.deepcopy(var_conv[scen_short]["NH3SMLVL"][segment][2050])
            bila_trade_proxy *= np.ones((num_countries, num_countries)) - np.eye(num_countries)
            top3_sources = list(bila_trade_proxy.loc[:, reg].nlargest(5).index)
            if "RoW" in top3_sources:
                top3_sources = [reg2 for reg2 in top3_sources if reg2 != 'RoW']
            elif reg in top3_sources:
                top3_sources = [reg2 for reg2 in top3_sources if reg2 != reg]
                
            top3_sources = top3_sources[:3]
            
            top3_sources = [reg] + top3_sources
            labels = copy.deepcopy(top3_sources)
            labels[0] = "Self-consumption:\n{}".format(reg)
            
            
            # Organise data
            plot_data = pd.DataFrame(0.0, index=deliv_cost_comps, columns=top3_sources)
            imp_share = pd.Series(0.0, index=top3_sources)
        
            for rt, reg_source in enumerate(top3_sources):
                
                
                plot_data.loc['Hydrogen production', reg_source] = var_conv[scen_short]["WPPR"][segment][2050].loc[reg_source] * 0.179 *1000
                plot_data.loc['Haber Bosch', reg_source] = var_conv[scen_short]["NH3LC"][segment][2050].loc[reg_source] - plot_data.loc['Hydrogen production', reg_source]
                plot_data.loc['Transportation costs', reg_source] = var_conv[scen_short]["NH3TCCout"][segment][2050].loc[reg_source, reg]
                plot_data.loc['CBAM penalty', reg_source] = var_conv[scen_short]["NH3CBAM"][segment][2050].loc[reg_source, reg]
                
                tot_imp = var_conv[scen_short]["NH3DEM"]['Total'].loc[reg, 2050]
                imp_share.loc[reg_source] = var_conv[scen_short]["NH3SMLVL"][segment][2050].loc[reg_source, reg] / tot_imp * 100
                
                bottom = np.zeros(len(plot_data.columns))
                for var, colour in colour_map.items():
            
                    axes[s, r].bar(np.arange(len(top3_sources)), plot_data.loc[var, :].values, width=bar_width, bottom=bottom,
                                color=colour, label=var, edgecolor='white', linewidth=0.2)
                
                    bottom += plot_data.loc[var, :].values  
            
            # Plot secondary axis
            axes2.scatter(np.arange(len(top3_sources)), imp_share.values, color='black')
            axes2.set_ylim(0, 100)
            if r == len(main_importers)-1: 
                # axes2.set_yticks([np.arange(0, 101, step=25)])
                axes2.yaxis.set_visible(True)
                axes2.spines['right'].set_visible(True)
                axes2.set_ylabel("Supply share (%)")
            else:
                axes2.spines['right'].set_visible(False)
                axes2.yaxis.set_visible(False)
                
            if r == 0 :
                axes[s, r].yaxis.set_visible(True)
                axes[s, r].spines['left'].set_visible(True)
            else:
                axes[s, r].spines['left'].set_visible(False)
                axes[s, r].yaxis.set_visible(False)
    
            # X-axis setup
            axes[s, r].set_xticks(np.arange(len(top3_sources)))
            axes[s, r].set_xticklabels(top3_sources, fontsize=8, rotation=90)
            # axes_flat[r].set_xlim(x_positions[0] - 0.5, x_positions[-1] + bar_width + 0.5)
            
        
            # Labels
            if r ==0 : axes[s, r].set_ylabel("{}\n$\u2082\u2080\u2082\u2084/tNH\u2083".format(scen+ '\nDefault market'))
            if s ==0: axes[s, r].set_title("Target: {}".format(reg), fontsize=8)
            
            
        # ================== Add green market for the mandate scenario
        scen = list(scen_map.keys())[-1]
        scen_short = scen_map[scen]
        s = -1
        
        axes2 = axes[s, r].twinx()
        
        segment = 'Mandated'
        
        # Determine the top 3 targets
        bila_trade_proxy = copy.deepcopy(var_conv[scen_short]["NH3SMLVL"][segment][2050])
        bila_trade_proxy *= np.ones((num_countries, num_countries)) - np.eye(num_countries)
        top3_sources = list(bila_trade_proxy.loc[:, reg].nlargest(5).index)
        if "RoW" in top3_sources:
            top3_sources = [reg2 for reg2 in top3_sources if reg2 != 'RoW']
        elif reg in top3_sources:
            top3_sources = [reg2 for reg2 in top3_sources if reg2 != reg]
            
        top3_sources = top3_sources[:3]
        
        top3_sources = [reg] + top3_sources
        labels = copy.deepcopy(top3_sources)
        labels[0] = "Self-consumption:\n{}".format(reg)
        
        # Organise data
        plot_data_mandate = pd.DataFrame(0.0, index=deliv_cost_comps, columns=top3_sources)
        imp_share = pd.Series(0.0, index=top3_sources)
    
        for rt, reg_source in enumerate(top3_sources):
            
            
            plot_data_mandate.loc['Hydrogen production', reg_source] = var_conv[scen_short]["WPPR"][segment][2050].loc[reg_source] * 0.179 *1000
            plot_data_mandate.loc['Haber Bosch', reg_source] = var_conv[scen_short]["NH3LC"][segment][2050].loc[reg_source] - plot_data_mandate.loc['Hydrogen production', reg_source]
            plot_data_mandate.loc['Transportation costs', reg_source] = var_conv[scen_short]["NH3TCCout"][segment][2050].loc[reg_source, reg]
            plot_data_mandate.loc['CBAM penalty', reg_source] = var_conv[scen_short]["NH3CBAM"][segment][2050].loc[reg_source, reg]
            
            tot_imp = var_conv[scen_short]["NH3IMP"]['Total'].loc[reg, 2050]
            imp_share.loc[reg_source] = var_conv[scen_short]["NH3SMLVL"][segment][2050].loc[reg_source, reg] / tot_imp * 100
            
            bottom = np.zeros(len(plot_data.columns))
            for var, colour in colour_map.items():
        
                axes[s, r].bar(np.arange(len(top3_sources)), plot_data_mandate.loc[var, :].values, width=bar_width, bottom=bottom,
                            color=colour, label=var, edgecolor='white', linewidth=0.2)
            
                bottom += plot_data_mandate.loc[var, :].values     
                
        # Plot secondary axis
        axes2.scatter(np.arange(len(top3_sources)), imp_share.values, color='black')
        axes2.set_ylim(0, 100)
        if r == len(main_importers)-1: 
            # axes2.set_yticks([np.arange(0, 101, step=25)])
            axes2.yaxis.set_visible(True)
            axes2.spines['right'].set_visible(True)
            axes2.set_ylabel("Supply share (%)")
        else:
            axes2.spines['right'].set_visible(False)
            axes2.yaxis.set_visible(False)
            
        if r == 0 :
            axes[s, r].yaxis.set_visible(True)
            axes[s, r].spines['left'].set_visible(True)
        else:
            axes[s, r].spines['left'].set_visible(False)
            axes[s, r].yaxis.set_visible(False)

        # X-axis setup
        axes[s, r].set_xticks(np.arange(len(top3_sources)))
        axes[s, r].set_xticklabels(top3_sources, fontsize=8, rotation=90)
        # axes_flat[r].set_xlim(x_positions[0] - 0.5, x_positions[-1] + bar_width + 0.5)
    

        # Labels
        if r ==0 : axes[s, r].set_ylabel("{}\n$\u2082\u2080\u2082\u2084/tNH\u2083".format(scen + '\nMandated market'))
        # axes[s, r].set_title("Target: {}".format(reg), fontsize=8)
            
    
    # Legend
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    
    fig.legend(handles=h1[:len(deliv_cost_comps)],
                labels=l1[:len(deliv_cost_comps)], 
                loc='lower center',
                bbox_to_anchor=(0.5, 0.08),
                frameon=False,
                borderaxespad=0.,
                ncol=4,
                title="Cost categories",
                fontsize=8)
    
    fig.subplots_adjust(hspace=1.1, wspace=0.2, right=0.9, bottom=0.19, left=0.1, top=0.95)
    
    
        
    plt.show()
    plt.savefig(fp) 

# %% Graph 4b - Delivery costs from top 5 exporters to top 5 imports - v2

for assump in assumptions:
    
    if assump == 'Default':
        
        scen_map = scen_main_map
        scen_ref = 'S0'
        
    elif assump == 'Optimistic':
        
        scen_map = scen_opt_map
        scen_ref = 'S4'
    
    elif assump == 'Pessimistic':
        
        scen_map = scen_pes_map
        scen_ref = 'S8'
        
    # file path
    fp = os.path.join('Graphs', 'fig4v2_delivery_costs_top5_demanders_v{}_{}.{}'.format(VERSION, assump, FORMAT))

    
    # Figure params
    figsize = (7.5, 13)
    # # Create subplot    
    fig, axes = plt.subplots(nrows=len(scen_map.keys()),
                             ncols=len(main_importers),
                             figsize=figsize,
                             sharex=False,
                             sharey=True
                             )
    
    deliv_cost_comps = ['Haber Bosch', 'Hydrogen production', 'Transportation costs', 'CBAM penalty']
    
    colour_map = dict(zip(deliv_cost_comps, ['green', 'blue', 'purple', 'firebrick']))
    
    
    
    for r, reg in enumerate(main_importers):
        
        for s, scen in enumerate(scen_map.keys()):
            scen_short = scen_map[scen]
            
            axes2 = axes[s, r].twinx()
            
            if not 'Mandate' in scen:
            
                segment = 'Default'
                # Determine the top 3 targets
                bila_trade_proxy = copy.deepcopy(var_conv[scen_short]["NH3SMLVL"][segment][2050])
                bila_trade_proxy *= np.ones((num_countries, num_countries)) - np.eye(num_countries)
                top3_sources = list(bila_trade_proxy.loc[:, reg].nlargest(5).index)
                if "RoW" in top3_sources:
                    top3_sources = [reg2 for reg2 in top3_sources if reg2 != 'RoW']
                elif reg in top3_sources:
                    top3_sources = [reg2 for reg2 in top3_sources if reg2 != reg]
                    
                top3_sources = top3_sources[:3]
                
                top3_sources = [reg] + top3_sources
                labels = copy.deepcopy(top3_sources)
                labels[0] = "Self-consumption:\n{}".format(reg)
                
                
                # Organise data
                plot_data = pd.DataFrame(0.0, index=deliv_cost_comps, columns=top3_sources)
                imp_share = pd.Series(0.0, index=top3_sources)
            
                for rt, reg_source in enumerate(top3_sources):
                    
                    
                    plot_data.loc['Hydrogen production', reg_source] = var_conv[scen_short]["WPPR"][segment][2050].loc[reg_source] * 0.179 *1000
                    plot_data.loc['Haber Bosch', reg_source] = var_conv[scen_short]["NH3LC"][segment][2050].loc[reg_source] - plot_data.loc['Hydrogen production', reg_source]
                    plot_data.loc['Transportation costs', reg_source] = var_conv[scen_short]["NH3TCCout"][segment][2050].loc[reg_source, reg]
                    plot_data.loc['CBAM penalty', reg_source] = var_conv[scen_short]["NH3CBAM"][segment][2050].loc[reg_source, reg]
                    
                    tot_imp = var_conv[scen_short]["NH3DEM"]['Total'].loc[reg, 2050]
                    imp_share.loc[reg_source] = var_conv[scen_short]["NH3SMLVL"][segment][2050].loc[reg_source, reg] / tot_imp * 100
                    
                    bottom = np.zeros(len(plot_data.columns))
                    for var, colour in colour_map.items():
                
                        axes[s, r].bar(np.arange(len(top3_sources)), plot_data.loc[var, :].values, width=bar_width, bottom=bottom,
                                    color=colour, label=var, edgecolor='white', linewidth=0.2)
                    
                        bottom += plot_data.loc[var, :].values  
                
                # Plot secondary axis
                axes2.scatter(np.arange(len(top3_sources)), imp_share.values, color='black')
                axes2.set_ylim(0, 100)
                if r == len(main_importers)-1: 
                    # axes2.set_yticks([np.arange(0, 101, step=25)])
                    axes2.yaxis.set_visible(True)
                    axes2.spines['right'].set_visible(True)
                    axes2.set_ylabel("Supply share (%)")
                else:
                    axes2.spines['right'].set_visible(False)
                    axes2.yaxis.set_visible(False)
                    
                if r == 0 :
                    axes[s, r].yaxis.set_visible(True)
                    axes[s, r].spines['left'].set_visible(True)
                else:
                    axes[s, r].spines['left'].set_visible(False)
                    axes[s, r].yaxis.set_visible(False)
    
                    
                # else:
                    # axes2.set_yticks([])
        
                # X-axis setup
                axes[s, r].set_xticks(np.arange(len(top3_sources)))
                axes[s, r].set_xticklabels(top3_sources, fontsize=8, rotation=90)
                # axes_flat[r].set_xlim(x_positions[0] - 0.5, x_positions[-1] + bar_width + 0.5)
                
            
                # Labels
                if r ==0 : axes[s, r].set_ylabel("{}\n$\u2082\u2080\u2082\u2084/tNH\u2083".format(scen+ '\nDefault market'))
                if s ==0: axes[s, r].set_title("Target: {}".format(reg), fontsize=8)
                
            else:
                
                
                segment = 'Mandated'
                # Determine the top 3 targets
                bila_trade_proxy = copy.deepcopy(var_conv[scen_short]["NH3SMLVL"][segment][2050])
                bila_trade_proxy *= np.ones((num_countries, num_countries)) - np.eye(num_countries)
                top3_sources = list(bila_trade_proxy.loc[:, reg].nlargest(5).index)
                if "RoW" in top3_sources:
                    top3_sources = [reg2 for reg2 in top3_sources if reg2 != 'RoW']
                elif reg in top3_sources:
                    top3_sources = [reg2 for reg2 in top3_sources if reg2 != reg]
                    
                top3_sources = top3_sources[:3]
                
                top3_sources = [reg] + top3_sources
                labels = copy.deepcopy(top3_sources)
                labels[0] = "Self-consumption:\n{}".format(reg)
                
                
                # Organise data
                plot_data = pd.DataFrame(0.0, index=deliv_cost_comps, columns=top3_sources)
                imp_share = pd.Series(0.0, index=top3_sources)
            
                for rt, reg_source in enumerate(top3_sources):
                    
                    
                    plot_data.loc['Hydrogen production', reg_source] = var_conv[scen_short]["WPPR"][segment][2050].loc[reg_source] * 0.179 *1000
                    plot_data.loc['Haber Bosch', reg_source] = var_conv[scen_short]["NH3LC"][segment][2050].loc[reg_source] - plot_data.loc['Hydrogen production', reg_source]
                    plot_data.loc['Transportation costs', reg_source] = var_conv[scen_short]["NH3TCCout"][segment][2050].loc[reg_source, reg]
                    plot_data.loc['CBAM penalty', reg_source] = var_conv[scen_short]["NH3CBAM"][segment][2050].loc[reg_source, reg]
                    
                    tot_imp = var_conv[scen_short]["NH3DEM"]['Total'].loc[reg, 2050]
                    imp_share.loc[reg_source] = var_conv[scen_short]["NH3SMLVL"][segment][2050].loc[reg_source, reg] / tot_imp * 100
                    
                    bottom = np.zeros(len(plot_data.columns))
                    for var, colour in colour_map.items():
                
                        axes[s, r].bar(np.arange(len(top3_sources)), plot_data.loc[var, :].values, width=bar_width, bottom=bottom,
                                    color=colour, label=var, edgecolor='white', linewidth=0.2)
                    
                        bottom += plot_data.loc[var, :].values  
                
                # Plot secondary axis
                axes2.scatter(np.arange(len(top3_sources)), imp_share.values, color='black')
                axes2.set_ylim(0, 100)
                if r == len(main_importers)-1: 
                    # axes2.set_yticks([np.arange(0, 101, step=25)])
                    axes2.yaxis.set_visible(True)
                    axes2.spines['right'].set_visible(True)
                    axes2.set_ylabel("Supply share (%)")
                else:
                    axes2.spines['right'].set_visible(False)
                    axes2.yaxis.set_visible(False)
                    
                if r == 0 :
                    axes[s, r].yaxis.set_visible(True)
                    axes[s, r].spines['left'].set_visible(True)
                else:
                    axes[s, r].spines['left'].set_visible(False)
                    axes[s, r].yaxis.set_visible(False)
    
                    
                # else:
                    # axes2.set_yticks([])
        
                # X-axis setup
                axes[s, r].set_xticks(np.arange(len(top3_sources)))
                axes[s, r].set_xticklabels(top3_sources, fontsize=8, rotation=90)
                # axes_flat[r].set_xlim(x_positions[0] - 0.5, x_positions[-1] + bar_width + 0.5)
                
            
                # Labels
                if r ==0 : axes[s, r].set_ylabel("{}\n$\u2082\u2080\u2082\u2084/tNH\u2083".format(scen+ '\nMandated market'))
                if s ==0: axes[s, r].set_title("Target: {}".format(reg), fontsize=8)
        
        
    
    # Legend
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    
    fig.legend(handles=h1[:len(deliv_cost_comps)],
                labels=l1[:len(deliv_cost_comps)], 
                loc='lower center',
                bbox_to_anchor=(0.5, 0.08),
                frameon=False,
                borderaxespad=0.,
                ncol=4,
                title="Cost categories",
                fontsize=8)
    
    fig.subplots_adjust(hspace=1.0, wspace=0.2, right=0.9, bottom=0.2, left=0.1, top=0.95)
    
    
        
    plt.show()
    plt.savefig(fp) 

# %% Graph 5 - LCOH by technology and region

for assump in assumptions:
    
    if assump == 'Default':
        
        scen_map = scen_main_map
        scen_ref = 'S0'
        
    elif assump == 'Optimistic':
        
        scen_map = scen_opt_map
        scen_ref = 'S4'
    
    elif assump == 'Pessimistic':
        
        scen_map = scen_pes_map
        scen_ref = 'S8'

    # file path
    fp = os.path.join('Graphs', 'fig5_levelised_cost_v{}_{}.{}'.format(VERSION, assump, FORMAT))
    
    # Figure params
    figsize = (7.5, 20)
    # # Create subplot    
    fig, axes = plt.subplots(nrows=len(reg_aggs_wo_glo),
                             ncols=len(scen_map.keys()),
                             figsize=figsize,
                             sharex=True,
                             sharey='row')
    
    for scen_no, scen in enumerate(scen_map.keys()):
        
        scen_short = scen_map[scen]
        
        axes[0, scen_no].set_title(scen)
        
        for r, reg in enumerate(reg_aggs_wo_glo):
            
            # axes[r, 0].set_ylabel("{}\n{}{}".format(reg, "$", r"$^{2024}/\\text{kg } H_{2}$"))
            axes[r, 0].set_ylabel("{}\n$\u2082\u2080\u2082\u2084/kgH\u2082".format(reg))
            
            for tech in tech_col_map.keys():
            
                axes[r, scen_no].plot(np.asarray(np.arange(2023, 2051)),
                                      var_conv[scen_short]['HYLC'][reg].loc[tech, 2023:2050].T.values,
                                      color=tech_col_map[tech],
                                      label=tech)
                
            # axes[r, scen_no].set_ylim(0, 20)
            axes[r, scen_no].set_xlim(2023, 2050)
            
    # Legend
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    
    fig.legend(handles=h1,
                labels=l1, 
                loc='lower center',
                bbox_to_anchor=(0.5, 0.055),
                frameon=False,
                borderaxespad=0.,
                ncol=6,
                title="Technologies",
                fontsize=8)
    
    fig.subplots_adjust(hspace=0.15, wspace=0.0, right=0.97, bottom=0.13, left=0.1, top=0.95)
    
    plt.show()
    plt.savefig(fp) 

# %% Graph 6 - Investment

# from matplotlib.patches import Patch

for assump in assumptions:
    
    if assump == 'Default':
        
        scen_map = scen_main_map
        scen_ref = 'S0'
        
    elif assump == 'Optimistic':
        
        scen_map = scen_opt_map
        scen_ref = 'S4'
    
    elif assump == 'Pessimistic':
        
        scen_map = scen_pes_map
        scen_ref = 'S8'

    # file path
    fp = os.path.join('Graphs', 'fig6_cumulative_investment_v{}_{}.{}'.format(VERSION, assump, FORMAT))
    
    # Figure params
    figsize = (7.5, 20)
    # # Create subplot    
    fig, axes = plt.subplots(nrows=len(scen_map.keys()),
                             ncols=3,
                             figsize=figsize,
                             # sharex=True,
                             # sharey='row'
                             )
    
    region_col_map_wo_glo = {reg: colour for reg, colour in region_col_map.items() if reg != 'Global'}
        
    for scen_no, scen in enumerate(scen_map.keys()):
        
        # Set titles
        if scen_no == 0:
            axes[scen_no, 0].set_title('Haber-Bosch capacity\nby region')
            axes[scen_no, 1].set_title('H2 capacity\nby region')
            axes[scen_no, 2].set_title('H2 capacity\nby technology')
            
        # Set labels
        axes[scen_no, 0].set_ylabel("{}\n\n\n\n".format(scen))
        
        scen_short = scen_map[scen]
        
        # Set a centre circle to make each pie chart a doughnut plot
        
        # centre_circle = axes[scen_no, 1].Circle((0, 0), 0.70, fc='white')
        # centre_circle = axes[scen_no, 2].Circle((0, 0), 0.70, fc='white')
            
        # HB investment by region
        hb_inv_by_reg = pd.Series(0.0, index=reg_aggs_wo_glo)
        # Base it on NH3 Production, assume 85% capacity factor
        for reg in reg_aggs_wo_glo:
            
            # Capacity is estimated using a 85% capacity factor on production
            capacity_estimate = var_conv[scen_short]['NH3PROD']['Total'] / 0.85
            # Get annual differences
            cap_diff = capacity_estimate.diff(axis=1)
            # Remove negative values
            cap_diff[cap_diff<0.0] = 0.0
            # Estimate replacement of depreciated capacity
            # cap_dprc = capacity_estimate * (1/30)
            # Now estimate investment using IEA's CAPEX number (770 USD/tNH3)
            # [kt Nh3] * [USD/tNH3] * [t NH3/ktNH3] * [bUSD/USD]
            annual_investment = cap_diff * 770 * 1e3 * 1e-9
            hb_inv_by_reg = annual_investment.loc[:, 2025:2050].cumsum(axis=1).loc[:, 2050]
        
            
        # plot chart
        explode  = 0.05 * np.ones((len(reg_aggs_wo_glo)))
        axes[scen_no, 0].pie(hb_inv_by_reg.values, 
                             colors=region_col_map_wo_glo.values(),
                              labels=region_col_map_wo_glo.keys(),
                              labeldistance=None,
                             autopct='%1.1f%%', pctdistance=1.33,
                             explode=explode,
                             textprops={'fontsize': 8})
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axes[scen_no, 0].add_artist(centre_circle)
        axes[scen_no, 0].text(
            -0.3, 0, "{}\nbillion".format(hb_inv_by_reg.sum().round(0)),
            va='center', fontsize=10, fontweight="bold", color='black')                             
        
        # H2 investment by region
        h2_inv_by_reg = pd.Series(0.0, index=reg_aggs_wo_glo)
        for reg in reg_aggs_wo_glo:
            h2_inv_by_reg.loc[reg] = var_conv[scen_short]['HYIT'][reg].loc[:, 2025:2050].cumsum(axis=1).loc[:, 2050].sum() * 0.001       
            
        # plot chart
        explode  = 0.05 * np.ones((len(reg_aggs_wo_glo)))
        axes[scen_no, 1].pie(h2_inv_by_reg.values, 
                             colors=region_col_map_wo_glo.values(),
                               labels=region_col_map_wo_glo.keys(),
                               labeldistance=None,
                             autopct='%1.1f%%', pctdistance=1.33,
                             explode=explode,
                             textprops={'fontsize': 8})
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axes[scen_no, 1].add_artist(centre_circle)        
        axes[scen_no, 1].text(
            -0.3, 0, "{}\nbillion".format(h2_inv_by_reg.sum().round(0)),
            va='center', fontsize=10, fontweight="bold", color='black')        
        
        
        # H2 investment by tech group
        h2_inv_by_tech = var_conv[scen_short]['HYIT']['Global'].loc[:, 2025:2050].cumsum(axis=1).loc[:, 2050]* 0.001 
    
            
        # plot chart
        explode  = 0.05 * np.ones((len(tech_col_map.keys())))
        axes[scen_no, 2].pie(h2_inv_by_tech.values, 
                             colors=tech_col_map.values(),
                               labels=tech_col_map.keys(),
                               labeldistance=None,
                             autopct='%1.1f%%', pctdistance=1.33,
                             explode=explode,
                             textprops={'fontsize': 8})
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axes[scen_no, 2].add_artist(centre_circle)
        axes[scen_no, 2].text(
            -0.3, 0, "{}\nbillion".format(h2_inv_by_tech.sum().round(0)),
            va='center', fontsize=10, fontweight="bold", color='black')
        
    # Legend
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    # l1 = list(region_col_map_wo_glo.keys())
    
    # h1 = [Patch(facecolor=c, edgecolor='white', label=l) for c, l in region_col_map_wo_glo.items()]
    
    fig.legend(handles=h1,
                labels=l1, 
                loc='lower center',
                bbox_to_anchor=(0.3, 0.15),
                frameon=False,
                borderaxespad=0.,
                ncol=3,
                title="Regions",
                fontsize=8)
    
    h2, l2 = axes[0, 2].get_legend_handles_labels()
    # l2 = list(tech_col_map.keys())
    fig.legend(handles=h2,
                labels=l2, 
                loc='lower center',
                bbox_to_anchor=(0.8, 0.15),
                frameon=False,
                borderaxespad=0.,
                ncol=2,
                title="Technologies",
                fontsize=8)
    
    fig.subplots_adjust(hspace=0.2, wspace=0.15, right=0.97, bottom=0.25, left=0.1, top=0.95)
     
    plt.show()
    plt.savefig(fp)         

# %% Graph 7 - Sensitivity around costs



# %% Graph 8 - Sensitivity around market share of production 

# %% Table 1 - Emissions (total, direct, indirect, transport)

emis_cats_map = {"Direct emissions":    "HYEF",  
                 "Indirect emissions":  "NH3EFINDIRECT",
                 "Transport emissions": "NH3TRANSPORTEMISSIONFACTOR",
                 "Total emissions":     None}


emissions_table = {}

year = 2050

for segment in market_segments:

    emissions_table[segment] = pd.DataFrame(0.0, index=['2025']+list(scen_all_map.keys()), columns=emis_cats_map.keys())
    
    emissions_table[segment].loc['2025', "Direct emissions"] =  np.sum(var_conv['S0']['HYEF'][segment][2025])
    emissions_table[segment].loc['2025', "Indirect emissions"] =  np.sum(var_conv['S0']['HYEFINDIRECT'][segment][2025])
    emissions_table[segment].loc['2025', "Indirect emissions"] +=  np.sum(var_conv['S0']['NH3EFINDIRECT'][segment][2025])
    emissions_table[segment].loc['2025', "Transport emissions"] = np.sum(var_conv['S0']['NH3TRANSPORTEMISSIONFACTOR'][segment][2025].sum())
    emissions_table[segment].loc['2025', "Total emissions"] = emissions_table[segment].loc['2025', :].sum()
    
    for scen_name, scen in scen_all_map.items():
        
        emissions_table[segment].loc[scen_name, "Direct emissions"] =  np.sum(var_conv[scen]['HYEF'][segment][year])
        emissions_table[segment].loc[scen_name, "Indirect emissions"] =  np.sum(var_conv[scen]['HYEFINDIRECT'][segment][year])
        emissions_table[segment].loc[scen_name, "Indirect emissions"] +=  np.sum(var_conv[scen]['NH3EFINDIRECT'][segment][year])
        emissions_table[segment].loc[scen_name, "Transport emissions"] = np.sum(var_conv[scen]['NH3TRANSPORTEMISSIONFACTOR'][segment][year].sum())
        emissions_table[segment].loc[scen_name, "Total emissions"] = emissions_table[segment].loc[scen_name, :].sum()
        
    # Convert to Mt CO2
    emissions_table[segment] *= 0.001
    
    emissions_table[segment] .to_csv("Emissions_in_{}_market.csv".format(segment))
    





# %% Annex graph x - Grid emission factors

# %% Annex graph x 



    
    

