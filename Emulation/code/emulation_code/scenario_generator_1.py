import numpy as np
import pandas as pd
from scipy.stats import qmc
from pyDOE import lhs

def generate_samples(sampling_method, Nscens, lower, upper, round_decimals=3):
    """
    Generate samples using the specified sampling method and scale them to the given range.
    
    Parameters:
    - sampling_method: function, the sampling method to use (e.g., lhs, qmc.LatinHypercube)
    - Nscens: int, number of scenarios
    - lower: float, lower bound of the range
    - upper: float, upper bound of the range
    - round_decimals: int, number of decimal places to round to
    
    Returns:
    - np.ndarray, the generated samples
    """
    samples = sampling_method(1, samples=Nscens)
    scaled_samples = lower + samples * (upper - lower)
    return np.round(scaled_samples, round_decimals)

def ambition_generator(regions, params, Nscens=1, round_decimals=3):
    """
    Generate ambition levels for each region using Latin Hypercube Sampling (LHS).
    
    Parameters:
    - regions: list of str, regions for ambition generation
    - params: list of str, parameters for each region
    - Nscens: int, number of scenarios
    - round_decimals: int, number of decimal places to round to
    
    Returns:
    - pd.DataFrame, DataFrame containing the generated ambition levels
    """
    
    ambition_df = pd.DataFrame()
    
    # Extract policies on the second level
    policies = [policy for policy_type in params.values() for policy in policy_type.keys()]
    sampler = qmc.LatinHypercube(d=len(policies))

    for region in regions:
        values = sampler.random(n=Nscens)
        values = np.round(values, round_decimals)
        region_df = pd.DataFrame(values, columns=[f"{region}_{policy}" for policy in policies])
        ambition_df = pd.concat([ambition_df, region_df], axis=1)
    return ambition_df

def uncertainty_generator(ranges, Nscens=1, sampling_method=lhs, round_decimals=3):
    """
    Generate uncertainty scenarios using Latin Hypercube Sampling (LHS).
    
    Parameters:
    - ranges: dict, dictionary of variable ranges with variable names as keys and (lower, upper) tuples as values
    - Nscens: int, number of scenarios
    - sampling_method: function, the sampling method to use (default is lhs)
    - round_decimals: int, number of decimal places to round to
    
    Returns:
    - pd.DataFrame, DataFrame containing the generated scenarios
    """
    uncertainty_df = pd.DataFrame()
    for var, (lower, upper) in ranges.items():
        samples = generate_samples(sampling_method, Nscens, lower, upper, round_decimals)
        uncertainty_df[var] = samples.flatten()
    return uncertainty_df

def scen_generator(regions, params, Nscens, scen_code, ranges, round_decimals=2, output_path=None):
    """
    Combines ambition levels and uncertainty levels into comprehensive scenarios and saves to a CSV file.
    
    Parameters:
    - regions: list of str, regions for ambition generation
    - params: list of str, parameters for each region
    - Nscens: int, number of scenarios
    - scen_code: str, scenario identifier prefix
    - ranges: dict, ranges for uncertainty generation
    - round_decimals: int, number of decimal places to round to
    - output_path: str, path to save the generated scenarios CSV file
    
    Returns:
    - pd.DataFrame, DataFrame containing the combined scenario data
    """
    ambition_df = ambition_generator(regions, params, Nscens, round_decimals)
    uncertainty_df = uncertainty_generator(ranges, Nscens, round_decimals=round_decimals)
    combined_df = pd.concat([ambition_df, uncertainty_df], axis=1)
    combined_df['ID'] = [f"{scen_code}_{i+1}" for i in range(Nscens)]
    
    # Reorder columns to make 'ID' the first column
    cols = ['ID'] + [col for col in combined_df.columns if col != 'ID']
    combined_df = combined_df[cols]
    
    if output_path:
        combined_df.to_csv(output_path, index=False)
    
    return combined_df