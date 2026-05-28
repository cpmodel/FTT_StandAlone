"""
<<<<<<< HEAD
Femke Nijsse, Rosie Hayward and Ian Burton. Created 2021, finished March 2025
=======
Femke Nijsse, Rosie Hayward and Ian Burton. 
>>>>>>> origin/main

A simulated annealing algorithm for finding the gamma values used in FTT simulations.

This algorithm should find the values of gamma such that the diffusion of shares
<<<<<<< HEAD
is constant across the boundary between the historical and simulated period.

"""

# Reminder
# Code now crashes unless you have multiple models selected in settings. To do with batteries? Make that code more robust.


=======
is smooth across the boundary between the historical and simulated period.

For the power sector, MWKA (exogenous capacity) is automatically set to -1 during calibration so that
exogenous capacity constraints do not interfere with the gamma optimisation.

The data gets saved into the Inputs folder, and needs to be manually copied into the Masterfiles (to be further automated when data structures are finalised.)

"""

>>>>>>> origin/main
# %%
# Third party imports
import numpy as np
import os
<<<<<<< HEAD


os.chdir("C:\\Users\Work profile\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone")
#os.chdir("C:\\Users\\fjmn202\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone")


from SourceCode import model_class
# %%

=======
from pathlib import Path

# Go two levels up: Support → SourceCode → repo root
repo_root = Path(__file__).resolve().parents[2]
os.chdir(repo_root)

from SourceCode import model_class


# %%

def get_model_maps(model):
    """Return a small dict of commonly used model title mappings and cache it on the model.

    The function is intentionally simple: it builds the mappings once and stores them on
    the model as ``_cached_maps`` so repeated calls are cheap.
    """
    if hasattr(model, '_cached_maps') and isinstance(model._cached_maps, dict):
        return model._cached_maps

    maps = {}
    ms = model.titles.get('Models_short', [])
    maps['shares'] = dict(zip(ms, model.titles.get('shares_var', [])))
    maps['costs'] = dict(zip(ms, model.titles.get('Cost_var', [])))
    maps['gamma_inds'] = dict(zip(ms, model.titles.get('Gamma_ind', [])))
    maps['histend'] = dict(zip(ms, model.titles.get('histend_var', [])))
    maps['gamma_value'] = dict(zip(ms, model.titles.get('Gamma_Value', [])))
    maps['techs'] = dict(zip(ms, model.titles.get('tech_var', [])))

    model._cached_maps = maps
    return maps


>>>>>>> origin/main
def set_timeline(model, modules_to_assess):
    "Run the models in parallel, so that the final year of the timeline is the max of histend"
    
    # Make sure the timeline in the settings file at least covers this range!
<<<<<<< HEAD
    histend_vars = dict(zip(model.titles['Models_short'], model.titles['Models_histend_var']))
    
    max_end = np.max([model.histend[histend_vars[module]] for module in modules_to_assess]) + 5
    
    return np.arange(2010, max_end)

def automation_init(model):
=======
    maps = get_model_maps(model)
    histend_vars = maps['histend']
    
    max_end = np.max([model.histend[histend_vars[mod]] for mod in modules_to_assess]) + 5
    
    return np.arange(2010, max_end)

def initialise_state(model):
>>>>>>> origin/main
    '''Initialise automation variables and run the model for the first time'''
    
    scenario = 'S0'

<<<<<<< HEAD
    # Identifying the variables needed for automation
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    cost_variables = dict(zip(model.titles['Models_short'], model.titles['Cost_var']))
    gamma_inds = dict(zip(model.titles['Models_short'], model.titles['Gamma_ind']))
    techs_vars = dict(zip(model.titles['Models_short'], model.titles['tech_var']))

    # Initialising container for automation variables
    automation_variables = {}
    
    # Establisting timeline for automation algorithm
=======
    # Create maps from model to specific variable name
    maps = get_model_maps(model)
    share_vars = maps['shares']
    cost_vars = maps['costs']
    gamma_inds = maps['gamma_inds']
    techs_vars = maps['techs']

    # Initialising dictionary for state variable
    state = {}
    
    # Establish timeline
>>>>>>> origin/main
    # Note: for power, we must start in 2010, otherwise things go wrong, in this model version. Why?
    model.timeline = set_timeline(model, modules_to_assess)
    
    # Set up various empty values 
<<<<<<< HEAD
    for module in modules_to_assess:
        print(f"Initialising for model {module}")
        
        N_regions = len(model.titles['RTI_short'])
        N_techs = len(model.titles[techs_vars[module]])
      
        # Automation variable list for this module
        automation_variables[module] = {}
            
        automation_variables[module][share_variables[module]] = (
                        np.zeros_like(model.input[scenario][share_variables[module]][:, :, :, :len(model.timeline)]) )
        
        automation_variables[module]['gamma'] = np.zeros((N_regions, N_techs, 1))
        automation_variables[module]['gamma'+'_LAG'] = np.zeros_like(automation_variables[module]['gamma'])

        # Create container for roc variables
        automation_variables[module]['roc_gradient'] = np.zeros((N_regions, N_techs))
        automation_variables[module]['roc_change'] = np.zeros((N_regions, N_techs))
        automation_variables[module]['roc_change' + '_LAG'] = np.zeros((N_regions, N_techs))

        automation_variables[module]['hist_share_avg'] = np.zeros((N_regions, N_techs))
        automation_variables[module]['score'] =  np.zeros((N_regions, N_techs))
        automation_variables[module]['score_LAG'] =  np.zeros((N_regions, N_techs))


    # Looping through years in automation timeline
    for year_index, year in enumerate(model.timeline):
        
        # Solving the model for each year
        model.variables, model.lags = model.solve_year(year, year_index, scenario)
        
        # Save initial values of interest
        for module in modules_to_assess:
            
            # # Resetting gamma values to zero (smarter to start from existing gamma values?)
            # model.input[scenario]['gamma'][:, :, 0, :] = np.zeros_like(
            #     model.input[scenario]['gamma'][:, :, 0, :])
            
            # Resetting gamma values to zero (smarter to start from existing gamma values?)
            model.input[scenario][cost_variables[module]][:, :, gamma_inds[module], :] = np.zeros_like(
                model.input[scenario][cost_variables[module]][:, :, gamma_inds[module], :])
            
            # Read in the historical and simulated shares
            var = share_variables[module]
            automation_variables[module][var][:, :, :, year_index] =  model.variables[var]
        
            # Compute the rate of change from the shares variable
            automation_variables[module]["rate of change"] = compute_roc(automation_variables, share_variables, module)
            automation_variables[module]["roc_change"] = roc_change(automation_variables, model, module)
            automation_variables[module]["score"] = get_scores(automation_variables, module)

    return automation_variables
# %%
def run_model(automation_variables, model):
    '''Run the model with new gamma values'''
    scenario = 'S0'

    # Identifying the model variables needed
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    cost_variables = dict(zip(model.titles['Models_short'], model.titles['Cost_var']))
    gamma_inds = dict(zip(model.titles['Models_short'], model.titles['Gamma_ind']))
  
    model.timeline = set_timeline(model, modules_to_assess)
    
    # Looping through all FTT modules to update gamma values
    for module in modules_to_assess:
            
        model.input[scenario][cost_variables[module]][:, :, gamma_inds[module], :] = (
            automation_variables[module]['gamma'][:, :, :])

    # Looping through years in timeline
    for year_index, year in enumerate(model.timeline):
    
        # Solving the model for each year
        model.variables, model.lags = model.solve_year(year, year_index, scenario)
        
        for module in modules_to_assess:
            
            # Save share variables for each module
            automation_variables[module][share_variables[module]][:, :, :, year_index] = (
                 model.variables[share_variables[module]] )
            # Define the ROC variable for each module
            automation_variables[module]["rate of change"] = compute_roc(automation_variables, share_variables, module)
            automation_variables[module]["roc_change"] = roc_change(automation_variables, model, module)
            automation_variables[module]["score"] = get_scores(automation_variables, module)
            
    return automation_variables


def compute_roc(automation_variables, share_variables, module):
    '''Compute the first differences (rate of change)'''
    roc =   (automation_variables[module][share_variables[module]][:, :, :, 1:]
           - automation_variables[module][share_variables[module]][:, :, :, :-1])
    
    return roc



# %%
def roc_change(automation_variables, model, module):
    '''
    Calculate the relative average historical rate of change (roc) and the simulated roc for each module
    Then, calculate the difference between them
    difference = (average_roc / average_share_sim) - (average_hist_roc / average_share_hist)
    
    Compared to a ratio of rate_of_change, this should be more stable for periods of low historical change.
    
    '''
    timeline = model.timeline
    
    # Identifying the variables needed
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    histend_vars = dict(zip(model.titles['Models_short'], model.titles['Models_histend_var']))
        
    mid = np.where(timeline == model.histend[histend_vars[module]])[0][0] + 1
    start, end, end1 = mid-4, mid+4, mid+4
    if end == 0: end1=None
    
    avg_share_hist = automation_variables[module][share_variables[module]][:, :, 0, start:mid].sum(axis=-1) / 4
    avg_share_sim  = automation_variables[module][share_variables[module]][:, :, 0, mid:end1  ].sum(axis=-1) / 4
    avg_growth_hist = automation_variables[module]["rate of change"][:, :, 0, start-1:mid-1].sum(axis=-1) / 4
    avg_growth_sim  = automation_variables[module]["rate of change"][:, :, 0, mid-1:end-1  ].sum(axis=-1) / 4  
    
    rel_change_hist = np.divide(avg_growth_hist, avg_share_hist, where=avg_share_hist != 0, out=np.zeros_like(avg_share_hist))
    rel_change_sim = np.divide(avg_growth_sim, avg_share_sim, where=avg_share_sim != 0, out=np.zeros_like(avg_share_sim))
    
    roc_change = np.where( (rel_change_hist == 0) | (rel_change_sim == 0), 0, rel_change_sim - rel_change_hist)
       
    return roc_change 



def adjust_gamma_values_simulated_annealing(automation_variables, model, module, Nyears, it):
    """Randomly choose delta gamma for each model version, using a standard deviation with 
    an adjustable standard deviation."""
       
    # Get model variables needed, this needs doing for every iter so not great
    gamma = automation_variables[module]['gamma'][:, :, 0]
    
    roc_change = automation_variables[module]["roc_change"]
    
    # Step size is a function of the number of iterations
    step_size = 0.2 * 0.98 ** it
=======
    for mod in modules_to_assess:
        print(f"Initialising for model {mod}")
        
        N_regions = len(model.titles['RTI_short'])
        N_techs = len(model.titles[techs_vars[mod]])
      
        # Initialise state variable dict per model
        state[mod] = {}
            
        state[mod][share_vars[mod]] = (
                        np.zeros_like(model.input[scenario][share_vars[mod]][:, :, :, :len(model.timeline)]) )
        
        state[mod]['gamma'] = np.zeros((N_regions, N_techs, 1))
        state[mod]['gamma_lag'] = np.zeros_like(state[mod]['gamma'])

        # Create container for rate of change (roc) vars
        state[mod]['roc_diff'] = np.zeros((N_regions, N_techs))

        state[mod]['hist_share_avg'] = np.zeros((N_regions, N_techs))
        state[mod]['score'] =  np.zeros((N_regions, N_techs))
        state[mod]['score_lag'] =  np.zeros((N_regions, N_techs))

    # Set all gamma values to zero
    for mod in modules_to_assess:
        
        model.input[scenario][cost_vars[mod]][:, :, gamma_inds[mod], :] = 0

    # Disable exogenous capacity for FTT-P: -1 means the constraint is inactive.
    # Without this, MWKA (e.g. exogenous capacity in the simulation period from incomplete data) 
    # fights the gamma optimisation and produces spuriously gamma values
    
    if 'FTT-P' in modules_to_assess:
        model.input[scenario]['MWKA'][:] = -1
          
    # Loop through timeline
    for year_index, year in enumerate(model.timeline):
        
        # Solve the model for each year
        model.variables, model.lags = model.solve_year(year, year_index, scenario)
            
        # Save initial values of interest
        for mod in modules_to_assess:
        
            # Read in the historical and simulated shares
            var = share_vars[mod]
            state[mod][var][:, :, :, year_index] =  model.variables[var]
        
            # Compute the rate of change and score from the shares variable
            state[mod]["rate of change"] = compute_roc(state, share_vars, mod)            
            state[mod]["roc_diff"], state[mod]["hist_share_avg"] = compute_roc_logit(state, model, mod)
            state[mod]["score"] = compute_scores(state, mod)

    return state
# %%
def run_model(state, model):
    '''Run the model with new gamma values'''
    scenario = 'S0'

    # Identify the model vars needed
    maps = get_model_maps(model)
    share_vars = maps['shares']
    cost_vars = maps['costs']
    gamma_inds = maps['gamma_inds']
  
    model.timeline = set_timeline(model, modules_to_assess)
    
    # Loop through all FTT mods to update gamma values
    for mod in modules_to_assess:
            
        model.input[scenario][cost_vars[mod]][:, :, gamma_inds[mod], :] = (
            state[mod]['gamma'][:, :, :])

    # Loop through timeline
    for year_index, year in enumerate(model.timeline):
    
        # Solve the model for each year
        model.variables, model.lags = model.solve_year(year, year_index, scenario)
        
        for mod in modules_to_assess:
            
            # Save share vars for each mod
            state[mod][share_vars[mod]][:, :, :, year_index] = (
                 model.variables[share_vars[mod]] )
    
    for mod in modules_to_assess:
        
        # Define the ROC variable for each model
        state[mod]["rate of change"] = compute_roc(state, share_vars, mod)        
        state[mod]['roc_diff'], _ = compute_roc_logit(state, model, mod)
        state[mod]['score'] = compute_scores(state, mod)
        
    return state


def compute_roc(state, share_vars, mod):
    '''Compute the first differences (rate of change)'''
    roc =   (state[mod][share_vars[mod]][:, :, :, 1:]
           - state[mod][share_vars[mod]][:, :, :, :-1])
    
    return roc

def compute_roc_logit(state, model, mod, epsilon=1e-6, sim_window=5, max_hist_window=12):
    """
    Computes the difference in logit-share trend slopes across the histend boundary.

    Uses OLS linear regression on logit(s) = log(s / (1-s)), which is the correct
    linearising transform for logistic (S-curve) diffusion:
      - Moderate-to-large shares (e.g. wind at 15-30%): log(s) flattens
        artificially; logit does not.
      - New/emerging technologies: logit(epsilon) is a large negative constant,
        so its OLS slope is zero (correctly: "not growing yet").

    Historical window is chosen adaptively (sim_window..max_hist_window years) by
    selecting the window size that minimises the standard error of the OLS slope.
    This handles high-variability technologies (e.g. wind, where interannual
    resource variation can make a short window unrepresentative of the real trend)
    without penalising fast-changing technologies that genuinely need a short window.

    The simulation window is fixed at sim_window years — FTT output is smooth so
    there is no benefit to adaptive windowing on that side.

    FTT-Tr uses TDA1, a per-country variable for the last year of historical
    share data, so the boundary index is looked up per region.
    """
    timeline = model.timeline
    maps = get_model_maps(model)
    share_vars = maps['shares']
    histend_vars = maps['histend']

    # Shape: (regions, techs, years)
    shares = state[mod][share_vars[mod]][:, :, 0, :]
    s_safe = np.clip(shares, epsilon, 1 - epsilon)
    logit_shares = np.log(s_safe / (1 - s_safe))

    N_regions, N_techs = shares.shape[:2]

    def ols_slope_fixed(ls_window):
        """OLS slope over last axis for a fixed window."""
        w = ls_window.shape[-1]
        t = np.arange(w, dtype=float) - (w - 1) / 2.0
        ss = np.dot(t, t)
        return np.sum((ls_window - ls_window.mean(axis=-1, keepdims=True)) * t, axis=-1) / ss

    def ols_slope_adaptive(ls_max_window, min_w):
        """
        Try trailing windows of size min_w..ls_max_window.shape[-1] and return
        the slope whose standard error is smallest. Vectorised over all leading
        dimensions (..., max_w).
        """
        max_w = ls_max_window.shape[-1]
        best_slope = None
        best_se    = None
        for w in range(min_w, max_w + 1):
            data = ls_max_window[..., -w:]
            t = np.arange(w, dtype=float) - (w - 1) / 2.0
            ss = np.dot(t, t)
            slope  = np.sum((data - data.mean(axis=-1, keepdims=True)) * t, axis=-1) / ss
            resid  = data - (slope[..., np.newaxis] * t + data.mean(axis=-1, keepdims=True))
            se     = np.sqrt(np.mean(resid ** 2, axis=-1) / ss)
            if best_slope is None:
                best_slope, best_se = slope, se
            else:
                pick_new   = se < best_se
                best_slope = np.where(pick_new, slope, best_slope)
                best_se    = np.where(pick_new, se,    best_se)
        return best_slope

    if mod == 'FTT-Tr':
        # Per-country boundary: TDA1 records the last historical year per region
        hist_end_years = model.input['S0']['TDA1'][:, 0, 0].astype(int)
        slope_hist     = np.zeros((N_regions, N_techs))
        slope_sim      = np.zeros((N_regions, N_techs))
        avg_share_hist = np.zeros((N_regions, N_techs))
        for r in range(N_regions):
            mid_r  = np.where(timeline == hist_end_years[r])[0][0]
            avail  = min(max_hist_window, mid_r)
            slope_hist[r]     = ols_slope_adaptive(logit_shares[r, :, mid_r-avail:mid_r], sim_window)
            slope_sim[r]      = ols_slope_fixed(logit_shares[r, :, mid_r:mid_r+sim_window])
            avg_share_hist[r] = np.mean(shares[r, :, mid_r-avail:mid_r+1], axis=-1)
    else:
        # Uniform boundary for all regions
        hist_end_year = model.histend[histend_vars[mod]]
        mid   = np.where(timeline == hist_end_year)[0][0]
        avail = min(max_hist_window, mid)
        slope_hist     = ols_slope_adaptive(logit_shares[:, :, mid-avail:mid], sim_window)
        slope_sim      = ols_slope_fixed(logit_shares[:, :, mid:mid+sim_window])
        avg_share_hist = np.mean(shares[:, :, mid-avail:mid+1], axis=-1)

    # The 'physics' difference: mismatch in trend growth rate across the boundary
    diff = slope_sim - slope_hist

    # Mask out techs that aren't present (avoid chasing noise)
    diff = np.where(avg_share_hist <= epsilon, 0, diff)

    return diff, avg_share_hist



def adjust_gamma_values_simulated_annealing(state, mod, it):
    """Randomly choose delta gamma for each model version, using a standard deviation with 
    an adjustable standard deviation."""
       
    # Get model vars needed, this needs doing for every iter so not great
    gamma = state[mod]['gamma'][:, :, 0]
    
    roc_diff = state[mod]["roc_diff"]
    
    # Step size is a function of the number of iterations
    step_size = starting_step_size * decay_rate_steps ** it
>>>>>>> origin/main
    
    # Generate a candidate step
    delta_gamma = np.random.normal(0, step_size, size=gamma.shape)

    # Propose a new gamma solution
    gamma = gamma + delta_gamma
    
    # Set gamma to zero if the roc is zero (usually as shares are zero)
<<<<<<< HEAD
    gamma = np.where(roc_change == 0, 0, gamma)
      
    gamma = np.clip(gamma, -1, 1)  # Ensure gamma values between -1 and 1
    
    # Tweak values towards zero, so we don't have highly negative or positive averages
    non_zero_total = (roc_change != 0).sum(axis=1)
    sum_gamma = gamma.sum(axis=1)
    country_averages = np.divide(sum_gamma, non_zero_total, out=np.zeros_like(sum_gamma),
                                 where=non_zero_total !=0)[:, np.newaxis] *  np.ones_like(gamma)
    gamma = np.where(roc_change !=0, gamma - 0.05 * country_averages, 0)
    
    gamma = gamma[:, :, np.newaxis]
    
    automation_variables[module]['gamma'] = np.copy(gamma)

    return automation_variables


def get_gamma_and_roc_change(automation_variables, model, module):
    
    roc_change_lag = automation_variables[module]["roc_change_LAG"]
    roc_change = automation_variables[module]["roc_change"]
    
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    gamma = automation_variables[module]['gamma'][:, :, 0]
    gamma_lag = automation_variables[module]['gamma'+'_LAG'][:, :, 0]
    
    return roc_change_lag, roc_change, gamma_variables, gamma, gamma_lag

def get_score_and_lagged_score(automation_variables, module):
    '''Get score and score lag to compare'''
    
    # Compute the regularised score for current and new solutions
    score = automation_variables[module]['score']
    score_lag = automation_variables[module]['score_LAG']
    
    return score, score_lag


def get_scores(automation_variables, module):
    '''Compute score, based on regulation which still need tweaking'''
    
    roc_change = automation_variables[module]['roc_change']
    gamma = automation_variables[module]['gamma'][:, :, 0]
    # Compute the regularised score for current and new solutions
    
    score = - np.abs(roc_change) - lambda_reg * gamma**2
    
    # Ensure that extreme values don't have an overly strong effect
    score = np.tanh(score)
    
    return score

def accept_or_reject_gamma_changes(automation_variables, model, module, Nyears, it, T0):
    
    '''
    Adjust the gamma values based on a regulated simulated annealing. 
=======
    gamma = np.where(roc_diff == 0, 0, gamma)
    gamma = np.clip(gamma, -1, 1)  # Ensure gamma values between -1 and 1
    
    # Tweak values towards zero, so we don't have highly negative or positive averages
    non_zero_total = (roc_diff != 0).sum(axis=1)
    sum_gamma = gamma.sum(axis=1)
    country_averages = np.divide(sum_gamma, non_zero_total, out=np.zeros_like(sum_gamma),
                                 where=non_zero_total !=0)[:, np.newaxis] * np.ones_like(gamma)
    
    gamma = np.where(roc_diff !=0, gamma - 0.05 * country_averages, 0)  
    state[mod]['gamma'] = gamma[:, :, np.newaxis]

    return state

def compute_scores(state, mod):
    """
    Computes a score that balances global and local fit,
    weighted by the importance (sqrt of share) of each technology.
    """
    roc_diff = state[mod]['roc_diff']
    gamma = state[mod]['gamma'][:, :, 0]
    shares = state[mod]['hist_share_avg']
    
    # 1. Importance Weighting (Square Root)
    # This ensures big techs are prioritized without ignoring small ones.
    weights = np.sqrt(shares + 0.01)
    
    # 2. Local Score
    # We squash the weighted error so one bad tech doesn't dominate everything
    weighted_error = -np.abs(roc_diff) * weights
    local_score = np.tanh(weighted_error) - lambda_reg * (gamma**2)
    
    # 3. Global Score (Region-level performance)
    # Encourages the algorithm to fix the 'worst' regions first
    active_mask = (shares > 1e-5)
    region_error = np.sum(np.abs(roc_diff) * weights * active_mask, axis=1)
    counts = np.maximum(np.count_nonzero(active_mask, axis=1), 1)
    global_score = -region_error / counts
    
    # Combine (using tanh again on global to keep scales comparable)
    combined = (1 - global_weight) * local_score + global_weight * np.tanh(global_score[:, np.newaxis])
    
    return combined


def accept_or_reject_gamma_changes(state, mod, it, T0):
    
    '''
    Adjust the gamma values based on regulated simulated annealing. 
>>>>>>> origin/main
    
    That is: penalise gamma values away from 0, accept some random changes
    to avoid getting in local minima as well'
    '''
<<<<<<< HEAD
    
    # Hyperparameter
    cooling_rate = 0.96
    
    # Cool down the temperature
    T = T0 * cooling_rate**it
    
    gamma = automation_variables[module]['gamma'][:,:,0]
    gamma_lag = automation_variables[module]['gamma_LAG'][:,:,0]
    score, score_lag = get_score_and_lagged_score(automation_variables, module)
=======
        
    # Cool down the temperature
    T = T0 * cooling_rate**it
    
    gamma = state[mod]['gamma'][:, :, 0]
    gamma_lag = state[mod]['gamma_lag'][:, :, 0]
    score = state[mod]['score']
    score_lag = state[mod]['score_lag']
>>>>>>> origin/main

    # Element-wise acceptance condition
    acceptance_mask = (score > score_lag) | (
        np.log(np.random.rand(*score.shape)) < (score - score_lag) / T
    )

    # Go back to old gamma/score values when values not accepted
    gamma[~acceptance_mask] = gamma_lag[~acceptance_mask]
<<<<<<< HEAD
    gamma = gamma[:, :, np.newaxis] # reshape format for whole period
    automation_variables[module]['gamma'] = np.copy(gamma)
    
    return automation_variables 

def set_initial_temperature(automation_variables, model, module):
    """ Set the initial temperature based on the typical change in score between
    the first two iterations
    """
    
    roc_change_lag, roc_change, gamma_variables, gamma, gamma_lag = (
        get_gamma_and_roc_change(automation_variables, model, module))
    
    score, score_lag = get_score_and_lagged_score(automation_variables, module)
    
    # Rule of thumb is to divide by 5. Ignoring non-zero values
    non_zero_mask = (score != 0) & (score_lag != 0)
    T0 = np.mean(np.abs(score[non_zero_mask] - score_lag[non_zero_mask])) / 5
    
    if T0 == 0 or np.isnan(T0):
        raise ValueError(f'T0 is {T0}. Is the module {module} included in the settings.ini file?')
    
    return T0 

def check_convergence(gamma, gamma_lag, module, it, max_it, already_converged):
=======
    state[mod]['gamma'] = gamma[:, :, np.newaxis]
    nonzero_mask = (score != 0) & (score_lag != 0)
    state[mod]['acceptance_rate'] = np.sum(acceptance_mask[nonzero_mask]) / acceptance_mask[nonzero_mask].size
    
    if it%50 == 0:
        print(f"At {it} it, the acceptance rate is {state[mod]['acceptance_rate']:.3f} at T {T:.5f}")
        
    return state 


def set_initial_temperature(state, model, mod):
    """ Set the initial temperature T0 based on the typical change in score
    between the first two iterations
    """
    
    score = state[mod]['score']
    score_lag = state[mod]['score_lag']
    
    # Rule of thumb is to divide by 5. Ignoring non-zero values
    non_zero_mask = (score != 0) & (score_lag != 0)
    
    T0 = np.mean(np.abs(score[non_zero_mask] - score_lag[non_zero_mask])) / 5
    
    if T0 == 0 or np.isnan(T0):
        raise ValueError(f'T0 is {T0}. Is the module {mod} included in the settings.ini file?')
    
    return T0 


def check_convergence(gamma, gamma_lag, mod, it, max_it, already_converged):
>>>>>>> origin/main
    '''Return true if gamma values no longer changing much, or if max_it is reached'''
    mask = (gamma != 0) & (gamma_lag != 0)
    converged = already_converged
    gamma_change = np.average(np.absolute(gamma[mask] - gamma_lag[mask]))
    
<<<<<<< HEAD
    if gamma_change < 0.015 and not already_converged:
        print(f"Convergence {module} reached at iter {it}, little change in gamma values last iteration")
        converged = True
        
    elif it >= max_it - 1:
        print(f"Maximum iterations reached at it {it}. Gammas {module} still changing by {gamma_change:.4f} on average")
        converged = True
    
    return converged
 
def get_median_score(variables, module):
    nonzero = variables[module]['score']!=0
    median_score = np.median(variables[module]['score'][nonzero])
=======
    if gamma_change < convergence_threshold and not already_converged:
        print(f"Convergence {mod} reached at iter {it}, little change in gamma values since last iteration")
        converged = True
        
    elif it >= max_it - 1:
        print(f"Maximum iterations reached at iter {it}. Gammas {mod} still changing by {gamma_change:.4f} on average")
        converged = True
    
    return converged

 
def get_median_score(scores):
    nonzero = (scores != 0)
    median_score = np.median(scores[nonzero])
>>>>>>> origin/main
    return median_score

# %%
def gamma_auto(model):
<<<<<<< HEAD
    '''Run the iterative script. First call automation_init and then
    run the simulated annealing script over all the models. 
    '''
    
    Nyears = len(model.timeline)

    # Initialising automation variables
    automation_variables = automation_init(model)
    run_variables = {}
    
    for module in modules_to_assess:
        run_variables[module] = {}
        
        N_regions = len(model.titles['RTI_short'])
        N_techs = automation_variables[module]['gamma'].shape[1]
        
        run_variables[module]['gamma'] = np.zeros((total_runs, N_regions, N_techs))
        run_variables[module]['score'] = np.zeros((total_runs, N_regions, N_techs))
    
    for run in range(total_runs):
        
 
        if run > 0:
            # Initialising automation variables
            automation_variables = automation_init(model) 
        
        # Initial roc_change values
        for module in modules_to_assess:
            automation_variables[module]['roc_change'] = roc_change(automation_variables, model, module)
=======
    '''Run the iterative script. First call initialise_state and then
    run the simulated annealing script over all the models. 
    '''
    
    # Initialise state
    state = initialise_state(model)
    run_history = {}
    
    for mod in modules_to_assess:
        run_history[mod] = {}
        
        N_regions = len(model.titles['RTI_short'])
        N_techs = state[mod]['gamma'].shape[1]
        
        run_history[mod]['gamma'] = np.zeros((total_runs, N_regions, N_techs))
        run_history[mod]['score'] = np.zeros((total_runs, N_regions, N_techs))
    
    for run in range(total_runs):
         
        if run > 0:
            # Initialising automation vars
            state = initialise_state(model) 
>>>>>>> origin/main
        
        # Break when all models have reached convergence
        convergence = [False] * len(modules_to_assess)
        
        # Computer after first iteration.
<<<<<<< HEAD
        T0 = [0.004] * len(modules_to_assess)
 
        # Iterative loop for gamma convergence
        #for iter in tqdm(range(5)): 
        for it in range(max_it):
            
            
            # First save the lagged variables, and find new gamma values to try
            for module in modules_to_assess:
                
                if it%25 == 0: # Print median score every 25 iterations
                    print(f"Median score {module} at {it}: {get_median_score(automation_variables, module):.3f}")    
                
                # Save previous gamma and roc values
                automation_variables[module]['gamma'+'_LAG'][:, :, 0] = (
                            automation_variables[module]['gamma'][:, :, 0])
                
                automation_variables[module]["roc_change" + '_LAG'][:, :] = (
                            automation_variables[module]["roc_change"][:, :] )
                
                automation_variables[module]["score" + '_LAG'][:, :] = (
                            automation_variables[module]["score"][:, :] )
                
                # Update gamma values semi-randomly
                automation_variables = adjust_gamma_values_simulated_annealing(
                                            automation_variables, model, module, Nyears, it)
                

            # Second: running the model, updating variables of interest
            automation_variables = run_model(automation_variables, model)
            
            
            
            # Third, save variables, accept and reject new gammas, and check convergence
            for n_module, module in enumerate(modules_to_assess):
                
                # Compute new rate of change and accept/reject gamma values
                automation_variables[module]['roc_change'] = roc_change(automation_variables, model, module)
                automation_variables = accept_or_reject_gamma_changes(automation_variables,
                                                                      model, module, Nyears, it, T0[n_module])
                
                if it == 0:
                    # Update initial temperature, based on differences in initial scores
                    T0[n_module] = set_initial_temperature(automation_variables, model, module)
                    print(f'Temperature for {module} is {T0[n_module]:.5f}')
                
                # Check for convergence each iteration and module loop
                gamma = automation_variables[module]['gamma'][:, :, 0]
                gamma_lag = automation_variables[module]['gamma' + '_LAG'][:, :, 0]
                
                convergence[n_module] = check_convergence(gamma, gamma_lag, module, it, max_it, convergence[n_module])
                
                if convergence[n_module]:
                    run_variables[module]['gamma'][run] = gamma
                    run_variables[module]['score'][run] = automation_variables[module]['score']
            
            # Fourth: re-run model with accepted gamma values, updating variables of interest
            automation_variables = run_model(automation_variables, model)
            
            
            # # Initial roc_change values
            # for module in modules_to_assess:
            print(f"Gamma values for condensed gas in Denkmark are: {automation_variables['FTT-H']['gamma'][1, 3, 0]:.4f}")
            print(f"roc_change and scores for diesel in Denmark are {automation_variables['FTT-H']['roc_change'][1,3]:.3f} and {automation_variables['FTT-H']['score'][1,3]:.4f}")
                
            if np.all(convergence):
                for module in modules_to_assess:
                    print(f"Median score {module} at {it}: {get_median_score(automation_variables, module):.3f}")
                break
            

    return automation_variables, run_variables
#%%
model = model_class.ModelRun()

# Only assess models if they exist and are turned on
modules_to_assess = set(model.titles['Models_short']) & set(model.ftt_modules)
modules_to_assess = ['FTT-Tr', 'FTT-P', 'FTT-H']

# %% Run combined function

# Let's try 3 runs (5 is better), and max of 100 its. Takes about 1h minutes with 3
total_runs = 3
max_it = 100
lambda_reg = 0.2  # Regularisation strength

automation_variables, run_variables = gamma_auto(model)

def select_best_gamma_values(run_variables, modules_to_assess):
    '''For each country and module, select the run with the best average score'''
    
    for module in modules_to_assess:
        avg_score = np.average(run_variables[module]['score'], axis=2)
        best_runs = np.argmax(avg_score, axis=0)
        run_variables[module]['best gamma'] = np.array([run_variables[module]['gamma'][best_runs[r], r, :]
                                                   for r in range(len(best_runs))])
        
        run_variables[module]['best score'] = np.array([run_variables[module]['score'][best_runs[r], r, :]
                                                   for r in range(len(best_runs))])
        
    return run_variables

run_variables = select_best_gamma_values(run_variables, modules_to_assess)

# print("The Belgium gamma values for transport are now:")
# for run in range(total_runs):
#     print(run_variables['FTT-Tr']['gamma'][run, 0, :-4])
    #print(f"Final best score transport: {np.max(get_median_score(run_variables, 'FTT-Tr')):.3f}")
# print(f"Final best score power: {np.max(get_median_score(run_variables, 'FTT-P')):.3f}")

# for run in range(total_runs):
#     gamma_tr = run_variables['FTT-Tr']['gamma'][run]
#     nonzero = run_variables['FTT-Tr']['score'][run]!=0
#     median_score_tr = np.median(run_variables['FTT-Tr']['score'][run][nonzero])
#     print(f"Share of gamma values in run {run} over 0.5: {np.sum(np.abs(gamma_tr) > 0.5)/np.sum(gamma_tr > -1) * 100:.1f}%")
#     print(f'Median score transport is: {median_score_tr}')
    
# nonzero = run_variables['FTT-Tr']['best score']!=0
# median_score_tr = np.median(run_variables['FTT-Tr']['best score'][nonzero])
# print(f'Median score transport is: {median_score_tr}')

# %% Saving almost to the right format (I'm naming the gamma row the same for each model.. )
import csv

n_placeholders = {"FTT-P": 11, "FTT-Tr": 4, "FTT-Fr": 0, "FTT-H": 7}
for module in modules_to_assess:
    data = run_variables[module]['best gamma'].T
    zeros = np.zeros((n_placeholders[module], data.shape[1]))
=======
        T0 = [0.001] * len(modules_to_assess)
 
        # Iterative loop for gamma convergence
        for it in range(max_it):
                        
            # 1. Save the lagged vars, and find new gamma values to try
            for mod in modules_to_assess:
                
                if it%50 == 0: # Print median score every 25 iterations
                    print(f"Median score {mod} at {it}: {get_median_score(state[mod]['score']):.3f}")
                    print(f"Mean roc_diff {mod} at {it}: {np.mean(state[mod]['roc_diff']):.4f}")
                    print(f"Mean gamma {mod} at {it}: {np.mean(abs(state[mod]['gamma'])):.4f}")

                
                # Save previous gamma and roc values
                state[mod]['gamma_lag'][:, :, 0] = state[mod]['gamma'][:, :, 0]
                state[mod]["score_lag"][:, :] = state[mod]["score"][:, :]
                
                # Update gamma values semi-randomly
                state = adjust_gamma_values_simulated_annealing(state, mod, it)
                

            # 2. Run the model, update vars of interest
            state = run_model(state, model)
            

            # 3. Save vars, accept and reject new gammas, and check convergence
            for n_mod, mod in enumerate(modules_to_assess):
                
                # Inside the for mod in modules_to_assess loop:
                state[mod]['roc_diff'], _ = compute_roc_logit(state, model, mod)
                state[mod]['score'] = compute_scores(state, mod)
                state = accept_or_reject_gamma_changes(state, mod, it, T0[n_mod])
                
                if it == 0:
                    # Update initial temperature, based on differences in initial scores
                    T0[n_mod] = set_initial_temperature(state, model, mod)
                
                # Check for convergence each iteration and module loop
                gamma = state[mod]['gamma'][:, :, 0]
                gamma_lag = state[mod]['gamma_lag'][:, :, 0]
                
                convergence[n_mod] = check_convergence(gamma, gamma_lag, mod, it, max_it, convergence[n_mod])
                
                if convergence[n_mod]:
                    run_history[mod]['gamma'][run] = gamma
                    run_history[mod]['score'][run] = state[mod]['score']
            
            # 4. Re-run model with accepted gamma values, updating vars of interest
            state = run_model(state, model)
            
                
            if np.all(convergence):
                for mod in modules_to_assess:
                    print(f"Mean roc_diff {mod} at final {it}: {np.mean(state[mod]['roc_diff']):.4f}")
                break
            

    return state, run_history
#%%
model = model_class.RunFTT()

# Compute gamma values for models turned on in settings.ini
modules_to_assess = [x.strip() for x in model.ftt_modules.split(',')]


# %% Run combined function

# Little difference between runs typically: 2 runs to be safe. 
total_runs = 2
max_it = 200
lambda_reg = 0.05  # Regularisation strength
starting_step_size = 0.3
decay_rate_steps = 0.96 
cooling_rate = 0.94 # Quite substantial cooling
convergence_threshold = 0.0010
global_weight = 0.5

state, run_history = gamma_auto(model)

def select_best_gamma_values(run_history, modules_to_assess):
    '''For each country and mod, select the run with the best average score'''
    
    for mod in modules_to_assess:
        avg_score = np.average(run_history[mod]['score'], axis=2)
        best_runs = np.argmax(avg_score, axis=0)  # shape: (regions,)
        n_regions = len(best_runs)
        region_idx = np.arange(n_regions)
        
        # Vectorized selection: gamma[best_runs[r], r, :] for all r
        run_history[mod]['best gamma'] = run_history[mod]['gamma'][best_runs, region_idx, :]
        run_history[mod]['best score'] = run_history[mod]['score'][best_runs, region_idx, :]
        
    return run_history


run_history = select_best_gamma_values(run_history, modules_to_assess)

for mod in modules_to_assess:
    print(f"Median best score {mod}: {get_median_score(run_history[mod]['best score']):.3f}")   



# %% Save almost to the right format (I'm naming the gamma row the same for each model.. )
import csv

# How many empty placeholders are in the masterfiles?
n_placeholders = {"FTT-P": 11, "FTT-Tr": 4, "FTT-Fr": 0, "FTT-H": 7}
for mod in modules_to_assess:
    data = run_history[mod]['best gamma'].T
    zeros = np.zeros((n_placeholders[mod], data.shape[1]))
>>>>>>> origin/main
    expanded_data = np.vstack([data, zeros])
    rounded_data = np.round(expanded_data, 2)
    
    # Save to CSV with mixed types
<<<<<<< HEAD
    with open(f"{module}_gamma.csv", "w", newline="") as f:
=======
    with open(f"Inputs/{mod}_gamma.csv", "w", newline="") as f:
>>>>>>> origin/main
        writer = csv.writer(f)
        
        # Write four empty lines for easier copy-paste
        for _ in range(4):
            writer.writerow([])
        
        for region in range(data.shape[1]):
            writer.writerow(["Gamma"])       # Write string row separately
            writer.writerows(rounded_data[:, region, np.newaxis])  # Write numerical data
<<<<<<< HEAD
   


=======
 
>>>>>>> origin/main
