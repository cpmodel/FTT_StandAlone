"""
Femke Nijsse, Rosie Hayward and Ian Burton. 

A simulated annealing algorithm for finding the gamma values used in FTT simulations.

This algorithm should find the values of gamma such that the diffusion of shares
is smooth across the boundary between the historical and simulated period.

For the power sector, ensure that MWKA is set to zero, as the algorithm cannot deal with partial data yet.

The data gets saved into the Inputs folder, and needs to be manually copied into the Masterfiles (to be further automated when data structures are finalised.)

"""

# %%
# Third party imports
import sys
import numpy as np
import os
from pathlib import Path

# Go two levels up: Support → SourceCode → repo root
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
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


def set_timeline(model, modules_to_assess):
    "Run the models in parallel, so that the final year of the timeline is the max of histend"
    
    # Make sure the timeline in the settings file at least covers this range!
    maps = get_model_maps(model)
    histend_vars = maps['histend']
    
    max_end = np.max([model.histend[histend_vars[mod]] for mod in modules_to_assess]) + 5
    
    return np.arange(2010, max_end)

def initialise_state(model):
    '''Initialise automation variables and run the model for the first time'''
    
    scenario = 'S0'

    # Create maps from model to specific variable name
    maps = get_model_maps(model)
    share_vars = maps['shares']
    cost_vars = maps['costs']
    gamma_inds = maps['gamma_inds']
    techs_vars = maps['techs']

    # Initialising dictionary for state variable
    state = {}
    
    # Establish timeline
    # Note: for power, we must start in 2010, otherwise things go wrong, in this model version. Why?
    model.timeline = set_timeline(model, modules_to_assess)
    
    # Set up various empty values 
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
            state[mod]["roc_diff"], state[mod]["hist_share_avg"] = compute_roc_log(state, model, mod)
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
        state[mod]['roc_diff'], _ = compute_roc_log(state, model, mod)
        state[mod]['score'] = compute_scores(state, mod)
        
    return state


def compute_roc(state, share_vars, mod):
    '''Compute the first differences (rate of change)'''
    roc =   (state[mod][share_vars[mod]][:, :, :, 1:]
           - state[mod][share_vars[mod]][:, :, :, :-1])
    
    return roc

def compute_roc_log(state, model, mod, epsilon=1e-6):
    """
    Computes the difference in percentage growth rates across the boundary.
    Uses log-space for stability and symmetry.

    FTT-Tr uses TDA1, a per-country variable for the last year of historical
    share data, so the boundary index is looked up per region.
    """
    timeline = model.timeline
    maps = get_model_maps(model)
    share_vars = maps['shares']
    histend_vars = maps['histend']

    # Extract shares and apply floor to avoid ln(0)
    # Shape: (regions, techs, years)
    shares = state[mod][share_vars[mod]][:, :, 0, :]
    shares_safe = np.maximum(shares, epsilon)

    # Calculate Log-Rate of Change (Annual % growth approximation)
    # log(S_t) - log(S_t-1)
    lroc = np.diff(np.log(shares_safe), axis=-1)

    N_regions, N_techs = shares.shape[:2]

    if mod == 'FTT-Tr':
        # Per-country boundary: TDA1 records the last historical year per region
        hist_end_years = model.input['S0']['TDA1'][:, 0, 0].astype(int)
        avg_lroc_hist = np.zeros((N_regions, N_techs))
        avg_lroc_sim  = np.zeros((N_regions, N_techs))
        avg_share_hist = np.zeros((N_regions, N_techs))
        for r in range(N_regions):
            mid_r = np.where(timeline == hist_end_years[r])[0][0]
            avg_lroc_hist[r] = np.mean(lroc[r, :, mid_r-4:mid_r], axis=-1)
            avg_lroc_sim[r]  = np.mean(lroc[r, :, mid_r:mid_r+4], axis=-1)
            avg_share_hist[r] = np.mean(shares[r, :, mid_r-4:mid_r+1], axis=-1)
    else:
        # Uniform boundary for all regions
        hist_end_year = model.histend[histend_vars[mod]]
        mid = np.where(timeline == hist_end_year)[0][0]
        avg_lroc_hist = np.mean(lroc[:, :, mid-4:mid], axis=-1)
        avg_lroc_sim  = np.mean(lroc[:, :, mid:mid+4], axis=-1)
        avg_share_hist = np.mean(shares[:, :, mid-4:mid+1], axis=-1)

    # The 'Physics' Difference
    diff = avg_lroc_sim - avg_lroc_hist

    # Mask out techs that aren't present (to avoid chasing noise)
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
    
    # Generate a candidate step
    delta_gamma = np.random.normal(0, step_size, size=gamma.shape)

    # Propose a new gamma solution
    gamma = gamma + delta_gamma
    
    # Set gamma to zero if the roc is zero (usually as shares are zero)
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
    
    That is: penalise gamma values away from 0, accept some random changes
    to avoid getting in local minima as well'
    '''
        
    # Cool down the temperature
    T = T0 * cooling_rate**it
    
    gamma = state[mod]['gamma'][:, :, 0]
    gamma_lag = state[mod]['gamma_lag'][:, :, 0]
    score = state[mod]['score']
    score_lag = state[mod]['score_lag']

    # Element-wise acceptance condition
    acceptance_mask = (score > score_lag) | (
        np.log(np.random.rand(*score.shape)) < (score - score_lag) / T
    )

    # Go back to old gamma/score values when values not accepted
    gamma[~acceptance_mask] = gamma_lag[~acceptance_mask]
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
    '''Return true if gamma values no longer changing much, or if max_it is reached'''
    mask = (gamma != 0) & (gamma_lag != 0)
    converged = already_converged
    gamma_change = np.average(np.absolute(gamma[mask] - gamma_lag[mask]))
    
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
    return median_score

# %%
def gamma_auto(model):
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
        
        # Break when all models have reached convergence
        convergence = [False] * len(modules_to_assess)
        
        # Computer after first iteration.
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
                state[mod]['roc_diff'], _ = compute_roc_log(state, model, mod)
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
model = model_class.ModelRun()

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
    expanded_data = np.vstack([data, zeros])
    rounded_data = np.round(expanded_data, 2)
    
    # Save to CSV with mixed types
    with open(f"Inputs/{mod}_gamma.csv", "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write four empty lines for easier copy-paste
        for _ in range(4):
            writer.writerow([])
        
        for region in range(data.shape[1]):
            writer.writerow(["Gamma"])       # Write string row separately
            writer.writerows(rounded_data[:, region, np.newaxis])  # Write numerical data
 