"""
Femke Nijsse, Rosie Hayward and Ian Burton. 

A simulated annealing algorithm for finding the gamma values used in FTT simulations.

This algorithm should find the values of gamma such that the diffusion of shares
is smooth across the boundary between the historical and simulated period.

For the power sector, ensure that MWKA is set to zero, as the algorithm cannot deal with partial data yet.

"""

# %%
# Third party imports
import numpy as np
import os
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


def set_timeline(model, modules_to_assess):
    "Run the models in parallel, so that the final year of the timeline is the max of histend"
    
    # Make sure the timeline in the settings file at least covers this range!
    maps = get_model_maps(model)
    histend_vars = maps['histend']
    
    max_end = np.max([model.histend[histend_vars[mod]] for mod in modules_to_assess]) + 5
    
    return np.arange(2010, max_end)

def automation_init(model):
    '''Initialise automation variables and run the model for the first time'''
    
    scenario = 'S0'

    # Identifying the variables needed for automation
    maps = get_model_maps(model)
    share_vars = maps['shares']
    cost_vars = maps['costs']
    gamma_inds = maps['gamma_inds']
    techs_vars = maps['techs']

    # Initialising container for automation variables
    annealing_state = {}
    
    # Establisting timeline for automation algorithm
    # Note: for power, we must start in 2010, otherwise things go wrong, in this model version. Why?
    model.timeline = set_timeline(model, modules_to_assess)
    
    # Set up various empty values 
    for mod in modules_to_assess:
        print(f"Initialising for model {mod}")
        
        N_regions = len(model.titles['RTI_short'])
        N_techs = len(model.titles[techs_vars[mod]])
      
        # Automation variable list for this module
        annealing_state[mod] = {}
            
        annealing_state[mod][share_vars[mod]] = (
                        np.zeros_like(model.input[scenario][share_vars[mod]][:, :, :, :len(model.timeline)]) )
        
        annealing_state[mod]['gamma'] = np.zeros((N_regions, N_techs, 1))
        annealing_state[mod]['gamma_lag'] = np.zeros_like(annealing_state[mod]['gamma'])

        # Create container for rate of change (roc) vars
        annealing_state[mod]['roc_diff'] = np.zeros((N_regions, N_techs))
        annealing_state[mod]['roc_diff_lag'] = np.zeros((N_regions, N_techs))

        annealing_state[mod]['hist_share_avg'] = np.zeros((N_regions, N_techs))
        annealing_state[mod]['score'] =  np.zeros((N_regions, N_techs))
        annealing_state[mod]['score_lag'] =  np.zeros((N_regions, N_techs))


    # Looping through years in automation timeline
    for year_index, year in enumerate(model.timeline):
        
        # Solving the model for each year
        model.variables, model.lags = model.solve_year(year, year_index, scenario)
        
        # Save initial values of interest
        for mod in modules_to_assess:
            
            # Resetting gamma values to zero (smarter to start from existing gamma values?)
            model.input[scenario][cost_vars[mod]][:, :, gamma_inds[mod], :] = np.zeros_like(
                model.input[scenario][cost_vars[mod]][:, :, gamma_inds[mod], :])
            
            # Read in the historical and simulated shares
            var = share_vars[mod]
            annealing_state[mod][var][:, :, :, year_index] =  model.variables[var]
        
            # Compute the rate of change from the shares variable
            annealing_state[mod]["rate of change"] = compute_roc(annealing_state, share_vars, mod)            
            annealing_state[mod]["roc_diff"], annealing_state[mod]["hist_share_avg"] = compute_roc_log(annealing_state, model, mod)
            annealing_state[mod]["score"] = get_scores_refined(annealing_state, mod)

    return annealing_state
# %%
def run_model(annealing_state, model):
    '''Run the model with new gamma values'''
    scenario = 'S0'

    # Identifying the model vars needed
    maps = get_model_maps(model)
    share_vars = maps['shares']
    cost_vars = maps['costs']
    gamma_inds = maps['gamma_inds']
  
    model.timeline = set_timeline(model, modules_to_assess)
    
    # Looping through all FTT mods to update gamma values
    for mod in modules_to_assess:
            
        model.input[scenario][cost_vars[mod]][:, :, gamma_inds[mod], :] = (
            annealing_state[mod]['gamma'][:, :, :])

    # Looping through years in timeline
    for year_index, year in enumerate(model.timeline):
    
        # Solving the model for each year
        model.variables, model.lags = model.solve_year(year, year_index, scenario)
        
        for mod in modules_to_assess:
            
            # Save share vars for each mod
            annealing_state[mod][share_vars[mod]][:, :, :, year_index] = (
                 model.variables[share_vars[mod]] )
    
    for mod in modules_to_assess:
        # Define the ROC variable for each mod
        annealing_state[mod]["rate of change"] = compute_roc(annealing_state, share_vars, mod)
        # annealing_state[mod]["roc_diff"], _ = roc_diff(annealing_state, model, mod)
        # annealing_state[mod]["score"] = get_scores(annealing_state, mod)
        
        # Inside the for mod in modules_to_assess loop:
        annealing_state[mod]['roc_diff'], _ = compute_roc_log(annealing_state, model, mod)
        annealing_state[mod]['score'] = get_scores_refined(annealing_state, mod)
        
    return annealing_state


def compute_roc(annealing_state, share_vars, mod):
    '''Compute the first differences (rate of change)'''
    roc =   (annealing_state[mod][share_vars[mod]][:, :, :, 1:]
           - annealing_state[mod][share_vars[mod]][:, :, :, :-1])
    
    return roc

def compute_roc_log(annealing_state, model, mod, epsilon=1e-6):
    """
    Computes the difference in percentage growth rates across the boundary.
    Uses log-space for stability and symmetry.
    """
    timeline = model.timeline
    maps = get_model_maps(model)
    share_vars = maps['shares']
    histend_vars = maps['histend']
    
    # Boundary identification
    hist_end_year = model.histend[histend_vars[mod]]
    mid = np.where(timeline == hist_end_year)[0][0]
    
    # Extract shares and apply floor to avoid ln(0)
    # Shape: (regions, techs, years)
    shares = annealing_state[mod][share_vars[mod]][:, :, 0, :]
    shares_safe = np.maximum(shares, epsilon)
    
    # Calculate Log-Rate of Change (Annual % growth approximation)
    # log(S_t) - log(S_t-1)
    lroc = np.diff(np.log(shares_safe), axis=-1)
    
    # Define Windows: 4 years of growth rate
    # Historical LROC: the 4 growth steps leading UP to the boundary
    # Simulated LROC: the first 4 growth steps OF the simulation
    avg_lroc_hist = np.mean(lroc[:, :, mid-4:mid], axis=-1)
    avg_lroc_sim  = np.mean(lroc[:, :, mid:mid+4], axis=-1)
    
    # The 'Physics' Difference
    diff = avg_lroc_sim - avg_lroc_hist
    
    # Calculate avg share for weighting
    avg_share_hist = np.mean(shares[:, :, mid-4:mid+1], axis=-1)
    
    # Mask out techs that aren't present (to avoid chasing noise)
    diff = np.where(avg_share_hist <= epsilon, 0, diff)
    
    return diff, avg_share_hist



# %%
def roc_diff(annealing_state, model, mod):
    '''
    Calculate the relative average historical rate of change (roc) and the simulated roc for each mod
    Then, calculate the difference between them
    difference = (average_roc / average_share_sim) - (average_hist_roc / average_share_hist)
    
    Compared to a ratio of rate_of_change, this should be more stable for periods of low historical change.
    
    '''
    timeline = model.timeline
    
    # Identifying the vars needed
    maps = get_model_maps(model)
    share_vars = maps['shares']
    histend_vars = maps['histend']
        
    mid = np.where(timeline == model.histend[histend_vars[mod]])[0][0] + 1
    start, end, end1 = mid-4, mid+4, mid+4
    if end == 0: end1=None
    
    avg_share_hist = annealing_state[mod][share_vars[mod]][:, :, 0, start:mid].sum(axis=-1) / 4
    avg_share_sim  = annealing_state[mod][share_vars[mod]][:, :, 0, mid:end1  ].sum(axis=-1) / 4
    avg_growth_hist = annealing_state[mod]["rate of change"][:, :, 0, start-1:mid-1].sum(axis=-1) / 4
    avg_growth_sim  = annealing_state[mod]["rate of change"][:, :, 0, mid-1:end-1  ].sum(axis=-1) / 4  
    boundary_share = annealing_state[mod][share_vars[mod]][:, :, 0, mid-1]
    
    rel_change_hist = np.divide(avg_growth_hist, avg_share_hist, where=avg_share_hist != 0, out=np.zeros_like(avg_share_hist))
    rel_change_sim = np.divide(avg_growth_sim, avg_share_sim, where=avg_share_sim != 0, out=np.zeros_like(avg_share_sim))
    
    roc_diff = np.where( (rel_change_hist == 0) | (rel_change_sim == 0), 0, rel_change_sim - rel_change_hist)
    
    return roc_diff, boundary_share 



def adjust_gamma_values_simulated_annealing(annealing_state, model, mod, it):
    """Randomly choose delta gamma for each model version, using a standard deviation with 
    an adjustable standard deviation."""
       
    # Get model vars needed, this needs doing for every iter so not great
    gamma = annealing_state[mod]['gamma'][:, :, 0]
    
    roc_diff = annealing_state[mod]["roc_diff"]
    
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
    annealing_state[mod]['gamma'] = gamma[:, :, np.newaxis]

    return annealing_state


def get_gamma_and_roc_diff(annealing_state, model, mod):
    
    roc_diff_lag = annealing_state[mod]["roc_diff_lag"]
    roc_diff = annealing_state[mod]["roc_diff"]
    
    maps = get_model_maps(model)
    gamma_vars = maps['gamma_value']
    gamma = annealing_state[mod]['gamma'][:, :, 0]
    gamma_lag = annealing_state[mod]['gamma_lag'][:, :, 0]
    
    return roc_diff_lag, roc_diff, gamma_vars, gamma, gamma_lag

def get_score_and_lagged_score(annealing_state, mod):
    '''Get score and score lag to compare'''
    
    # Compute the regularised score for current and new solutions
    score = annealing_state[mod]['score']
    score_lag = annealing_state[mod]['score_lag']
    
    return score, score_lag



def get_scores(annealing_state, mod):
    """Compute score combining local and global contributions, ignoring inactive techs."""
    
    roc_diff = annealing_state[mod]['roc_diff']
    gamma = annealing_state[mod]['gamma'][:, :, 0]
    
    # Local score
    local_score = -np.abs(roc_diff) - lambda_reg * gamma**2
    
    # Mask for active techs (exclude technologies that are off)
    active_mask = (roc_diff != 0) | (gamma != 0)
    
    # Compute global score per region over active techs
    sums = np.sum(np.abs(roc_diff) * active_mask, axis=1)
    counts = np.count_nonzero(active_mask, axis=1)
    # avoid divide by zero
    counts = np.where(counts == 0, 1, counts)
    global_score = np.where(active_mask, -sums[:, np.newaxis] / counts[:, np.newaxis], 0)
    
    # Combine local and global scores
    combined_score = (1 - global_weight) * local_score + global_weight * global_score
    
    return np.tanh(combined_score)


def get_scores_refined(annealing_state, mod):
    """
    Computes a score that balances global and local fit,
    weighted by the importance (sqrt of share) of each technology.
    """
    roc_diff = annealing_state[mod]['roc_diff']
    gamma = annealing_state[mod]['gamma'][:, :, 0]
    shares = annealing_state[mod]['hist_share_avg']
    
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




def accept_or_reject_gamma_changes(annealing_state, model, mod, it, T0):
    
    '''
    Adjust the gamma values based on regulated simulated annealing. 
    
    That is: penalise gamma values away from 0, accept some random changes
    to avoid getting in local minima as well'
    '''
        
    # Cool down the temperature
    T = T0 * cooling_rate**it
    
    gamma = annealing_state[mod]['gamma'][:, :, 0]
    gamma_lag = annealing_state[mod]['gamma_lag'][:, :, 0]
    score, score_lag = get_score_and_lagged_score(annealing_state, mod)

    # Element-wise acceptance condition
    acceptance_mask = (score > score_lag) | (
        np.log(np.random.rand(*score.shape)) < (score - score_lag) / T
    )

    # Go back to old gamma/score values when values not accepted
    gamma[~acceptance_mask] = gamma_lag[~acceptance_mask]
    annealing_state[mod]['gamma'] = gamma[:, :, np.newaxis]
    nonzero_mask = (score != 0) & (score_lag != 0)
    annealing_state[mod]['acceptance_rate'] = np.sum(acceptance_mask[nonzero_mask]) / acceptance_mask[nonzero_mask].size
    
    if it%50 == 0:
        print(f"At {it} it, the acceptance rate is {annealing_state[mod]['acceptance_rate']:.3f} at T {T:.5f}")

        
    return annealing_state 

def set_initial_temperature(annealing_state, model, mod):
    """ Set the initial temperature T0 based on the typical change in score
    between the first two iterations
    """
    
    roc_diff_lag, roc_diff, gamma_vars, gamma, gamma_lag = (
        get_gamma_and_roc_diff(annealing_state, model, mod))
    
    score, score_lag = get_score_and_lagged_score(annealing_state, mod)
    
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
        print(f"Convergence {mod} reached at iter {it}, little change in gamma values last iteration")
        converged = True
        
    elif it >= max_it - 1:
        print(f"Maximum iterations reached at it {it}. Gammas {mod} still changing by {gamma_change:.4f} on average")
        converged = True
    
    return converged
 
def get_median_score(scores):
    nonzero = (scores != 0)
    median_score = np.median(scores[nonzero])
    return median_score

# %%
def gamma_auto(model):
    '''Run the iterative script. First call automation_init and then
    run the simulated annealing script over all the models. 
    '''
    
    # Initialising automation vars
    annealing_state = automation_init(model)
    run_vars = {}
    
    for mod in modules_to_assess:
        run_vars[mod] = {}
        
        N_regions = len(model.titles['RTI_short'])
        N_techs = annealing_state[mod]['gamma'].shape[1]
        
        run_vars[mod]['gamma'] = np.zeros((total_runs, N_regions, N_techs))
        run_vars[mod]['score'] = np.zeros((total_runs, N_regions, N_techs))
    
    for run in range(total_runs):
         
        if run > 0:
            # Initialising automation vars
            annealing_state = automation_init(model) 
        
        # Initial roc_diff values
        for mod in modules_to_assess:
            annealing_state[mod]['roc_diff'], annealing_state[mod]['hist_share_avg'] = roc_diff(annealing_state, model, mod)
        
        # Break when all models have reached convergence
        convergence = [False] * len(modules_to_assess)
        
        # Computer after first iteration.
        T0 = [0.001] * len(modules_to_assess)
 
        # Iterative loop for gamma convergence
        for it in range(max_it):
                        
            # First save the lagged vars, and find new gamma values to try
            for mod in modules_to_assess:
                
                if it%50 == 0: # Print median score every 25 iterations
                    print(f"Median score {mod} at {it}: {get_median_score(annealing_state[mod]['score']):.3f}")
                    print(f"Mean roc_diff {mod} at {it}: {np.mean(annealing_state[mod]['roc_diff']):.4f}")
                    print(f"Mean gamma {mod} at {it}: {np.mean(abs(annealing_state[mod]['gamma'])):.4f}")

                
                # Save previous gamma and roc values
                annealing_state[mod]['gamma_lag'][:, :, 0] = (
                            annealing_state[mod]['gamma'][:, :, 0])
                
                annealing_state[mod]["roc_diff_lag"][:, :] = (
                            annealing_state[mod]["roc_diff"][:, :] )
                
                annealing_state[mod]["score_lag"][:, :] = (
                            annealing_state[mod]["score"][:, :] )
                
                # Update gamma values semi-randomly
                annealing_state = adjust_gamma_values_simulated_annealing(
                                            annealing_state, model, mod, it)
                

            # Second: running the model, updating vars of interest
            annealing_state = run_model(annealing_state, model)
            
            
            
            # Third, save vars, accept and reject new gammas, and check convergence
            for n_mod, mod in enumerate(modules_to_assess):
                
                # Inside the for mod in modules_to_assess loop:
                annealing_state[mod]['roc_diff'], _ = compute_roc_log(annealing_state, model, mod)
                annealing_state[mod]['score'] = get_scores_refined(annealing_state, mod)
                annealing_state = accept_or_reject_gamma_changes(annealing_state, model, mod, it, T0[n_mod])
                
                if it == 0:
                    # Update initial temperature, based on differences in initial scores
                    T0[n_mod] = set_initial_temperature(annealing_state, model, mod)
                
                # Check for convergence each iteration and module loop
                gamma = annealing_state[mod]['gamma'][:, :, 0]
                gamma_lag = annealing_state[mod]['gamma_lag'][:, :, 0]
                
                convergence[n_mod] = check_convergence(gamma, gamma_lag, mod, it, max_it, convergence[n_mod])
                
                gamma_global.append(gamma[1, 3])
                
                if convergence[n_mod]:
                    run_vars[mod]['gamma'][run] = gamma
                    run_vars[mod]['score'][run] = annealing_state[mod]['score']
            
            # Fourth: re-run model with accepted gamma values, updating vars of interest
            annealing_state = run_model(annealing_state, model)
            
                
            if np.all(convergence):
                for mod in modules_to_assess:
                    print(f"Mean roc_diff {mod} at final {it}: {np.mean(annealing_state[mod]['roc_diff']):.4f}")
                break
            

    return annealing_state, run_vars
#%%
model = model_class.ModelRun()
# Cache simple model maps for reuse across this module
try:
    model.maps = get_model_maps(model)
except Exception:
    # Don't fail if mapping cannot be built here; code will still work using inline dict(zip(...))
    model.maps = {}

# Compute gamma values for models turned on in settings.ini
modules_to_assess = [x.strip() for x in model.ftt_modules.split(',')]


# %% Run combined function

# Little difference between runs typically: 3 runs to be safe. 
# Consider reducing no_it for more rapid estimates
total_runs = 2
max_it = 200
lambda_reg = 0.05  # Regularisation strength
starting_step_size = 0.3
decay_rate_steps = 0.96 
cooling_rate = 0.94 # Quite substantial cooling
convergence_threshold = 0.0010
global_weight = 0.5

gamma_global = []

annealing_state, run_vars = gamma_auto(model)

def select_best_gamma_values(run_vars, modules_to_assess):
    '''For each country and mod, select the run with the best average score'''
    
    for mod in modules_to_assess:
        avg_score = np.average(run_vars[mod]['score'], axis=2)
        best_runs = np.argmax(avg_score, axis=0)  # shape: (regions,)
        n_regions = len(best_runs)
        region_idx = np.arange(n_regions)
        
        # Vectorized selection: gamma[best_runs[r], r, :] for all r
        run_vars[mod]['best gamma'] = run_vars[mod]['gamma'][best_runs, region_idx, :]
        run_vars[mod]['best score'] = run_vars[mod]['score'][best_runs, region_idx, :]
        
    return run_vars


run_vars = select_best_gamma_values(run_vars, modules_to_assess)

for mod in modules_to_assess:
    print(f"Median best score {mod}: {get_median_score(run_vars[mod]['best score']):.3f}")   



# %% Saving almost to the right format (I'm naming the gamma row the same for each model.. )
import csv

# How many empty placeholders are in the masterfiles?
n_placeholders = {"FTT-P": 11, "FTT-Tr": 4, "FTT-Fr": 0, "FTT-H": 7}
for mod in modules_to_assess:
    data = run_vars[mod]['best gamma'].T
    zeros = np.zeros((n_placeholders[mod], data.shape[1]))
    expanded_data = np.vstack([data, zeros])
    rounded_data = np.round(expanded_data, 2)
    
    # Save to CSV with mixed types
    with open(f"{mod}_gamma.csv", "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write four empty lines for easier copy-paste
        for _ in range(4):
            writer.writerow([])
        
        for region in range(data.shape[1]):
            writer.writerow(["Gamma"])       # Write string row separately
            writer.writerows(rounded_data[:, region, np.newaxis])  # Write numerical data
 


