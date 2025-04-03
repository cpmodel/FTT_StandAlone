"""
Femke Nijsse, Rosie Hayward and Ian Burton. Created 2021, finished March 2025

A simulated annealing algorithm for finding the gamma values used in FTT simulations.

This algorithm should find the values of gamma such that the diffusion of shares
is constant across the boundary between the historical and simulated period.

"""

# Reminder
# Code now crashes unless you have multiple models selected in settings. To do with batteries? Make that code more robust.


# %%
# Third party imports
import numpy as np
import os


os.chdir("C:\\Users\Work profile\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone")
#os.chdir("C:\\Users\\fjmn202\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone")


from SourceCode import model_class
# %%

def set_timeline(model, modules_to_assess):
    "Run the models in parallel, so that the final year of the timeline is the max of histend"
    
    # Make sure the timeline in the settings file at least covers this range!
    histend_vars = dict(zip(model.titles['Models_short'], model.titles['Models_histend_var']))
    
    max_end = np.max([model.histend[histend_vars[module]] for module in modules_to_assess]) + 5
    
    return np.arange(2010, max_end)

def automation_init(model):
    '''Initialise automation variables and run the model for the first time'''
    
    scenario = 'S0'

    # Identifying the variables needed for automation
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    cost_variables = dict(zip(model.titles['Models_short'], model.titles['Cost_var']))
    gamma_inds = dict(zip(model.titles['Models_short'], model.titles['Gamma_ind']))
    techs_vars = dict(zip(model.titles['Models_short'], model.titles['tech_var']))

    # Initialising container for automation variables
    automation_variables = {}
    
    # Establisting timeline for automation algorithm
    # Note: for power, we must start in 2010, otherwise things go wrong, in this model version. Why?
    model.timeline = set_timeline(model, modules_to_assess)
    
    # Set up various empty values 
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
    
    # Generate a candidate step
    delta_gamma = np.random.normal(0, step_size, size=gamma.shape)

    # Propose a new gamma solution
    gamma = gamma + delta_gamma
    
    # Set gamma to zero if the roc is zero (usually as shares are zero)
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
    
    That is: penalise gamma values away from 0, accept some random changes
    to avoid getting in local minima as well'
    '''
    
    # Hyperparameter
    cooling_rate = 0.96
    
    # Cool down the temperature
    T = T0 * cooling_rate**it
    
    gamma = automation_variables[module]['gamma'][:,:,0]
    gamma_lag = automation_variables[module]['gamma_LAG'][:,:,0]
    score, score_lag = get_score_and_lagged_score(automation_variables, module)

    # Element-wise acceptance condition
    acceptance_mask = (score > score_lag) | (
        np.log(np.random.rand(*score.shape)) < (score - score_lag) / T
    )

    # Go back to old gamma/score values when values not accepted
    gamma[~acceptance_mask] = gamma_lag[~acceptance_mask]
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
    '''Return true if gamma values no longer changing much, or if max_it is reached'''
    mask = (gamma != 0) & (gamma_lag != 0)
    converged = already_converged
    gamma_change = np.average(np.absolute(gamma[mask] - gamma_lag[mask]))
    
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
    return median_score

# %%
def gamma_auto(model):
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
        
        # Break when all models have reached convergence
        convergence = [False] * len(modules_to_assess)
        
        # Computer after first iteration.
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
    expanded_data = np.vstack([data, zeros])
    rounded_data = np.round(expanded_data, 2)
    
    # Save to CSV with mixed types
    with open(f"{module}_gamma.csv", "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write four empty lines for easier copy-paste
        for _ in range(4):
            writer.writerow([])
        
        for region in range(data.shape[1]):
            writer.writerow(["Gamma"])       # Write string row separately
            writer.writerows(rounded_data[:, region, np.newaxis])  # Write numerical data
   


