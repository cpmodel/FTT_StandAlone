"""
Femke Nijsse, Rosie Hayward and Ian Burton. Created 02/11/2021, finished March 2025

A simulated annealing algorithm for finding the gamma values used in FTT simulations.

This algorithm should find the values of gamma such that the diffusion of shares
is constant across the boundary between the historical and simulated period.

"""

# Reminder
# Code now crashes unless you have multiple models selected in settings. To do with batteries? Make that code more robust.
################################################


# %%
# Third party imports
import numpy as np
from tqdm import tqdm
import os

test_module = ['FTT-Tr']

#os.chdir("C:\\Users\Work profile\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone")
os.chdir("C:\\Users\\fjmn202\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone")


from SourceCode import model_class
# %%

def set_timeline(model, modules_to_assess):
    "Which years are the model run?"
    # Note, make sure the timeline in the settings file at least covers this range
    histend_vars = dict(zip(model.titles['Models_short'], model.titles['Models_histend_var']))
    
    max_end = np.max([model.histend[histend_vars[module]] for module in modules_to_assess]) + 5
    
    return np.arange(2010, max_end)

def automation_init(model):
    '''Initialising automation variables and running the model for the first time'''
    
    scenario = 'S0'
    # model = model_class.ModelRun()

    # Identifying the variables needed for automation
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
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
        automation_var_list = [share_variables[module], gamma_variables[module]]

        automation_variables[module] = {var:
                                        np.zeros_like(model.input[scenario][var][:, :, :, :len(model.timeline)])
                                        for var in automation_var_list}
        
        automation_variables[module][gamma_variables[module]+'_LAG'] = np.full_like(model.input[scenario][gamma_variables[module]], 0)

        # Create container for roc variables
        automation_variables[module]['roc_gradient'] = np.zeros((N_regions, N_techs))
        automation_variables[module]['roc_change'] = np.zeros((N_regions, N_techs))
        automation_variables[module]['roc_change' + '_LAG'] = np.zeros((N_regions, N_techs))

        automation_variables[module]['hist_share_avg'] = np.zeros((N_regions, N_techs))
        automation_variables[module]['score'] =  np.zeros((N_regions, N_techs))
    

    # Looping through years in automation timeline
    for year_index, year in enumerate(model.timeline):
        
        # Resetting gamma values to zero (smarter to not do this?)
        model.input[scenario][gamma_variables[module]][:, :, 0, :] = np.zeros_like(
            model.input[scenario][gamma_variables[module]][:, :, 0, :])
        
        # Solving the model for each year
        model.variables, model.lags = model.solve_year(year, year_index, scenario)
        
        # Save initial values of interest
        for module in modules_to_assess:
            
            # Automation variable list for this module
            automation_var_list = [share_variables[module], gamma_variables[module]]

            # Populate variable container
            for var in automation_var_list:
                if 'TIME' in model.dims[var]:
                    automation_variables[module][var][:, :, :, year_index] = model.variables[var]
                else:
                    automation_variables[module][var][:, :, :, 0] = model.variables[var]
        
            # Compute the rate of change from the shares variable
            automation_variables[module]["rate of change"] = compute_roc(automation_variables, share_variables, module)

    return automation_variables
# %%
def run_model(automation_variables, model):
    '''Runs the model with new gamma values'''
    scenario = 'S0'

    # Identifying the variables needed for automation
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
  
    model.timeline = set_timeline(model, modules_to_assess)
    
    # Looping through all FTT modules to update gamma values
    for module in modules_to_assess:
       
        # Overwriting input data for gamma values (broadcast to all years)
        model.input[scenario][gamma_variables[module]][:, :, 0, :] = (
            automation_variables[module][gamma_variables[module]][:, :, :, 0])

    # Looping through years in automation timeline
    for year_index, year in enumerate(model.timeline):
    
        # Solving the model for each year
        model.variables, model.lags = model.solve_year(year, year_index, scenario)
        
        for module in modules_to_assess:
            
            # Save share variables for each module
            automation_variables[module][share_variables[module]][:, :, :, year_index] = (
                 model.variables[share_variables[module]] )
            # Define the ROC variable for each module
            automation_variables[module]["rate of change"] = compute_roc(automation_variables, share_variables, module)
                
    return automation_variables


def compute_roc(automation_variables, share_variables, module):
    '''Computes the first differences (rate of change)'''
    roc =   (automation_variables[module][share_variables[module]][:, :, :, 1:]
           - automation_variables[module][share_variables[module]][:, :, :, :-1])
    
    return roc



# %%
def roc_change(automation_variables, model, module):
    '''
    This function calculates the relative average historical rate of change (roc) and the simulated roc for each module
    Then, it calculates the difference between them
    difference = (average_roc / average_share_sim) - (average_hist_roc / average_share_hist)
    
    Compared to a ratio of rate_of_change, this should be more stable for periods of low historical change.
    
    '''
    timeline = model.timeline
    
    # Identifying the variables needed
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    histend_vars = dict(zip(model.titles['Models_short'], model.titles['Models_histend_var']))
        
    mid = np.where(timeline == model.histend[histend_vars[module]])[0][0] + 1
    start, end = mid-4, mid+4
    if end == 0: end=None
    
    avg_share_hist = automation_variables[module][share_variables[module]][:, :, 0, start:mid].sum(axis=-1) / 4
    avg_share_sim  = automation_variables[module][share_variables[module]][:, :, 0, mid:end  ].sum(axis=-1) / 4
    avg_growth_hist = automation_variables[module]["rate of change"][:, :, 0, start:mid].sum(axis=-1) / 4
    avg_growth_sim  = automation_variables[module]["rate of change"][:, :, 0, mid:end  ].sum(axis=-1) / 4  
    
    rel_change_hist = np.divide(avg_growth_hist, avg_share_hist, where=avg_share_hist != 0, out=np.zeros_like(avg_share_hist))
    rel_change_sim = np.divide(avg_growth_sim, avg_share_sim, where=avg_share_sim != 0, out=np.zeros_like(avg_share_sim))
    
    roc_change = np.where( (rel_change_hist == 0) | (rel_change_sim == 0), 0, rel_change_sim - rel_change_hist)
  
    automation_variables[module]['roc_change'] = roc_change
        
    return automation_variables 



def adjust_gamma_values_simulated_annealing(automation_variables, model, module, Nyears, it):
    """Randomly choose delta gamma for each model version, using a standard deviation with 
    an adjustable standard deviation."""
       
    # Get model variables needed, this needs doing for every iter so not great
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    gamma = automation_variables[module][gamma_variables[module]][:, :, 0, 0]
    
    roc_change = automation_variables[module]["roc_change"]
    
    # Step size is a function of the number of iterations
    step_size = 0.02 * 0.995 ** it
    
    # Generate a candidate step
    delta_gamma = np.random.normal(0, step_size, size=gamma.shape)

    # Propose a new gamma solution
    gamma = gamma + delta_gamma
    
    # Set gamma to zero if the roc is zero (usually as shares are zero)
    gamma = np.where(roc_change == 0, 0, gamma)
      
    gamma = np.clip(gamma, -1, 1)  # Ensure gamma values between -1 and 1
    gamma = np.repeat(gamma[:, :, np.newaxis, np.newaxis], Nyears, axis=3) # reshape format for whole period
    automation_variables[module][gamma_variables[module]] = np.copy(gamma)

    return automation_variables


def get_gamma_and_roc_change(automation_variables, model, module):
    
    roc_change_lag = automation_variables[module]["roc_change_LAG"]
    roc_change = automation_variables[module]["roc_change"]
    
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    gamma = automation_variables[module][gamma_variables[module]][:, :, 0, 0]
    gamma_lag = automation_variables[module][gamma_variables[module]+'_LAG'][:, :, 0, 0]
    
    return roc_change_lag, roc_change, gamma_variables, gamma, gamma_lag

def compute_scores(gamma, gamma_lag, roc_change, roc_change_lag):
    '''Compute score, based on regulation which still need tweaking'''
    
    lambda_reg = 0.05  # Regularisation strength

    # Compute the regularised score for current and new solutions
    score_lag = - np.abs(roc_change_lag) - lambda_reg * gamma_lag**2
    score = - np.abs(roc_change) - lambda_reg * gamma**2
    
    score = np.clip(score, -0.8, 0)
    score_lag = np.clip(score_lag, -0.8, 0)
    
    return score, score_lag

def accept_or_reject_gamma_changes(automation_variables, model, module, Nyears, it, T0):
    
    '''
    This function adjusts the gamma values based on a regulated simulated annealing. 
    
    That is: it'll penalise gamma values away from 0, accept some random changes
    to avoid getting in local minima as well'
    '''
    
    # Hyperparameter
    cooling_rate = 0.96
    
    # Cool down the temperature
    T = T0 * cooling_rate**it
    
    roc_change_lag, roc_change, gamma_variables, gamma, gamma_lag = (
        get_gamma_and_roc_change(automation_variables, model, module))
    
    score, score_lag = (
            compute_scores(gamma, gamma_lag, roc_change, roc_change_lag) )

    # Element-wise acceptance condition
    acceptance_mask = (score > score_lag) | (
        np.log(np.random.rand(*score.shape)) < (score - score_lag) / T
    )

    # Go back to old gamma/score values when values not accepted
    gamma[~acceptance_mask] = gamma_lag[~acceptance_mask]
    roc_change[~acceptance_mask] = roc_change_lag[~acceptance_mask]
    score[~acceptance_mask] = score_lag[~acceptance_mask]
    
    gamma = np.repeat(gamma[:, :, np.newaxis, np.newaxis], Nyears, axis=3) # reshape format for whole period
    
    automation_variables[module][gamma_variables[module]] = np.copy(gamma)
    automation_variables[module]['roc_change'] = np.copy(roc_change)
    automation_variables[module]['score'] = np.copy(score)
    
    return automation_variables 

def set_initial_temperature(automation_variables, model, module):
    """ Set the initial temperature based on the typical change in score between
    the first two iterations
    """
    
    roc_change_lag, roc_change, gamma_variables, gamma, gamma_lag = (
        get_gamma_and_roc_change(automation_variables, model, module))
    
    score, score_lag = (
            compute_scores(gamma, gamma_lag, roc_change, roc_change_lag) )
    
    # Rule of thumb is to divide by 5. Bit bigger as I've clipped scores. Ignoring non-zero values
    non_zero_mask = (score != 0) & (score_lag != 0)
    T0 = np.mean(np.abs(score[non_zero_mask] - score_lag[non_zero_mask])) / 4
    
    if T0 == 0:
        raise ValueError('T0 is zero')
    
    return T0 

def check_convergence(gamma, gamma_lag, it):
    mask = (gamma != 0) & (gamma_lag != 0)
    converged = False
    if it > 30 and np.average(np.absolute(gamma[mask] - gamma_lag[mask])) < 0.002:
        print(f"Convergence reached at iter {it}, gamma values no longer changing")
        converged = True
        
    return converged
 

def select_best_gamma_values(run_variables, modules_to_assess):
    '''For each country and module, select the run with the best score'''
    
    for module in modules_to_assess:
        avg_score = np.average(run_variables[module]['score'], axis=2)
        best_runs = np.argmin(avg_score, axis=0)
        run_variables[module]['gamma'] = np.array([run_variables[module]['gamma'][best_runs[r], r, :]
                                                   for r in range(len(best_runs))])
        
    return run_variables

    

# %%
def gamma_auto(model):
    '''Running the iterative script. It first calls automation_init and then
    runs the simulated annealing script over all the models. 
    '''
    
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    Nyears = len(model.timeline)

    # Initialising automation variables
    automation_variables = automation_init(model)
    
    # Let's try 3
    total_runs = 2
    
    run_variables = {}
    
    for module in modules_to_assess:
        run_variables[module] = {}
        
        N_regions = len(model.titles['RTI_short'])
        N_techs = automation_variables[module][gamma_variables[module]].shape[1]
        
        run_variables[module]['gamma'] = np.zeros((total_runs, N_regions, N_techs))
        run_variables[module]['score'] = np.zeros((total_runs, N_regions, N_techs))
    
    for run in range(total_runs):
        
        if run > 0:
            # Initialising automation variables
            automation_variables = automation_init(model)
    
        
            
        # Should be roughly equal to expected initial improvement / 5. Computer after first iteration.
        T0 = [0.0001] * len(modules_to_assess)
        
        # Initial roc_change values
        automation_variables = roc_change(automation_variables, model, module)
 
        # Iterative loop for gamma convergence
        #for iter in tqdm(range(5)): 
        for it in range(150):
            
            # First save the lagged variables, and find new gamma values to try
            for module in modules_to_assess:
                
                # Save previous gamma and roc values
                automation_variables[module][gamma_variables[module]+'_LAG'][:, :, 0, 0] = (
                            automation_variables[module][gamma_variables[module]][:, :, 0, 0])
                
                automation_variables[module]["roc_change" + '_LAG'][:, :] = (
                            automation_variables[module]["roc_change"][:, :] )
                
                # Update gamma values semi-randomly
                automation_variables = adjust_gamma_values_simulated_annealing(
                                            automation_variables, model, module, Nyears, it)

            # Second: running the model, updating variables of interest
            automation_variables = run_model(automation_variables, model)
            
            # Third, save variables, accept and reject new gammas, and check convergence
            for n_module, module in enumerate(modules_to_assess):
                
                # Compute new rate of change and accept/reject gamma values
                automation_variables = roc_change(automation_variables, model, module)
                automation_variables = accept_or_reject_gamma_changes(automation_variables,
                                                                      model, module, Nyears, it, T0[n_module])
                
                # Check for convergence each iteration and module loop
                gamma = automation_variables[module][gamma_variables[module]][:, :, 0, 0]
                gamma_lag = automation_variables[module][gamma_variables[module] + '_LAG'][:, :, 0, 0]
                
                if check_convergence(gamma, gamma_lag, it):
                    run_variables[module]['gamma'][run] = gamma
                    run_variables[module]['score'][run] = automation_variables[module]['score']
                    break
                
                if it == 0:
                    # Update initial temperature, based on differences in initial scores
                    T0[n_module] = set_initial_temperature(automation_variables, model, module)
                    print(f'Temperature is {T0[n_module]:.5f}')
                
                # if (it+1)%10 == 1:
                #     roc = automation_variables[module]['roc_change']
                #     print(f"The Germany rate of change differences are now \n: {automation_variables[module]['roc_change'][2][:19]}")
                #     #print(f'Shares of gas are {automation_variables[module]["MEWS"][0, 6, 0, -8:]}')
                #     print(f'gamma values are now {automation_variables[module][gamma_variables[module]][2, :22, 0, 0]}')


    return automation_variables, run_variables
#%%
model = model_class.ModelRun()

# Only assess models if they exist and are turned on
modules_to_assess = set(model.titles['Models_short']) & set(model.ftt_modules)
modules_to_assess = ['FTT-Tr', 'FTT-P']

# %% Run combined function
automation_variables, run_variables = gamma_auto(model)

run_variables = select_best_gamma_values(run_variables, modules_to_assess)

print("The German gamma values for transport are now:")
print(run_variables['FTT-Tr']['gamma'][2, :-4])
print(f"Best score: {np.max(np.average(run_variables['FTT-Tr']['score'][:, 2], axis=1))}")

#%%

#def gamma_save(automation_variables, model):


    # $$
#    How to ensure gamma values are overwritten? Check Jamie's code
#    model.input["Gamma"][gamma_code][reg_pos,:,0,:] = np.array(gamma_values).reshape(-1,1)
    # model_folder = models.loc[ftt,"Short name"]

    # gamma_file = "{}_{}.csv".format(gamma_code,reg)
    # base_dir = "Inputs\\S0\\{}\\".format(model_folder)

    # gamma_df = pd.read_csv(os.path.join(rootdir,base_dir,gamma_file),index_col=0)
    # gamma_df.loc[:,:] = np.array(gamma_values).reshape(-1,1)

    # gamma_df.to_csv(os.path.join(rootdir,base_dir, gamma_file))

