"""
NB this code is unfinished and the functions do not currently work correctly

Rosie Hayward, Ian Burton and Femke Nijsse, Created 02/11/2021, finished 2025

An algorithm for finding the gamma values used in FTT simulations. This algorithm should find the values of gamma such that
the diffusion of shares is constant across the boundary between the historical and simulated period.


"""



# Reminder for Monday
# Code now crashes if you have multiple models selected in settings. To do with batteries? Make that code more robust.
################################################



# %%
# Third party imports
import numpy as np
from tqdm import tqdm
import os

os.chdir("C:\\Users\Work profile\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone")
#os.chdir("C:\\Users\\fjmn202\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone")

from SourceCode import model_class
# %%

def set_timeline(histend):
    "Which years are the model run?"
    # Note, make sure the timeline in the settings file at least covers this range
    return np.arange(2010, histend + 5)

def automation_init(model):
    '''Initialising automation variables and running the model for the first time'''
    
    scenario = 'S0'
    # model = model_class.ModelRun()
    modules = model.ftt_modules

    # Identifying the variables needed for automation
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    histend_vars = dict(zip(model.titles['Models_short'], model.titles['Models_histend_var']))
    techs_vars = dict(zip(model.titles['Models_short'], model.titles['tech_var']))

    # Initialising container for automation variables
    automation_variables = {}
    
    # Looping through all FTT modules
    for module in model.titles['Models_short']:
        if module in modules:
            
            print(f"Initialising for model {module}")
            # Automation variable list for this module
            automation_var_list = [share_variables[module], gamma_variables[module]]
            # Establisting timeline for automation algorithm
            model.timeline = set_timeline(model.histend[histend_vars[module]])
            # Note: for power, we must start in 2010, otherwise things go wrong, in this model version. Why?
           

            automation_variables[module] = {var: np.full_like(model.input[scenario][var][:, :, :, :len(model.timeline)], 0)
                                            for var in automation_var_list}

            automation_variables[module][gamma_variables[module]+'_LAG'] = np.full_like(model.input[scenario][gamma_variables[module]], 0)

            # Create container for roc variables
            automation_variables[module]['roc_gradient'] = np.zeros((len(model.titles['RTI_short']), len(model.titles[techs_vars[module]])))
            automation_variables[module]['roc_change'] = np.zeros((len(model.titles['RTI_short']), len(model.titles[techs_vars[module]])))
            automation_variables[module]['roc_change' + '_LAG'] = np.full_like(automation_variables[module]['roc_change'], 0)

            automation_variables[module]['hist_share_avg'] = np.zeros((len(model.titles['RTI_short']), len(model.titles[techs_vars[module]])))
            
            # Looping through years in automation timeline
            for year_index, year in enumerate(model.timeline):

                # Solving the model for each year
                model.variables, model.lags = model.solve_year(year, year_index, scenario)


                # Populate variable container for automation
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
    modules = model.ftt_modules

    # Identifying the variables needed for automation
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    histend_vars = dict(zip(model.titles['Models_short'], model.titles['Models_histend_var']))
  

    # Looping through years in automation timeline
    for year_index, year in enumerate(model.timeline):
    
        # Solving the model for each year
        model.variables, model.lags = model.solve_year(year, year_index, scenario)
        
        # Looping through all FTT modules
        for module in model.titles['Models_short']:
            if module in modules:
                print("Running model")
        
                # Automation variable list for this module
                automation_var_list = [share_variables[module], gamma_variables[module]]
                
                # Establisting timeline for automation algorithm
                model.timeline = set_timeline(model.histend[histend_vars[module]])
        
                # Overwriting input data for gamma values (broadcast to all years)
                model.input[scenario][gamma_variables[module]][:, :, 0, :] = (
                    automation_variables[module][gamma_variables[module]][:, :, :, 0])
  

                # Populate variable container for automation
                for var in automation_var_list:
                    if 'TIME' in model.dims[var]:
                        automation_variables[module][var][:, :, :, year_index] = model.variables[var]
                    else:
                        automation_variables[module][var][:, :, :, 0] = model.variables[var]
                        
            # Define the ROC variable for each module
            automation_variables[module]["rate of change"] = compute_roc(automation_variables, share_variables, module)
                

    return automation_variables

def compute_roc(automation_variables, share_variables, module):
    roc =   (automation_variables[module][share_variables[module]][:, :, :, 1:]
           - automation_variables[module][share_variables[module]][:, :, :, :-1])
    
    return roc



# %%
def roc_change(automation_variables, model, module):
    '''
    This function calculates the relative average historical rate of change (roc) and the simulated roc for each module
    Then, it calculates the difference between them
    difference = (average_roc / average_share_sim) - ( average_hist_roc / average_share_hist)
    
    Compared to a ratio of rate_of_change, this should be more stable for periods of low historical change.
    
    '''

    # Identifying the variables needed
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    
    avg_share_hist = automation_variables[module][share_variables[module]][:, :, 0, -8:-4].sum(axis=-1) / 4
    avg_share_sim  = automation_variables[module][share_variables[module]][:, :, 0, -4:  ].sum(axis=-1) / 4
    avg_growth_hist = automation_variables[module]["rate of change"][:, :, 0, -8:-4].sum(axis=-1) / 4
    avg_growth_sim  = automation_variables[module]["rate of change"][:, :, 0, -4:  ].sum(axis=-1) / 4  
    
    rel_change_hist = np.divide(avg_growth_hist, avg_share_hist, where=avg_share_hist != 0, out=np.zeros_like(avg_share_hist))
    rel_change_sim = np.divide(avg_growth_sim, avg_share_sim, where=avg_share_sim != 0, out=np.zeros_like(avg_share_sim))
    
    roc_change = np.where( (rel_change_hist == 0) | (rel_change_sim == 0), 0, rel_change_sim - rel_change_hist)
  
    automation_variables[module]['roc_change'] = roc_change
        
    return automation_variables 




def adjust_gamma_values(automation_variables, model, module, Nyears):
    '''
    This function adjusts the gamma values based on the change of gradients
    '''

    # Get model variables needed, this needs doing for every iter so not great
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    gamma = automation_variables[module][gamma_variables[module]][:, :, 0, 0]

    roc_change = automation_variables[module]['roc_change']
    gamma[roc_change > 0.02] += 0.02
    gamma[roc_change < -0.02] += -0.02

    gamma = np.clip(gamma, -1, 1)  # Ensure gamma values between -1 and 1
    gamma = np.repeat(gamma[:, :, np.newaxis, np.newaxis], Nyears, axis=3) # reshape format for whole period

    automation_variables[module][gamma_variables[module]] = np.copy(gamma)
    

    return automation_variables


def adjust_gamma_values_simulated_annealing(automation_variables, model, module, Nyears, it):
    """Randomly choose delta gamma for each model version, using a standard deviation with 
    an adjustable standard deviation."""
       
    # Get model variables needed, this needs doing for every iter so not great
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    gamma = automation_variables[module][gamma_variables[module]][:, :, 0, 0]
    
    roc_change = automation_variables[module]["roc_change"]
    
    # Generate a candidate step
    delta_gamma = np.random.normal(0, 0.015, size=gamma.shape)

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
    
    score = np.clip(score, -0.5, 0)
    score_lag = np.clip(score_lag, -0.5, 0)
    
    return score, score_lag

def accept_or_reject_gamma_changes(automation_variables, model, module, Nyears, it, T0):
    
    '''
    This function adjusts the gamma values based on a regulated simulated annealing. 
    
    That is: it'll penalise gamma values away from 0, accept some random changes
    to avoid getting in local minima as well'
    '''
    
    # Hyperparameters
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
    
    Nyears = automation_variables[module][gamma_variables[module]].shape[-1]
    gamma = np.repeat(gamma[:, :, np.newaxis, np.newaxis], Nyears, axis=3) # reshape format for whole period
    
    automation_variables[module][gamma_variables[module]] = np.copy(gamma)
    automation_variables[module]['roc_change'] = np.copy(roc_change)

    
    return automation_variables 

def set_initial_temperature(automation_variables, model, module):
    """ Set the initial temperature based on the typical change in score between
    the first two iterations
    """
    
    roc_change_lag, roc_change, gamma_variables, gamma, gamma_lag = (
        get_gamma_and_roc_change(automation_variables, model, module))
    
    score, score_lag = (
            compute_scores(gamma, gamma_lag, roc_change, roc_change_lag) )
    
    T0 = np.mean(np.abs(score - score_lag)) / 5
    
    return T0 

# %%
def gamma_auto(model):
    '''Running the iterative script. It first calls automation_init and then
    runs the simulated annealing script over all the models. 
    
   
    '''
    
    # Initialising variables
    modules = model.ftt_modules

    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    Nyears = len(model.timeline)

    # Initialising automation variables
    automation_variables = automation_init(model)
    
    
    for module in model.titles['Models_short']:
        #if module in modules:
        if module in ['FTT-P']:   
            # Should be roughly equal to expected initial improvement / 5. Computer after first iteration.
            T0 = 0.0001  
            
            # Initial roc_change values
            automation_variables = roc_change(automation_variables, model, module)
 
            # Iterative loop for gamma convergence
            #for iter in tqdm(range(5)): 
            for it in range(150):
                
                # Save previous gamma values
                automation_variables[module][gamma_variables[module]+'_LAG'][:, :, 0, 0] = (
                            automation_variables[module][gamma_variables[module]][:, :, 0, 0])
                
                # Save previous roc change values
                automation_variables[module]["roc_change" + '_LAG'][:, :] = (
                            automation_variables[module]["roc_change"][:, :] )
                
                # Update gamma values semi-randomly
                # automation_variables = adjust_gamma_values(automation_variables, model, module, Nyears)
                automation_variables = adjust_gamma_values_simulated_annealing(
                                            automation_variables, model, module, Nyears, it)

                # Running the model, updating variables of interest. 
                automation_variables = run_model(automation_variables, model)
                
                # Compute new rate of change and accept/reject gamma values
                automation_variables = roc_change(automation_variables, model, module)
                automation_variables = accept_or_reject_gamma_changes(automation_variables, model, module, Nyears, it, T0)
                
                # Check for convergence each iteration and module loop
                gamma = automation_variables[module][gamma_variables[module]][:, :, 0, 0]
                gamma_lag = automation_variables[module][gamma_variables[module] + '_LAG'][:, :, 0, 0]
                
                if it > 30 and np.average(np.absolute(gamma - gamma_lag)) < 0.001:
                    print(f"Convergence reached at iter {it}, gamma values no longer changing")
                    break
                
                if it == 0:
                    # Update initial temperature, based on differences in initial scores
                    T0 = set_initial_temperature(automation_variables, model, module) / 5
                    print(f'Temperature is {T0}')
                
                if (it+1)%10 == 1:
                    roc = automation_variables[module]['roc_change']
                    print(f"Median rate of change at {it} is {np.median(np.abs(roc))}")
                    print(f"The Germany rate of change differences are now \n: {automation_variables[module]['roc_change'][2][:19]}")
                    #print(f'Shares of gas are {automation_variables[module]["MEWS"][0, 6, 0, -8:]}')
                    print(f'gamma values are now {automation_variables[module][gamma_variables[module]][2, :19, 0, 0]}')


    return automation_variables
#%%
model = model_class.ModelRun()

# %% Run combined function
automation_variables = gamma_auto(model)
print(automation_variables['FTT-P']['MGAM'][0, :, 0, 0])


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

# %% ###################   START OF MAIN CODE RUN
