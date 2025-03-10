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
    return np.arange(histend - 12, histend + 5)

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
def automation_var_update(automation_variables, model):
    '''Runs the model with new gamma values'''
    scenario = 'S0'

    modules = model.ftt_modules

    # Identifying the variables needed for automation
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    histend_vars = dict(zip(model.titles['Models_short'], model.titles['Models_histend_var']))

    

    # Looping through all FTT modules
    for module in model.titles['Models_short']:
        if module in modules:
            # Automation variable list for this module
            automation_var_list = [share_variables[module], gamma_variables[module]]
            
            # Establisting timeline for automation algorithm
            model.timeline = set_timeline(model.histend[histend_vars[module]])

            # Overwriting input data for gamma values (broadcast to all years)
            model.input[scenario][gamma_variables[module]][:, :, 0, :] = automation_variables[module][gamma_variables[module]][:, :, :, 0]

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
                        
            # Define the ROC variable for each module
            automation_variables[module]["rate of change"] = compute_roc(automation_variables, share_variables, module)
                

    return automation_variables

def compute_roc(automation_variables, share_variables, module):
    roc =   (automation_variables[module][share_variables[module]][:, :, :, 1:]
           - automation_variables[module][share_variables[module]][:, :, :, :-1])
    
    return roc
# %%
def roc_ratio(automation_variables, model, module):
    '''
    This function calculates the average historical rate of change (roc) and the simulated roc for each module
    Then, it calculates the ratio between them
    ratio = average_roc / average_hist_roc
    
    '''

    # Setting the variables of interest
    sim_share_avg = automation_variables[module]["rate of change"][:, :, 0, -4:].sum(axis=-1) / 4  
    hist_share_avg = automation_variables[module]["rate of change"][:, :, 0, -8:-4].sum(axis=-1) / 4
    roc_gradient = np.divide(sim_share_avg, hist_share_avg, where=hist_share_avg != 0)
    
    automation_variables[module]['roc_gradient'] = roc_gradient
    automation_variables[module]['hist_share_avg'] = hist_share_avg
        
    
    return automation_variables  




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





# %%
def adjust_gamma_values(automation_variables, model, module, Nyears):
    '''
    This function adjusts the gamma values based on the ratio of gradients
    '''

    # Get model variables needed, this needs doing for every iter so not great
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    gamma = np.copy(automation_variables[module][gamma_variables[module]][:, :, 0, 0])

     # Extract gamma and other variables
    gamma = np.copy(automation_variables[module][gamma_variables[module]][:, :, 0, 0])
    roc_gradient = automation_variables[module]['roc_gradient']
    hist_share_avg = automation_variables[module]['hist_share_avg']
    
    # Gradient ratio conditions
    negative_gradient = roc_gradient < 0
    positive_gradient = roc_gradient > 0
    small_gradient = roc_gradient < 0.5
    large_gradient = roc_gradient > 2
    
    # Historical share conditions
    negative_hist = hist_share_avg < 0
    positive_hist = hist_share_avg > 0
    
    # Apply changes to gamma based on conditions
    # For negative gradient
    gamma += np.where(negative_gradient & negative_hist, 0.03, 0)  # Add 0.03 if both negative
    gamma -= np.where(negative_gradient & positive_hist, 0.03, 0)  # Subtract 0.03 if negative gradient and positive history
    
    # For positive gradient
    gamma += np.where(positive_gradient & large_gradient & positive_hist, 0.01, 0)  # Add 0.01 if large gradient and positive history
    gamma -= np.where(positive_gradient & large_gradient & negative_hist, 0.01, 0)  # Subtract 0.01 if large gradient and negative history
    
    gamma += np.where(positive_gradient & small_gradient & negative_hist, 0.01, 0)  # Add 0.01 if small gradient and negative history
    gamma -= np.where(positive_gradient & small_gradient & positive_hist, 0.01, 0)  # Subtract 0.01 if small gradient and positive history
    
    # Ensure gamma is within bounds
    gamma = np.clip(gamma, -1, 1)


    gamma = np.repeat(gamma[:, :, np.newaxis, np.newaxis], Nyears, axis=3) # reshape format for whole period
    automation_variables[module][gamma_variables[module]] = np.copy(gamma)
    
    return automation_variables



def adjust_gamma_values2(automation_variables, model, module, Nyears):
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



# %%
def gamma_auto(model):
    '''Running the iterative script. It calls automation_init and 
    
   
    '''
    
    # Initialising variables
    modules = model.ftt_modules

    # TODO - STORE IN AUTOMATION VARIABLES
    # Identifying the variables needed for automation
    # Do we need these if they need creating inside the functions?
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    Nyears = len(model.timeline)

    # Initialising automation variables
    automation_variables = automation_init(model)
    for module in model.titles['Models_short']:
        if module in modules:
            
            
            # Iterative loop for gamma convergence goes here
            # TODO make max number of iterations a variable
            #for iter in tqdm(range(5)): 
            for it in range(50):
                
                # Calculate ROC ratio
                automation_variables = roc_ratio(automation_variables, model, module)
                
                # Calculate change in relative ROC
                automation_variables = roc_change(automation_variables, model, module)
                    
                # Save previous gamma values
                automation_variables[module][gamma_variables[module]+'_LAG'][:, :, 0, 0] = (
                            automation_variables[module][gamma_variables[module]][:, :, 0, 0])
                
                # Perform automation algorithm, this updates the gamma values
                automation_variables = adjust_gamma_values(automation_variables, model, module, Nyears)
   
                # Check for convergence each iteration and module loop
                gamma = automation_variables[module][gamma_variables[module]][:, :, 0, 0]
                gamma_lag = automation_variables[module][gamma_variables[module] + '_LAG'][:, :, 0, 0]
                
                if it > 2 and np.all(np.absolute(gamma - gamma_lag) < 0.01):
                    print(f"Convergence reached at iter {iter}, gamma values no longer changing")
                    break

                # Running the model, updating variables of interest. 
                automation_variables = automation_var_update(automation_variables, model)
                
                # print('Gamma automation iteration:', iter, 'completed')
                if (it+1)%10 == 1:
                    print(f"The Germany rate of change differences are now \n: {automation_variables[module]['roc_change'][2][:19]}")
                    print(f'Shares of gas are {automation_variables[module]["MEWS"][0, 6, 0, -8:]}')


    return automation_variables
#%%
model = model_class.ModelRun()

# %% Run combined function
automation_variables = gamma_auto(model)
print(automation_variables['FTT-P']['MGAM'][0, :, 0, 0])
# #%%
# automation_variables = automation_init(model)
# print(automation_variables['FTT-P']['MGAM'][0, :, 0, :])

# #%%
# automation_variables = roc_ratio(automation_variables, model)
# print(automation_variables['FTT-P']['MGAM'][0, :, 0, :])

# #%%

# automation_variables = adjust_gamma_values(automation_variables, model)
# print(automation_variables['FTT-P']['MGAM'][0, :, 0, :])

# #%%

# automation_variables = automation_var_update(automation_variables, model)
# print(automation_variables['FTT-P']['MGAM'][0, :, 0, :])

#%%








#%%
region = 0 # this will be provided by the loop
#automation_variables = adjust_gamma_values(automation_variables, L, shar_dot_hist, region)





#def gamma_save(automation_variables, model):


    # $$
#    How to ensure gamma values are overwritten? Check Jamie's code
#    model.input["Gamma"][gamma_code][reg_pos,:,0,:] = np.array(gamma_values).reshape(-1,1)
    # model_folder = models.loc[ftt,"Short name"]

    # gamma_file = "{}_{}.csv".format(gamma_code,reg)
    # base_dir = "Inputs\\S0\\{}\\".format(model_folder)

    # gamma_df = pd.read_csv(os.path.join(rootdir,base_dir,gamma_file),index_col=0)
    # gamma_df.loc[:,:] = np.array(gamma_values).reshape(-1,1)

    # gamma_df.to_csv(os.path.join(rootdir,base_dir,gamma_file))

# %% ###################   START OF MAIN CODE RUN
