"""
NB this code is unfinished and the functions do not currently work correctly

Rosie Hayward, Ian Burton and Femke Nijsse, Created 02/11/2021, finished 2025

An algorithm for finding the gamma values used in FTT simulations. This algorithm should find the values of gamma such that
the diffusion of shares is constant across the boundary between the historical and simulated period.


"""









# Reminder for Monday
# Code now crashes if you have multiple models selected in settings
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
def automation_init(model):
    '''Initialising automation variables and running the model for the first time'''
    
    scenario = 'S0'
    # model = model_class.ModelRun()
    modules = model.ftt_modules

    # Identifying the variables needed for automation
    share_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_var']))
    #roc_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_roc_var']))
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    histend_vars = dict(zip(model.titles['Models_short'], model.titles['Models_histend_var']))
    techs_vars = dict(zip(model.titles['Models_short'], model.titles['tech_var']))

    # Initialising container for automation variables
    automation_variables = {}#{module: {var: np.full_like(model.input[scenario][var][:,:,:,:len(model.timeline)], 0) for var in } for module in model.titles['Models_short']}
    # Looping through all FTT modules
    for module in model.titles['Models_short']:
        if module in modules:
            # Automation variable list for this module
            # TODO: remove the ROC variables if I can compute them endogenously
            automation_var_list = [share_variables[module], gamma_variables[module]]
            # Establisting timeline for automation algorithm
            model.timeline = np.arange(model.histend[histend_vars[module]] - 12,
                                       model.histend[histend_vars[module]] + 5) 
            years = list(model.timeline)
            years = [int(x) for x in years] ## do we need these? they are not used

            automation_variables[module] = {var: np.full_like(model.input[scenario][var][:,:,:,:len(model.timeline)], 0)
                                            for var in automation_var_list}

            automation_variables[module][gamma_variables[module]+'_LAG'] = np.full_like(model.input[scenario][gamma_variables[module]], 0)
            # Create container for roc variables
            automation_variables[module]['roc_gradient'] = np.zeros((len(model.titles['RTI_short']), len(model.titles[techs_vars[module]])))
            automation_variables[module]['hist_share_avg'] = np.zeros((len(model.titles['RTI_short']), len(model.titles[techs_vars[module]])))

            # Compute the rate of change from the shares variable
            automation_variables[module]["rate of change"] = compute_roc(automation_variables, share_variables, module)
            
            # Looping through years in automation timeline
            for year_index, year in enumerate(model.timeline):

                # Solving the model for each year
                model.variables, model.lags = model.solve_year(year, year_index,scenario)
                # the year_index starts at 2013 = 0, is this right given we start earlier?
                # Nans are introduced here

                # Populate variable container for automation
                for var in automation_var_list:
                    if 'TIME' in model.dims[var]:
                        automation_variables[module][var][:, :, :, year_index] = model.variables[var]
                    else:
                        automation_variables[module][var][:, :, :, 0] = model.variables[var]
                

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
            model.timeline = np.arange(model.histend[histend_vars[module]] - 12,
                                       model.histend[histend_vars[module]] + 4)
            years = list(model.timeline)
            years = [int(x) for x in years]

            # Overwriting input data for gamma values (broadcast to all years)
            model.input[scenario][gamma_variables[module]][:,:,0,:] = automation_variables[module][gamma_variables[module]][:, :, :, 0]

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
                        
            # Define the ROC variable for each module (can have same name I think), rather than defining it in the model
            automation_variables[module]["rate of change"] = compute_roc(automation_variables, share_variables, module)
                
            test=1

    return automation_variables

def compute_roc(automation_variables, share_variables, module):
    roc = (automation_variables[module][share_variables[module]][:, :, :, 1:]
                                            - automation_variables[module][share_variables[module]][:, :, :, :-1])
    
    return roc
# %%
def roc_ratio(automation_variables, model, module, region):
    '''
    This function calculates the average historical rate of change (roc) and the simulated roc for each module
    Then, it calculates the ratio between them
    ratio = average_roc / average_hist_roc
    
    '''

    # Identifying the variables needed for automation
    #roc_variables = dict(zip(model.titles['Models_short'], model.titles['Models_shares_roc_var']))
    techs_vars = dict(zip(model.titles['Models_short'], model.titles['tech_var']))


    # Get number of technologies
    N_techs = len(model.titles[techs_vars[module]])

    # Initialize containers for rates of change
    roc_gradient = np.zeros(N_techs)  # container for ratio of average roc to average hist roc for region
    hist_share_avg = np.zeros((N_techs))  # container for average roc for all region
    sim_share_avg = np.zeros((N_techs))  # container for average hist roc for all region

                            
    # Loop through technologies
    for i in range(N_techs):
        # seems to be an extra dimension in there set at 0 
        sim_share_avg[i] = automation_variables[module]["rate of change"][region][i][0][-4:].sum() / 4  
        
        hist_share_avg[i] = automation_variables[module]["rate of change"][region][i][0][-8:-4].sum() / 4

        if hist_share_avg[i] == 0:
            roc_gradient[i] = 0
        else:
            roc_gradient[i] = sim_share_avg[i]/hist_share_avg[i]

    automation_variables[module]['roc_gradient'][region] = roc_gradient
    automation_variables[module]['hist_share_avg'][region] = hist_share_avg
        
    
    return automation_variables  
# %%
def adjust_gamma_values(automation_variables, model, module, region):
    '''
    This function adjusts the gamma values based on the ratio of gradients
    '''

    # Get model variables needed, this needs doing for every iter so not great
    techs_vars = dict(zip(model.titles['Models_short'], model.titles['tech_var']))
    N_techs = len(model.titles[techs_vars[module]])
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))
    gamma = np.copy(automation_variables[module][gamma_variables[module]][region, :, 0, 0])


    for i in range(N_techs):
        # Get gradient between historical and simulated ROC
        gradient_ratio = automation_variables[module]['roc_gradient'][region, i]
        hist_share_avg = automation_variables[module]['hist_share_avg'][region, i]

        # Check if gradient ratio is negative (sign switch)
        if gradient_ratio < 0:
            # Check if historical average roc is negative
            if hist_share_avg < 0:
                # If yes, add to gamma value
                gamma[i] += 0.03
            # Check if historical average roc is positive
            if hist_share_avg > 0:
                # If yes, subtract from gamma value
                gamma[i] -= 0.03
        
        # Check if gradient ratio is positive ()
        if gradient_ratio > 0:
            # Check if gradient ratio is very small
            if gradient_ratio < 0.5:
                # Historical gradient is steeper than simulated gradient
                # If steep and negative, add to gamma value
                if hist_share_avg < 0:
                    gamma[i] += 0.01
                # If steep and positive, subtract from gamma value
                if hist_share_avg > 0:
                    gamma[i] -= 0.01
            # Check if gradient ratio is very large
            if gradient_ratio > 2:
                # Simulated gradient is steeper than historical gradient
                # If steep and negative, subtract from gamma value
                if hist_share_avg < 0:
                    gamma[i] -= 0.01
                # If steep and positive, add to gamma value
                if hist_share_avg > 0:
                    gamma[i] += 0.01
        # Check if gamma value is within bounds
        if gamma[i] > 1: gamma[i] = 1 
        if gamma[i] < -1: gamma[i] = -1


    gamma = np.tile(gamma.reshape(-1, 1), (1, 17)).reshape(N_techs, 1, 17) # reshape format for whole period

    automation_variables[module][gamma_variables[module]][region, :, :, :] = np.copy(gamma)
    print('Gamma values updated for region', model.titles['RTI'][region])
    

    return automation_variables

# %%
def gamma_auto(model):
    '''Running the iterative script. It calls automation_init and 
    
    TODO: There is a loop by module and by country. Can we do all countries parallel? 
    That would make the algorithm up to 70x faster
    '''
    
    # Initialising variables
    modules = model.ftt_modules
    regions = len(model.titles['RTI_short'])

    # TODO - STORE IN AUTOMATION VARIABLES
    # Identifying the variables needed for automation
    # Do we need these if they need creating inside the functions?
    gamma_variables = dict(zip(model.titles['Models_short'], model.titles['Gamma_Value']))


    # Initialising automation variables
    automation_variables = automation_init(model)
    for module in model.titles['Models_short']:
        if module in modules:
            
            for region in range(1):

                # Iterative loop for gamma convergence goes here
                # TODO make number of iterations a variable
                #for iter in tqdm(range(5)): ## generalise
                for it in range(10):

                    # Calculate ROC ratio
                    automation_variables = roc_ratio(automation_variables, model, module, region)
                    
                    # Updated lagged gamma values
                    automation_variables[module][gamma_variables[module]+'_LAG'][region, :, 0, 0] = np.copy(automation_variables[module][gamma_variables[module]][region, :, 0, 0])

                    # Perform automation algorithm, this updates the gamma values
                    automation_variables = adjust_gamma_values(automation_variables, model, module, region)
       
                    # # Check for convergence in region and module loop
                    # gamma = automation_variables[module][gamma_variables[module]][region, :, 0, 0]
                    # gamma_lag = automation_variables[module][gamma_variables[module] + '_LAG'][region, :, 0, 0]
                    
                    # if all(np.absolute(gamma - gamma_lag)) < 0.01:
                    #     print(f"Convergence reached at iter {iter}, gamma values no longer changing")
                    #     break

                    # Updating automation variables
                    automation_variables = automation_var_update(automation_variables, model)
                    
                    # print('Gamma automation iteration:', iter, 'completed')
                    if (it+1)%2 == 1:
                        print(f"The rate of change gradients are now \n: {automation_variables[module]['roc_gradient'][region][:19]}")


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
