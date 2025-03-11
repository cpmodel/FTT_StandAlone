import pandas as pd
import os
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar


# Local imports
from SourceCode.support.titles_functions import load_titles
titles = load_titles()
# Opt-in to the future pandas behavior to avoid warnings
pd.set_option('future.no_silent_downcasting', True)

def load_comparison_data(comparison_path):
    """
    Load comparison data from the comparison file.
    
    Parameters:
    - comparison_path: str,s path to the comparison file
    
    Returns:
    - dict, dictionary containing DataFrames for each sheet
    - list, list of sheet names
    """
    # Get the sheet names from the workbook
    sheet_names = pd.ExcelFile(comparison_path).sheet_names
    
    compare_data = {}
    for sheet_name in sheet_names:
        compare_data[sheet_name] = pd.read_excel(comparison_path, sheet_name=sheet_name)
    
    return compare_data, sheet_names

def load_input_data(base_master_path, sheet_names):
    """
    Load input data from the master file.
    
    Parameters:
    - base_master_path: str, path to the master file
    - sheet_names: list of str, sheet names to load
    
    Returns:
    - dict, dictionary containing DataFrames for each sheet
    """
    input_data = {}
    for sheet_name in sheet_names:
        input_data[sheet_name] = pd.read_excel(base_master_path, 
                                               sheet_name=sheet_name, 
                                               skiprows=3, 
                                               usecols=lambda column: column not in range(0, 1) and column not in range(22, 26))
    return input_data

def load_cost_matrix(base_master_path, cost_matrix_var, cost_matrix_structure):
    """
    Load input data from the master file.
    
    Parameters:
    - base_master_path: str, path to the master file
    - sheet_names: list of str, sheet names to load
    
    Returns:
    - dict, dictionary containing DataFrames for each sheet
    """
    raw_cost_matrix = load_input_data(base_master_path, cost_matrix_var)
    raw_cost_matrix = raw_cost_matrix[cost_matrix_var[0]]
    
    # TODO overlap with titles, generalise
    # Get cost matrix info
    regions = cost_matrix_structure['regions']
    data_length = cost_matrix_structure['data_length']
    tech_number = cost_matrix_structure['tech_number']

    cost_matx = pd.DataFrame()
    for j in range(0, regions*data_length, data_length):  # loops through each country based on structure
        cost_matx = pd.concat([cost_matx, raw_cost_matrix.iloc[j:j+tech_number+1]]).reset_index(drop=True)

    return cost_matx

def scen_levels_extend(scenario_levels, region_groups):
    
    # List comprehension to filter elements that end with '_pol'
    regions = list({country.split('_')[0] for country in list(scenario_levels.columns) if country.endswith('_pol')})    
    
    # for key, additional_regions in region_groups.items():
    #     if key in regions:
    #         regions.extend(additional_regions)
    #         for region in additional_regions:
    #             scenario_levels[region + '_phase_pol'] = scenario_levels[key + '_phase_pol']
    #             scenario_levels[region + '_cp_pol'] = scenario_levels[key + '_cp_pol']
    #             scenario_levels[region + '_price_pol'] = scenario_levels[key + '_price_pol']
    
    new_columns = {}  # Dictionary to collect new column data

    for key, additional_regions in region_groups.items():
        if key in regions:
            regions.extend(additional_regions)  # Extend the list of regions
            for region in additional_regions:
                new_columns[region + '_phase_pol'] = scenario_levels[key + '_phase_pol']
                new_columns[region + '_cp_pol'] = scenario_levels[key + '_cp_pol']
                new_columns[region + '_price_pol'] = scenario_levels[key + '_price_pol']

    # Efficiently add new columns in one step using pd.concat
    return pd.concat([scenario_levels, pd.DataFrame(new_columns)], axis=1)


    #return scenario_levels

def pol_vary_general(updated_input_data, input_data, scen_level, compare_data, scenarios, region_groups, params):
    """
    Update input data based on scenario levels and comparison data.
    
    Parameters:
    - input_data: dict, dictionary containing input DataFrames
    - scen_level: pd.DataFrame, DataFrame containing the scenario levels
    - compare_data: dict, dictionary containing comparison DataFrames
    - amb_scenario: str, ambitious scenario identifier
    - region_groups: dict, dictionary containing region groups
    
    Returns:
    - dict, dictionary containing updated input DataFrames
    - str, new scenario code
    """
    new_scen_code = scen_level['scenario']
    amb_scenario = scenarios['ambitious']
    base_scenario = scenarios['base']

    # Region list from updated scenario levels, not DRY
    regions = list({country.split('_')[0] for country in scen_level.index if country.endswith('_pol')})

    # Loop through general policy parameters
    new_sheets = pd.DataFrame()

    for policy_parameter in params['general']: # isolates general policy parameters
        
        # Get variables from policy dictionary
        sheet_names = list(params['general'][policy_parameter])


        for sheet_name in sheet_names:
            if sheet_name not in compare_data:
                continue  # Skip to the next iteration if the sheet_name does not exist in compare_data
            
            # Initialise dictionary
            updated_input_data[new_scen_code][sheet_name] = {}

            var_df = compare_data[sheet_name]
            amb_df = var_df[var_df['Scenario'] == amb_scenario].reset_index(drop=True)
            base_df = var_df[var_df['Scenario'] == base_scenario].reset_index(drop=True)

            for row in range(0, len(amb_df.index)):
                technology = amb_df['Technology'].iloc[row]
                country = amb_df['Country'].iloc[row]
                if amb_df['Country'].iloc[row] in regions:
                    ambition = scen_level[f'{country}_{policy_parameter}']

                else:
                    print(f'No ambition level for {country} in {policy_parameter}')

                # Handle rollback for certain technologies TODO generalise
                # if (country == 'US') & (policy_parameter == 'phase_pol'):
                #     roll_back_techs = ["Onshore", "Offshore", "Solar PV"]
                
                #     if (technology in roll_back_techs) & (ambition >= 0.5):
                #         continue
                #     elif (technology not in roll_back_techs) & (ambition < 0.5):
                #         continue
                #     elif (technology in roll_back_techs) & (ambition < 0.5):
                #         ambition = (0.5 - ambition) / 0.5
                #     elif (technology not in roll_back_techs) & (ambition >= 0.5):
                #         ambition = (ambition - 0.5) / 0.5

                # Extract meta data and bounds
                meta = amb_df.iloc[row, 0:5]
                upper_bound = amb_df.iloc[row, 5:]
                lower_bound = base_df.iloc[row, 5:]
                
                # Calculate new level
                new_level = (upper_bound - lower_bound) * ambition
                
                new_level_meta = pd.concat([meta, new_level])
                new_level_meta = pd.DataFrame(new_level_meta.drop('Scenario')).T
                new_sheets = pd.concat([new_sheets, new_level_meta], axis=0)
    

            master = input_data[sheet_name]
            
            # read in dataframe of changes in new scenario, change name of df1 for better understanding
            variable_df = new_sheets[new_sheets['Sheet'] == sheet_name].reset_index(drop = True)
            
            # Get countries from comparison data, not DRY
            countries = pd.unique(variable_df['Country']) # list of countries to loop through
        
            for country in countries:            

                # Country dataframe to merge in
                country_df = variable_df[variable_df['Country'] == country].reset_index(drop = True)
                # Drop meta data
                country_df = country_df.drop(columns = list(country_df.columns[0:3]))
                country_df = country_df.set_index('Technology')
                
                # update master df, create it and deal with instance of BE
                if country != 'BE':
                    master_start = master[master.iloc[:, 0] == country].index[0]
                else: 
                    master_start = 0
                master_end = master_start + 23 # all rows until next country
                
                
                master_df = master[master_start:master_end].reset_index(drop = True)
                # Change title of first col to match up with current country_inputs
                master_df.iloc[0, 0] = 'Technology'
                master_df.columns = master_df.iloc[0]
                master_df = master_df[1:]
                master_df = master_df.set_index('Technology')
                
                # Update 
                master_df = master_df.astype('object')
                master_df.update(country_df) # as below make seperate object for comparison in debugging
                updated_df = master_df.reset_index()
                updated_df.columns = [''] + list(range(2001, 2101))
                
                # Add tech numbers
                for row in list(updated_df.index):
                    updated_df.loc[row, ''] = str(row + 1) + ' ' + updated_df.loc[row, '']

                updated_input_data[new_scen_code][sheet_name][country] = updated_df    


    return updated_input_data

def pol_vary_special(updated_input_data, scen_level, carbon_price_path): # TODO generalise paths in config?
    '''
    Update the technoeconomic parameters that are not in cost matrix

    Parameters:
    Returns:
    '''

    # load carbon price data
    cp_df = pd.read_csv(carbon_price_path)
    cp_df = cp_df.rename(columns={'Unnamed: 0': ''})
    cp_df = cp_df.astype({col: 'float64' for col in cp_df.columns[1:]})  # Force float dtype
    cp_df.iloc[:, 1:] = cp_df.iloc[:, 1:].astype(float)


    new_scen_code = scen_level.loc['scenario']
    sheet_name = 'REPPX' # hardcoded for now, handle via dimensions later
    


    # Region list from updated scenario levels, not DRY
    regions = list({country.split('_')[0] for country in scen_level.index if country.endswith('_pol')})

    for index, row in cp_df.iterrows():
        country = row['']

        # assign ambition levels
        if country in regions:
            ambition = scen_level.loc[country + '_cp_pol'] 
        else:
            print(f'No ambition level for {country} in carbon price')
        
        
        # Multiply all values in the row (except country col and 2010) by the ambition value
        cp_df.iloc[index, 14:] = round((cp_df.iloc[index, 14:] * ambition), 2)
        
    # Store single sheet
    updated_input_data[new_scen_code][sheet_name] = cp_df

def update_cost_matrix(cost_matrix, technology, updates, scen_level):
    """
    Update the cost matrix for a specific tech or fuel
    
    Parameters:
    - cost_matrix: pd.DataFrame
    - technology: str, the technology or fuel to update
    - updates: dict, dictionary containing the updates for the technology or fuel
    - scen_level: pd.Series, a single row from the scenario levels DataFrame
    
    Returns:
    - dataframe containing updated cost matrix
    """

    tech_update = cost_matrix['Unnamed: 1'] == technology
    if 'std_col' in updates:
        # Handle fuel price updates
        price_col = updates[f"{technology.lower()}_price"]
        std_col = updates['std_col']
        
        fuel_price = cost_matrix.loc[tech_update, price_col]
        fuel_std = cost_matrix.loc[tech_update, std_col]
        fuel_lower = fuel_price - (fuel_std * 2)
        fuel_upper = fuel_price + (fuel_std * 2)
        fuel_diff = fuel_upper - fuel_lower
        fuel_vary = fuel_diff * scen_level[f"{technology.lower()}_price"]
        fuel_price_new = fuel_lower + fuel_vary
        cost_matrix.loc[tech_update, price_col] = fuel_price_new
    else:
        # Handle technology updates
        for param, col in updates.items():
            cost_matrix.loc[tech_update, col] = scen_level[param]

def update_discount_rates(cost_matrix, technology, updates, scen_level,
                          tech_rates, scalar, min_rate, max_rate): ## Do we even need this?
    
    # Create country specific sheet
    country = cost_matrix.loc[i, 'Unnamed: 1']
    country_df = cost_matrix.iloc[i:i + tech_number + 1, :].reset_index(drop=True)

    # Scale and shift to the target range
    #scaled_rates = min_rate + (max_rate - min_rate) * scalar * normalized_rates
    min_rate = 0.01 # generalise to take these from config
    max_rate = 0.2

    # normalise the rates
    country_df.iloc[1:, 17] = country_df.iloc[1:, 17] / max_rate
    
    
    
    ############ Convert rates to an array
    rates = np.array(list(tech_rates.values()))
    
    # Normalize rates relative to their maximum value
    normalized_rates = rates / rates.max()
    
    # Scale and shift to the target range
    scaled_rates = min_rate + (max_rate - min_rate) * scalar * normalized_rates
    
    # Map scaled values back to tech names
    return {tech: scaled_rate for tech, scaled_rate in zip(tech_rates.keys(), scaled_rates)}

def inputs_vary_general(updated_input_data, scen_level, updates_config, region_groups, cost_matrix_structure):
    '''

    Parameters:
    Returns:

    '''
    # Extract the scenario code
    scen_code = scen_level['scenario']
    sheet_name = 'BCET' 
    updated_input_data[scen_code][sheet_name] = {}
    
    # Load baseline data
    cost_matrix = load_cost_matrix("Inputs/_Masterfiles/FTT-P/FTT-P-22x71_2024_S0.xlsx", ['BCET'], cost_matrix_structure) # GENERALISE
    tech_number = cost_matrix_structure['tech_number']
        

    # Apply updates for technologies and fuels
    for technology, updates in updates_config.items():
        update_cost_matrix(cost_matrix, technology, updates, scen_level)

    # Divide into country sheets and vary discount rates
    for i in range(0, len(cost_matrix), tech_number+1):
        # Create country specific sheet
        country = cost_matrix.loc[i, 'Unnamed: 1']
        country_df = cost_matrix.iloc[i:i + tech_number + 1, :].reset_index(drop=True)

        # Scale and shift to the target range
        #scaled_rates = min_rate + (max_rate - min_rate) * scalar * normalized_rates
        min_rate = 0.01 # generalise to take these from config
        max_rate = 0.2

        # normalise the rates
        country_df.iloc[1:, 17] = country_df.iloc[1:, 17] / max_rate

        # Apply scaling based on regional discount rate
        if country in region_groups['GN']:
            # scale and shift to target range
            country_df.iloc[1:, 17] = min_rate + (max_rate - min_rate) * scen_level['north_discr'] * country_df.iloc[1:, 17]

        else:
            country_df.iloc[1:, 17] = min_rate + (max_rate - min_rate) * scen_level['south_discr'] * country_df.iloc[1:, 17]


        # Columns adjust
        country_df.loc[0] = [''] + country_df.loc[0][1:].astype(str).tolist()   

        # Tech numbers add 
        for index, row in country_df.iloc[1:].iterrows():
            country_df.loc[index, 'Unnamed: 1'] = str(index) + ' ' + country_df.loc[index, 'Unnamed: 1']

        # Make the first row the column headers
        country_df.columns = country_df.iloc[0]
        # Drop the first row now that it is the header
        country_df = country_df[1:].reset_index(drop=True)        
        
        updated_input_data[scen_code][sheet_name][country] = country_df # needs generalising

    return updated_input_data 

def inputs_vary_special(updated_input_data, scen_level, titles): # need to add paths
    '''
    Update the technoeconomic parameters that are not in cost matrix

    Parameters:
    Returns:
    '''
    
    scen_code = scen_level['scenario']  
    sheet_name_1 = 'MEWDX'
    sheet_name_2 = 'MCSC'

    # Initialise dict for new scenario and sheet
    updated_input_data[scen_code][sheet_name_1] = {}
    updated_input_data[scen_code][sheet_name_2] = {} 

    for reg in range(0, len(titles['RTI_short'])):

        ### Electricity demand
        reg_short = titles['RTI_short'][reg]
        reg_long = titles['RTI'][reg]


        mewd_base = pd.read_csv(f'Inputs/S0/FTT-P/MEWDX_{reg_short}.csv')
        # Calculate growth rate
        mewd_grate = mewd_base.iloc[7, 13:].pct_change().fillna(0)
        # Calculate new growth rate based on scenario level
        mewd_grate_new = mewd_grate[1:] * scen_level['elec_demand']
        
        # Calculate & insert new growth rate after 2022 - add in BCET end year from variable_listing
        mewd_updated = mewd_base.copy()


        for i in range(1, len(mewd_updated.iloc[7, 14:])):
            mewd_updated.iloc[7, 13 + i] = round((1 + mewd_grate_new.iloc[i-1]) * mewd_updated.iloc[7, 14 + i], 4) 
                                                                                   
        mewd_updated.rename(columns={mewd_updated.columns[0]: ''}, inplace=True)

        # Export to dict
        updated_input_data[scen_code][sheet_name_1][reg_short] = mewd_updated

        ### Technical potential
        tech_base = pd.read_csv(f'Inputs/S0/General/MCSC_{reg_short}.csv')
        tech_lower = tech_base.iloc[:, 3] * 0.8 # technical potential
        tech_upper = tech_base.iloc[:, 3] * 1.2 # technical potential

        tech_diff = tech_upper - tech_lower
        tech_update = tech_lower + (tech_diff * scen_level['tech_potential']) 

        # update tech potential
        tech_updated = tech_base.copy()
        
        renewables_indices = [9, 10, 11] # indices of renewables in the tech_base df
        tech_updated.loc[renewables_indices, '2'] = tech_update.iloc[renewables_indices].values
        tech_updated.rename(columns={tech_updated.columns[0]: ''}, inplace=True)
        
        updated_input_data[scen_code][sheet_name_2][reg_short] = tech_updated


    return updated_input_data

def save_updated_data(updated_input_data, output_dir, general_vars):
    """
    Save updated input data to new csv files.
    
    Parameters:
    - updated_data: dict, dictionary containing updated input DataFrames
    - output_dir: str, directory to save the updated files
    
    """
    # Loop through updated data and save to new csv files
    for new_scen_code in updated_input_data.keys():
        for sheet_name in updated_input_data[new_scen_code].keys():

            # Filter out vars that are not go in to General folder of inputs
            if sheet_name not in general_vars:
                if sheet_name == 'REPPX': # hardcoded for now, handle via dimensions later
                    
                    df = updated_input_data[new_scen_code][sheet_name]

                    # Create a new directory for the new scenario code
                    new_output_dir = os.path.join(output_dir, new_scen_code, "FTT-P")
                    os.makedirs(new_output_dir, exist_ok=True)
                    output_path = os.path.join(new_output_dir, f"{sheet_name}.csv")
                    
                    df.to_csv(output_path, index=False)

                else:
                    for country in updated_input_data[new_scen_code][sheet_name].keys():
                        
                        df = updated_input_data[new_scen_code][sheet_name][country]
                        
                        # Repeating folder creation, can we remove this?
                        new_output_dir = os.path.join(output_dir, new_scen_code, "FTT-P")
                        os.makedirs(new_output_dir, exist_ok=True)
                        output_path = os.path.join(new_output_dir, f"{sheet_name}_{country}.csv")
                        
                        df.to_csv(output_path, index=False)

            else:
                for country in updated_input_data[new_scen_code][sheet_name].keys():
                    
                    df = updated_input_data[new_scen_code][sheet_name][country]
                    
                    # Create a new directory for the new scenario code
                    new_output_dir = os.path.join(output_dir, new_scen_code, "General")
                    os.makedirs(new_output_dir, exist_ok=True)

                    output_path = os.path.join(new_output_dir, f"{sheet_name}_{country}.csv")
                    df.to_csv(output_path, index=False)



def process_ambition_variation(base_master_path, scen_levels_path, comparison_path, scenarios, 
                               region_groups, params, general_vars, output_dir, cost_matrix_var, 
                               cost_matrix_structure, updates_config, titles, carbon_price_path):
    """
    Process ambition variation by loading and comparing data, updating inputs, and saving the results.
    
    Parameters:
    - base_master_path: str, path to the master file
    - scen_levels_path: str, path to the scenario levels CSV file
    - comparison_path: str, path to the comparison file
    - cp_path: str, path to the carbon price CSV file
    - output_dir: str, directory to save the updated files
    - amb_scenario: str, ambitious scenario identifier
    - region_groups: dict, dictionary containing region groups
    
    Returns:
    - dict, dictionary containing processed data
    """
    # Load comparison data and get sheet names
    compare_data, sheet_names = load_comparison_data(comparison_path)
    

    # Load input data using the sheet names from the comparison file
    input_data = load_input_data(base_master_path, sheet_names)
    cost_matrix = load_input_data(base_master_path, [cost_matrix_var]) # needed when generalised

    #carbon_price_data = pd.read_csv(cp_path)
    scen_levels = pd.read_csv(scen_levels_path)
    scen_levels = scen_levels_extend(scen_levels, region_groups)


    for i in tqdm(range(len(scen_levels['scenario'])), desc="Processing Scenarios", unit="scenario"):
        

        #updated_input_data = defaultdict(lambda: defaultdict(dict))
        # Get the scenario levels for the current iteration and code
        scen_level = scen_levels.iloc[i]
        scen_code = scen_level['scenario']
        
        # Initialise dictory for new scenario
        updated_input_data = {}
        updated_input_data[scen_code] = {}
        
        
        # Update input data based on scenario levels and comparison data
        # TODO COMBINE THESE
        inputs_vary_general(updated_input_data, scen_level, updates_config, region_groups, cost_matrix_structure)
        inputs_vary_special(updated_input_data, scen_level, titles)
        pol_vary_special(updated_input_data, scen_level, carbon_price_path)
        pol_vary_general(updated_input_data, input_data, scen_level, compare_data, scenarios, region_groups, params)


        # Save updated data to new csv files
        save_updated_data(updated_input_data, output_dir, general_vars)
        


        