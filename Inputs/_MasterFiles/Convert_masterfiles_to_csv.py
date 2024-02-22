# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 17:10:10 2019

This script extracts all the data from FTT excel sheets in the
"/In/FTTAssumptions/[model]" folders and and saves them in separate csv files.

The user can select one or more scenarios to convert from the excel sheet.

@author: MM and Femke Nijsse
"""

from pathlib import Path
import os

import pandas as pd
import numpy as np

from celib import DB1


# Structure of dict:
# Model name: [Model name, scenarios to read in, excel file]
# Scenario 0 = Baseline
# Scenario 1 = 2-degree scenario (default)
# Scenario 2 = 1.5-degree scenario (default)
# ENTER SCENARIO NUMBERS HERE! This will dictate which sheets are read in.


models = {'FTT-Tr': [[0, 2], 'FTT-Tr_31x71_2023']}
        #  'FTT-P': [[0], 'FTT-P-24x71_2022'],
        #  'FTT-H': [[0], 'FTT-H-13x70_2021'],
        #  'FTT-S': [[0], 'FTT-S-26x70_2021']}

# models = {'FTT-IH-CHI': [[0], 'FTT-IH-CHI-13x70_2022'],
#           'FTT-IH-FBT': [[0], 'FTT-IH-FBT-13x70_2022'],
#           'FTT-IH-MTM': [[0], 'FTT-IH-MTM-13x70_2022'],
#           'FTT-IH-NMM': [[0], 'FTT-IH-NMM-13x70_2022'],
#           'FTT-IH-OIS': [[0], 'FTT-IH-OIS-13x70_2022'],
# }

#%%

def get_sheets_to_convert(var_dict, model, scen):
    """ Get all the variables to convert to CSV files
    There are three options:
        First one (TODO) is to check which files don't exist, and make these
        Second option is to do all the S0 files
        Third option is for other scenarios, for which variables are sorted out
        based on whether they are different from S0 to S2
    """
    # In the baseline scenario, all variables are converted
    if scen == 0:
        vars_to_convert = [var for var in var_dict[model] if var_dict[model][var]['Read in?']]
    # In the other scenarios, only policy va
    else:
        vars_to_convert = [var for var in var_dict[model] if \
                       (var_dict[model][var]['Scenario'] == "All" and var_dict[model][var]['Read in?'])]

    sheets = ['Titles'] + vars_to_convert

    return vars_to_convert, sheets

def set_up_rows(model, var, var_dict):
    """Setting up the size of the rows, and the name of the rows"""

    rdim = len(dims[var_dict[model][var]['Dims'][0]])
    row_title = dims[var_dict[model][var]['Dims'][0]]

    return rdim, row_title

def set_up_cols(model, var, var_dict):
    """Setting up the size of the column, and the name of the column"""

    # If there is only a single column / scalar
    if len(var_dict[model][var]['Dims']) == 1:
        cdim = 1
        col_title = ['NA']
    # If there are multiple columns and the column isn' time
    elif var_dict[model][var]['Dims'][1] != 'TIME':
        cdim = len(dims[var_dict[model][var]['Dims'][1]])
        col_title = dims[var_dict[model][var]['Dims'][1]]
    # If the second dimension is time
    else:
        cdim = len(timeline_dict[var])
        try:
            col_title = timeline_dict[var]
        except KeyError as e:
            print(f"KeyError: {e}. Variable '{var}' not found in timeline_dict.")
            col_title = []

    return cdim, col_title

def costs_to_gam(data, var, reg, scen):
    """
    In FTT:Tr, the gamma values are not saved separately, but instead
    part of the BTTC variable. Here, those values are extracted to ensure the
    gamma values are defined for each year.
    
    """
    # Only create gamma sheets for the first scenario
    if scen != 0:
        return

    costvar_to_gam_dict = {"BTTC": "TGAM", "BHTC": "HGAM", "ZCET": "ZGAM"}
    gamma_index = {"BTTC": 14, "BHTC": 13, "ZCET": 14}
    gamma_var = costvar_to_gam_dict[var]
    gamma_1D = data[gamma_index[var]]
    col_names = timeline_dict[gamma_var]

    # Make data 2D with np.tile
    data = pd.DataFrame(np.tile(gamma_1D.values.T, (len(col_names), 1)).T)

    # Add column names
    data.columns = col_names

    out_fn = os.path.join(out_dir, f"{gamma_var}_{reg}.csv")
    df = pd.DataFrame(data.values, index=dims["VTTI"], columns=col_names)
    df.to_csv(out_fn)


def convert_1D_var_to_timeline(data, var, row_title, scen):
    """
    Some variables (e.g. TEWW, MEWW), are 1D in the excel sheets. 
    However, in the model, these variables change over time, 
    and we store this data. Therefore, the csv files should have a time dimension
    """
    # Only create csv files for the first scenario
    if scen != 0:
        return

    # Make data 2D with np.tile
    col_names = timeline_dict[var]
    data = pd.DataFrame(np.tile(data.values.T, (len(col_names), 1)).T)

    # Add column names
    data.columns = col_names

    out_fn = os.path.join(out_dir, f"{var}.csv")
    df = pd.DataFrame(data.values, index=row_title, columns=col_names)
    df.to_csv(out_fn)

    print(f"Data for {var} saved to CSV. Model: {model}")


if __name__ == '__main__':
### ----------------------------------------------------------------------- ###
### -------------------------- VARIABLE SETUP ----------------------------- ###
### ----------------------------------------------------------------------- ###
    # Define paths, directories and subfolders
    dirp = os.path.dirname(os.path.realpath(__file__))
    dirp_up = Path(dirp).parents[0]
    dirp_db = os.path.join(dirp, 'databank')

    # Time horizons, as defined in the Time Horizons sheet
    time_horizon_df = pd.read_excel(os.path.join(dirp, 'FTT_variables.xlsx'),
                                    sheet_name='Time_Horizons')

    timeline_dict = {}
    for i, var in enumerate(time_horizon_df['Variable name']):
        timeline_string = time_horizon_df.loc[i, 'Time horizon']
        start_year = int(timeline_string[-4:]) # The last 4 characters in the string are the start year
        timeline_dict[var] = list(range(start_year, 2100+1))


    # Dict to collect errors
    errors = {}

    # Dict to collect all data
    var_dict = {}

    # Which variables are converted?
    vars_to_convert = {}

    # Loop over all FTT models of interest
    for model in models.keys():
        scenarios = models[model][0]
        errors[model] = []
        var_dict[model] = {}

        # Get variable dimensions/attributes
        variables_df = pd.read_excel(os.path.join(dirp, 'FTT_variables.xlsx'),
                                     sheet_name=model,
                                     true_values='y',
                                     false_values='n',
                                     na_values='-')

        for i, var in enumerate(variables_df['Variable name']):
            var_dict[model][var] = {}
            var_dict[model][var]['Code'] = int(variables_df.loc[i, 'Code'])
            var_dict[model][var]['Desc'] = variables_df.loc[i, 'Description']
            var_dict[model][var]['Dims'] = [variables_df.loc[i, attr]
                                    for attr in ['RowDim', 'ColDim', '3DDim']
                                    if variables_df.loc[i, attr] not in [0, np.nan]]
            var_dict[model][var]['Read in?'] = variables_df.loc[i, 'Read in?']
            var_dict[model][var]["Conversion?"] = variables_df.loc[i, 'Conversion?']

            # Some variables are the same across scenarios, and should only
            # printed for the S0 scenario to save space
            var_dict[model][var]["Scenario"] = variables_df.loc[i, "Scenario"]
            var_dict[model][var]['Data'] = {}


        # Get model classifications
        dims = list(pd.concat([variables_df['RowDim'], variables_df['ColDim'], variables_df['3DDim']]))
        dims = list(set([dim for dim in dims if dim not in ['TIME', np.nan, 0]]))
        dims = {dim: None for dim in dims}
        with DB1(os.path.join(dirp_db, 'U.db1')) as db1:
            for dim in dims:
                dims[dim] = db1[dim]

        #%%
        ### ----------------------------------------------------------------------- ###
        ### ---------------------------- EXTRACT DATA ----------------------------- ###
        ### ----------------------------------------------------------------------- ###
   

        for scen in scenarios:

            # Define which sheets to convert
            vars_to_convert[model], sheets = \
                        get_sheets_to_convert(var_dict, model, scen)

            out_dir = os.path.join(dirp_up, f'S{scen}', model)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Check whether the excel files exist
            raw_f = models[model][1]
            raw_p = os.path.join(dirp, model, f'{raw_f}_S{scen}.xlsx')
            if not os.path.exists(raw_p):
                print(f"{raw_f} does not exists. No csv files will be created for {model} variables")
                continue

            # Tell the user that the file is being read in.
            print(f"Extracting {model} variables of scenario {scen} from the excelsheets")

            # Load sheets
            raw_data = pd.read_excel(raw_p, sheet_name=sheets, header=None)

            # Get titles from the Titles sheet in the excel file
            raw_titles = raw_data['Titles']
            ftt_titles = {}
            for col in range(1, raw_titles.shape[1], 2):
                var = raw_titles.iloc[0, col]
                titles = list(raw_titles.iloc[:, col].dropna())[1:]
                ftt_titles[var] = titles

            # Extract data & save to relevant variable
            row_start = 5
            regs = dims['RSHORTTI']

            ## Read in sheet by sheet
            ci = 2              # Column indenting
            for i, var in enumerate(vars_to_convert[model]):

                ndims = len(var_dict[model][var]['Dims'])
                rdim, row_title = set_up_rows(model, var, var_dict)
                cdim, col_title = set_up_cols(model, var, var_dict)

                excel_dim = len(ftt_titles[var_dict[model][var]['Dims'][0]])
                cf = ci + cdim                   # Final column
                sep = 1 + excel_dim - rdim
                sheet_name = var

                if ndims == 3:
                    var_dict[model][var]['Data'][scen] = {}
                    for i, reg in enumerate(regs):
                        ri = row_start + i*(rdim + sep)
                        rf = ri + rdim
                        data = raw_data[sheet_name].iloc[ri:rf, ci:cf]
                        var_dict[model][var]['Data'][scen][reg] = np.array(data.astype(np.float32))

                        out_fn = os.path.join(out_dir, f"{var}_{reg}.csv")
                        df = pd.DataFrame(data.values, index=row_title, columns=col_title)
                        df.to_csv(out_fn)

                        # Extract the gamma values from BTTC
                        if var_dict[model][var]["Conversion?"] == "GAMMA":
                            #print(data)
                            costs_to_gam(data, var, reg, scen)

                elif ndims == 2:
                    ri = row_start
                    rf = ri + rdim
                    data = raw_data[sheet_name].iloc[ri:rf, ci:cf]
                    var_dict[model][var]['Data'][scen] = np.array(data.astype(np.float32))


                    # Some variables have regions as second dimension in masterfile
                    # Transpose those
                    needs_transposing = variables_df.loc[variables_df["Variable name"] == var]["ColDim"] == "RSHORTTI"
                    if needs_transposing.item():
                        print(f'For var {var}, transposing the two dimensions so that RTI first')
                        col_title, row_title = row_title, col_title
                        data = data.T

                    out_fn = os.path.join(out_dir, f"{var}.csv")
                    df = pd.DataFrame(data.values, index=row_title, columns=col_title)
                    df.to_csv(out_fn)

                elif ndims == 1:

                    ri = row_start
                    rf = ri + rdim
                    data = raw_data[sheet_name].iloc[ri:rf, ci:cf]

                    # If a 1D variable needs to be converted into 2D
                    if var_dict[model][var]["Conversion?"] == "TIME":
                        convert_1D_var_to_timeline(data, var, row_title, scen)
                        continue  # continue to the next variable

                    var_dict[model][var]['Data'][scen] = np.array(data.astype(np.float32))
                    out_fn = os.path.join(out_dir, f"{var}.csv")
                    df = pd.DataFrame(data.values, index=row_title, columns=col_title)
                    df.to_csv(out_fn)

                print(f"Data for {var} saved to CSV. Model: {model}")
