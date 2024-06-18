"""
===========
idiom_io.py
===========
Scenarios and Assumptions IDIOM file IO script.
###############################################

Library of functions to read and write to standard E3ME IDIOM files, including
"Scenarios" and "Assumptions" files.

Functions included:
    - read_scens_idiom
        Reads Scenarios data from IDIOM file as dictionary of DataFrames.
    - write_scens_idiom
        Writes Scenarios data into IDIOM file format.
    - read_scens_excel
        Reads Scenarios data from (custom) Excel file format as dictionary of DataFrames.
    - write_scens_excel
        Writes Scenarios data to (custom) Excel file format.
    - edit_scens
        Replaces DataFrame for a one or more variables in a Scenarios dictionary.
    - scens_idiom_to_excel
        Converts Scenarios IDIOM file to (custom) Excel file format.
    - scens_excel_to_idiom
        Converts Scenarios Excel file to IDIOM file format.
    - read_asns_idiom
        Reads Assumptions data from IDIOM file format.
    - read_asns_csv
        Reads Assumptions data from CSV file format.
    - write_asns_idiom
        Writes Assumptions data to IDIOM file format.
    - write_asns_csv
        Writes Assumptions data to CSV file format.
    - asns_idiom_to_csv
        Converts Assumptions IDIOM file to CSV file format.
    - asns_csv_to_idiom
        Converts Assumptions CSV file to IDIOM file format.
"""

from copy import deepcopy
import pandas as pd, numpy as np
from celib import DB1

#%%
# --------------------------------------------------------------------------- #
#  SCENARIOS FILES - FUNCTIONS
# --------------------------------------------------------------------------- #
def read_scens_idiom(idiom_fp, dbu_fp, year_range=range(2001,2101), metadata=False):
    """
    Reads Scenarios IDIOM file as dictionary of DataFrames.

    Parameters
    ----------
    idiom_fp : string
        Filepath for Scenarios IDIOM file to read from.
    dbu_fp : string
        Filepath for U databank.
    year_range : iterable, default 2001-2100
        Year range of the Scenarios file (can differ for different E3 models).
    metadata : bool
        Include entry for metadata from IDIOM file.

    Returns
    -------
    data : dictionary of DataFrames
        Data from scenarios IDIOM file - one key-value pair per variable.
    """
    # Read IDIOM file line by line - store as list of strings
    with open(idiom_fp) as file:
        lines = [line.strip() for line in file.readlines()]
        # Store metadata line numbers
        meta_lines = [i for i, line in enumerate(lines) if '"' in line]

    # Parse data & metadata: store in dictionary of DataFrames
    data = {}
    for n, meta_i in enumerate(meta_lines):
        # Parse metadata elements
        meta = lines[meta_i].strip()
        dims, text, data_type = meta.split('"')

        # Get variable name and store metadata
        var = text.split()[0]
#        desc = ' '.join(text.split()[1:])
        data[var] = {}
        data[var]['meta'] = meta

        # Get dimensions from metadata
        dims = dims.split()[2:]
        dims = [dim if dim!='1' else 'R' for dim in dims]

        # Get index of first line of data
        start = meta_i+1

        # Matrix data
        if len(dims) == 2:
            # Get titles from U databank
            with DB1(dbu_fp) as dbu:
                for i, dim in enumerate(dims):
                    dims[i] = dbu[dim+'TI'] if dim!='Y1' else year_range

            # Parse length of data table
            end = meta_lines[n+1] if n < len(meta_lines)-1 else len(lines)

            # Store data table as dataframe
            data[var]['data'] = pd.DataFrame([lines[i].split() for i in range(start, end)],
                                             index=dims[0], columns=dims[1],
                                             dtype=float)
        # Scalar data
        elif len(dims) == 0:
            data[var]['data'] = pd.DataFrame(int(lines[start][0]), index=[0], columns=[0])

    # Option to remove metadata (default)
    if not metadata:
        data = {var: data[var]['data'] for var in data}

    return data


def write_scens_idiom(data, idiom_fp):
    """
    Writes Scenarios data into IDIOM file format.

    Parameters
    ----------
    idiom_fp : string
        Filepath for Scenarios IDIOM file to write to.
    data : dictionary of DataFrames
        Scenarios data to write to IDIOM file.

    Returns
    -------
    None
    """
    # Write new data to IDIOM file
    file_lines = []
    for var in data:
        metadata, df = data[var]['meta'], data[var]['data']

        file_lines.append(metadata+'\n')

        if isinstance(df, pd.DataFrame) :
            df = df.astype(str)
            df = df.replace('0.0', '0')
            df = df.replace('1.0', '1')

            for i in df.index:
                file_lines.append(' '.join(df.loc[i].values.tolist())+'\n')
        else:
            file_lines.append(str(df)+'\n')

    file_lines[-1] = file_lines[-1][:-1]

    with open(idiom_fp, 'w+') as file:
        file.writelines(file_lines)


def read_scens_excel(excel_fp):
    """
    Reads Scenarios data from (custom) Excel file format as dictionary of DataFrames.

    Parameters
    ----------
    excel_fp : string
        Filepath for Scenarios Excel file

    Returns
    -------
    excel_data : dictionary of DataFrames
        Data from scenarios Excel file - one key-value pair per variable
    """
    excel_data = pd.read_excel(excel_fp, sheet_name=None, index_col=0)
#    for var in excel_data:
#        if excel_data[var].shape == (0,0):
#            excel_data[var] = excel_data[var].reset_index().T.reset_index().T.iloc[0,0]
    return excel_data


def write_scens_excel(data, excel_fp):
    """
    Writes Scenarios data into (custom) Excel file format.

    Parameters
    ----------
    data : dictionary of DataFrames
        Scenarios data to write to IDIOM file.
    excel_fp : string
        Filepath for Scenarios IDIOM file to write to.

    Returns
    -------
    None
    """
    with pd.ExcelWriter(excel_fp) as xlsx:
        for var, df in data.items():
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame([df])
            else:
                df = df.reset_index().T.reset_index().T
                df.iloc[0,0] = np.nan

            df.to_excel(xlsx, sheet_name=var, header=None, index=None)


def edit_scens(old_data, new_data):
    """
    Overwrites Scenarios data for one or more variables.

    Parameters
    ----------
    old_data : dictionary of DataFrames
        Old Scenarios data to be overwritten.
    new_data : dictionary of DataFrames
        New Scenarios data with variables to overwrite.

    Returns
    -------
    data : dictionary of DataFrames
        Scenarios data with overwritten variables.
    """
    if type(new_data) != dict:
        raise TypeError('Data must be structured as a dictionary of dataframes/\
                        arrays (with variable short codes as keys).')

    data = deepcopy(old_data)
    # Replace old data with new data
    for var, matrix in new_data.items():
        # Check var name
        if var not in data:
            raise ValueError('Variable short code not found in Scenarios.idiom.')

        # Check dimensions
        old_mat = data[var]['data']
        if matrix.shape != old_mat.shape:
            raise ValueError('Matrix is the wrong shape.')

        # Check data type (must be matrix or DF)
        if isinstance(matrix, np.ndarray):
            matrix = pd.DataFrame(matrix, index=old_mat.index, columns=old_mat.columns)

        if not isinstance(matrix, pd.DataFrame):
            raise ValueError('Data must be stored in DataFrame or numpy array.')

        # Replace data after checks complete
        data[var]['data'] = matrix.copy()

    return data


def scens_idiom_to_excel(idiom_fp, excel_fp, dbu_fp, year_range=range(2001,2101)):
    """
    Converts Scenarios IDIOM file into (custom) Excel file format.

    Parameters
    ----------
    idiom_fp : string
        Filepath for Scenarios IDIOM file to read from.
    excel_fp : string
        Filepath for Scenarios Excel file to write to.
    dbu_fp : string
        Filepath for U databank.
    year_range : iterable, default 2001-2100
        Year range of the Scenarios file (can differ for different E3 models).

    Returns
    -------
    None
    """
    data = read_scens_idiom(idiom_fp, dbu_fp, year_range, metadata=False)
    write_scens_excel(data, excel_fp)


def scens_excel_to_idiom(excel_fp, idiom_meta_fp, dbu_fp,
                         year_range=range(2001,2101), idiom_new_fp=None):
    """
    Converts Scenarios Excel file into IDIOM file format.

    Parameters
    ----------
    excel_fp : string
        Filepath for Scenarios Excel file to read from.
    idiom_meta_fp : string
        Filepath for template Scenarios IDIOM file to get metadata from.
    dbu_fp : string
        Filepath for U databank.
    year_range : iterable, default 2001-2100
        Year range of the Scenarios file (can differ for different E3 models).
    idiom_new_fp : string, default Excel file name
        Filepath for Scenarios IDIOM file to write to.

    Returns
    -------
    None
    """
    if not idiom_new_fp:
        idiom_new_fp = excel_fp[:-5] + '.idiom'

    data = read_scens_excel(excel_fp)
    # Template idiom file required for metadata, which is not stored in Excel file
    template = read_scens_idiom(idiom_meta_fp, dbu_fp, year_range, True)
    data_new = edit_scens(template, data)
    write_scens_idiom(data_new, idiom_new_fp)


#%%
# --------------------------------------------------------------------------- #
#  ASSUMPTIONS FILES - FUNCTIONS
# --------------------------------------------------------------------------- #

def read_asns_idiom(idiom_fp, dbu_fp):
    """
    Reads Assumptions IDIOM file as dictionary of DataFrames.

    Parameters
    ----------
    idiom_fp : string
        Filepath for Assumptions IDIOM file to read from.
    dbu_fp : string
        Filepath for U databank.

    Returns
    -------
    data : dictionary of DataFrames
        Data from Assumptions IDIOM file - one key-value pair per variable.
    """
    with open(idiom_fp) as file:
        lines = file.readlines()
        header_lines = [i for i, line in enumerate(lines) if line[:7]=='01 YEAR']

    with DB1(dbu_fp) as dbu:
        regions = dbu['RTI']

    data = {}
    for n, header_i in enumerate(header_lines):
        header = lines[header_i].split()

        if n==0:
            reg = '0 GLOBAL'
        else:
            reg = regions[n-1].split('(')[0].strip().upper()
            header.remove(header[2])

        start = header_i+1
        end = header_lines[n+1] if n < len(header_lines)-1 else len(lines)
        data_lines = lines[start:end]

        index = [line.split()[1] for line in data_lines]
        cols = [int(float(y)) for y in header[2:]]
        df = pd.DataFrame([line.split()[2:] for line in data_lines],
                          index=index, columns=cols,
                          dtype=float)
        df[df.round()==df] = df.astype(int)

        data[reg] = df.copy()

    return data


def read_asns_csv(csv_fp):
    """
    Reads Assumptions CSV file as dictionary of DataFrames.

    Parameters
    ----------
    csv_fp : string
        Filepath for Assumptions CSV file to read from.

    Returns
    -------
    data : dictionary of DataFrames
        Data from Assumptions CSV file - one key-value pair per variable.
    """
    df = pd.read_csv(csv_fp, header=None)

    header_lines = df[df.iloc[:, 0].str[0].str.isnumeric()].index.tolist()

    data = {}
    for n, header_i in enumerate(header_lines):
        reg = df.iloc[header_i, 0]

        try:
            header_i_next = header_lines[n+1]
            data[reg] = df.iloc[header_i:header_i_next]
        except IndexError:
            data[reg] = df.iloc[header_i:]

        data[reg] = data[reg].set_index(0).T.set_index(reg).T
        data[reg].columns = map(int, data[reg].columns)

    return data


def write_asns_idiom(data, idiom_fp, dbu_fp):
    """
    Writes Assumptions data into IDIOM file format.

    Parameters
    ----------
    data : dictionary of DataFrames
        Assumptions data to write to IDIOM file.
    idiom_fp : string
        Filepath for Assumptions IDIOM file to write to.
    dbu_fp : string
        Filepath for U databank.

    Returns
    -------
    None
    """
    data = deepcopy(data)

    with DB1(dbu_fp) as dbu:
        regions = dbu['RSHORTTI']

    lines = []
    for n, df in enumerate(data.values()):
        df = df.reset_index().T.reset_index().T.reset_index(drop=True)
        if n == 0:
            df.iloc[0,0] = 'YEAR'
        else:
            df.iloc[0,0] = 'YEAR      ' + regions[n-1]
        df[0] = pd.Series("{:02d} ".format(n+1) for n in range(len(df))) + df[0]
        df[0] = df[0].str.pad(width=22, side='right')

        df.iloc[:, 1:] = df.iloc[:, 1:].astype(float).applymap('{:.3f}'.format)
        df = df.astype(str)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.str.pad(width=10))

        for line in df.values.tolist():
            line = ''.join(line) + '\n'
            lines.append(line)

    with open(idiom_fp, 'w+') as file:
        file.writelines(lines)


def write_asns_csv(data, csv_fp):
    """
    Writes Assumptions data into CSV file format.

    Parameters
    ----------
    data : dictionary of DataFrames
        Assumptions data to write to CSV file.
    idiom_fp : string
        Filepath for Assumptions CSV file to write to.

    Returns
    -------
    None
    """
    data = deepcopy(data)
    for n, (var, df) in enumerate(data.items()):
        df = df.reset_index().T.reset_index().T
        df.iloc[0,0] = var

        data[var] = df.copy()

    data = pd.concat((data[var] for var in data))

    data.to_csv(csv_fp, index=None, header=None)

    # Remove unnecessary decimal places
    with open(csv_fp) as file:
        lines = [line.replace('.0,', ',').replace('.0\n', '\n') for line in file.readlines()]

    with open(csv_fp, 'w+') as file:
        file.writelines(lines)


def asns_idiom_to_csv(idiom_fp, csv_fp, dbu_fp):
    """
    Converts Assumptions IDIOM file into CSV file format.

    Parameters
    ----------
    idiom_fp : string
        Filepath for Assumptions IDIOM file to read from.
    csv_fp : string
        Filepath for Assumptions CSV file to write to.
    dbu_fp : string
        Filepath for U databank.

    Returns
    -------
    None
    """
    data = read_asns_idiom(idiom_fp, dbu_fp)
    write_asns_csv(data, csv_fp)


def asns_csv_to_idiom(csv_fp, idiom_fp, dbu_fp):
    """
    Converts Assumptions CSV file into IDIOM file format.

    Parameters
    ----------
    csv_fp : string
        Filepath for Assumptions CSV file to read from.
    idiom_fp : string
        Filepath for Assumptions IDIOM file to write to.
    dbu_fp : string
        Filepath for U databank.

    Returns
    -------
    None
    """
    data = read_asns_csv(csv_fp)
    write_asns_idiom(data, idiom_fp, dbu_fp)


#%%
# ----------------------------------------------------------------------- #
#  DEMO
# ----------------------------------------------------------------------- #
if __name__ == '__main__':
    e3me_dir = 'C://E3ME/Kyoto_NEW/'
    dbu_fp = e3me_dir+'databank/U.db1'
    years = range(2001,2101)

    #%%
    # ----------------------------------------------------------------------- #
    #  SCENARIOS FILES - DEMO
    # ----------------------------------------------------------------------- #
    # READING
    # Read in exogenous carbon price assumptions
    input_filepath = e3me_dir + 'In/Carbon price calculation.xlsx'
    carbon_prices = pd.read_excel(input_filepath, sheet_name='nat_scen_input_v2', index_col=0)
    
    
    # Read in Scenarios data from IDIOM file
    scens_old_idiom_fp = e3me_dir+'In/Scenarios/B_ETS.idiom'
    scens_old_data = read_scens_idiom(scens_old_idiom_fp, dbu_fp, years, metadata=True)


    # EDITING IN PYTHON
    ## Edit Scenarios data in Python
    variables = ['REPX']
    scens_new_data = {var: scens_old_data[var]['data'].copy() for var in variables}
    
    # Find common countries / years
    common_indexes = carbon_prices.index.intersection(scens_new_data['REPX'].index)
    common_columns = carbon_prices.columns.intersection(scens_new_data['REPX'].columns)
    
    # Copy the values from 'carbon_prices' to 'REPX' for common indexes and columns
    for index in common_indexes:
        for column in common_columns:
            scens_new_data['REPX'].at[index, column] = carbon_prices.at[index, column]
    
    scens_new_data = edit_scens(scens_old_data, scens_new_data)
    ## Write new data to IDIOM file
    scens_new_idiom_fp = e3me_dir+'In/Scenarios/B_ETS_nat_v2.idiom'
    write_scens_idiom(scens_new_data, scens_new_idiom_fp)


    # EDITING IN EXCEL
    ## Convert a Scenarios IDIOM file to Excel
    scens_excel_fp = e3me_dir+'In/Scenarios/B_Edit_in_Excel.xlsx'
    scens_idiom_to_excel(scens_old_idiom_fp, scens_excel_fp, dbu_fp, years)

    ## (Edit Excel file manually...)

    ## Convert back to IDIOM format
    scens_excel_to_idiom(scens_excel_fp, scens_old_idiom_fp, dbu_fp, years)


    #%%
    # ----------------------------------------------------------------------- #
    #  ASSUMPTIONS FILES - DEMO
    # ----------------------------------------------------------------------- #
    # READING
    ## Read in an Assumptions IDIOM file
    asns_idiom_fp = e3me_dir+'In/Asns/Assumptions.idiom'
    asns_data_idiom = read_asns_idiom(asns_idiom_fp, dbu_fp)

    ## Read in an Assumptions CSV file
    asns_csv_fp = e3me_dir+'In/Asns/Assumptions.csv'
    asns_data_csv = read_asns_csv(asns_csv_fp)


    # WRITING
    ## Write Assumptions data to a CSV file
    asns_csv_new_fp = e3me_dir+'In/Asns/Assumptions_new.csv'
    write_asns_csv(asns_data_idiom, asns_csv_new_fp)

    ## Write Assumptions data to an IDIOM file
    asns_idiom_new_fp = e3me_dir+'In/Asns/Assumptions_new.idiom'
    write_asns_idiom(asns_data_idiom, asns_idiom_new_fp, dbu_fp)


    # CONVERTING (i.e. read then write)
    ## Convert an Assumptions CSV file to IDIOM
    asns_idiom_convert_fp = e3me_dir+'In/Asns/Assumptions_convert.idiom'
    asns_csv_to_idiom(asns_csv_fp, asns_idiom_convert_fp, dbu_fp)

    ## Convert an Assumptions IDIOM file to CSV
    asns_csv_convert_fp = e3me_dir+'In/Asns/Assumptions_convert.csv'
    asns_idiom_to_csv(asns_idiom_fp, asns_csv_convert_fp, dbu_fp)
