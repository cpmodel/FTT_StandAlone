import pandas as pd
import os

def load_and_preprocess(file_path, sheets_to_process):
    """
    Load and preprocess the Excel file.
    
    Parameters:
    - file_path: str, path to the Excel file
    - sheets_to_process: list of str, sheet names to process
    
    Returns:
    - dict, dictionary containing preprocessed DataFrames for each sheet
    """
    dataframes = {}
    for sheet_name in sheets_to_process:
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=3, usecols=lambda column: column not in range(22, 26))
        
        # Extract every 34th row from the column
        numb_col = df.iloc[:, 0].iloc[::34].reset_index(drop=True)
        country_col = df.iloc[:, 1].iloc[::34].reset_index(drop=True)
        
        # Repeat the extracted values for the next 34 rows
        numb_col = numb_col.repeat(34).reset_index(drop=True)
        country_col = country_col.repeat(34).reset_index(drop=True)
        
        # Insert the new column as the first 2 columns in the DataFrame
        df.iloc[:, 0] = numb_col
        df.insert(1, 'Country', country_col)
        
        # Delete the rows that were used to create the new column
        rows_to_delete = df.index[34::34]
        df = df.drop(rows_to_delete)
        
        # Set the first row as column names
        df.columns = df.iloc[0]
        df.columns.values[0:3] = ['Code', 'Country', 'Technology']
        
        # Delete the first row from the DataFrame
        df = df.iloc[1:].reset_index(drop=True)
        
        dataframes[sheet_name] = df
    
    return dataframes

def compare_scenarios(file1_path, file2_path, sheets_to_process, keep_equal=False):
    """
    Compare two scenario files to check for differences.
    
    Parameters:
    - file1_path: str, path to the first scenario file
    - file2_path: str, path to the second scenario file
    - sheets_to_process: list of str, sheet names to process
    - keep_equal: bool, whether to keep equal values in the comparison
    
    Returns:
    - dict, dictionary containing the comparison results for each sheet
    """
    df1 = load_and_preprocess(file1_path, sheets_to_process)
    df2 = load_and_preprocess(file2_path, sheets_to_process)
    
    comparison_results = {}
    for sheet_name in sheets_to_process:
        comparison = df1[sheet_name].compare(df2[sheet_name], align_axis=0, result_names=('S0', 'S3'), keep_equal=keep_equal)
        
        if not comparison.empty:
            comparison = comparison.reset_index(level=1)
            comparison.columns = ['Scenario'] + list(comparison.columns[1:])
            comparison.insert(1, 'Sheet', sheet_name)
            comparison.insert(2, 'Code', df1[sheet_name]['Code'])
            if 'Country' not in comparison.columns:
                comparison.insert(3, 'Country', df1[sheet_name]['Country'])
            comparison.insert(4, 'Technology', df1[sheet_name]['Technology'])
            
            comparison_results[sheet_name] = comparison
    
    return comparison_results

def export_compare(compare_output, output_file):
    """
    Export the comparison results to an Excel file.
    
    Parameters:
    - compare_output: dict, dictionary containing the comparison results for each sheet
    - output_file: str, path to the output Excel file
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in compare_output.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)