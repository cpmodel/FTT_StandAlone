# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:19:15 2024

@author: Femke Nijsse
"""

# Import the results pickle file
import pickle
import os
import sys

def define_path():
    # Assuming your script is in a subdirectory of the project root, adjust the relative path accordingly.
    # For example, if your script is in 'src/scripts', you might use '../..' to reach the project root.
    project_root_relative_path = '..'  # Adjust this path to match your project structure
    
    # Get the directory of the current script & the path to the pickle file
    script_dir = os.path.dirname(__file__)
    
    # Calculate the absolute path to the project root
    project_root_absolute_path = os.path.abspath(os.path.join(script_dir, project_root_relative_path))
    
    # Change the current working directory to the project root
    os.chdir(project_root_absolute_path)
    
    # # Add the SourceCode directory to Python's search path
    # source_code_path = os.path.join(project_root_absolute_path, 'SourceCode')
    # sys.path.append(source_code_path)
    
    return project_root_absolute_path



def get_metadata():
    """Get the classification by variable, the directory to print figures and the 
    names of the technology classification in each sector"""
    
    # Print the current working directory for debugging
    print("Current working directory:", os.getcwd())
    
    # Attempt to import
    try:
        from SourceCode.support.titles_functions import load_titles
        print("Import successful")
    except ModuleNotFoundError as e:
        # Troubleshooting: Use an absolute path
        additional_paths = [
            r"C:\Users\fjmn202\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone\FTT_StandAlone\SourceCode",
            r"C:\Users\fjmn202\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone_laptop_repos\FTT_StandAlone\SourceCode",
            r"C:\Users\Work profile\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone\SourceCode"
        ]
        for path in additional_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
       
        # Verify the existence of the file
        for path in additional_paths:
            full_path = os.path.join(path, 'support', 'titles_functions.py')
            if os.path.exists(full_path):
                print(f"File exists: {full_path}")
            else:
                print(f"File does not exist: {full_path}")
        
        try:
            from support.titles_functions import load_titles
        except ModuleNotFoundError as e:
            print(f"Import failed again: {e}")
            #raise e
            
    
    
    
    # Import classification titles from utilities
    titles = load_titles()
    
    project_root_absolute_path = define_path()
    
    fig_dir = os.path.join(project_root_absolute_path, "Output", "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Get names of the technologies of interest
    tech_titles = {"FTT:P": "T2TI", "FTT:Tr": "VTTI", "FTT:Fr": "FTTI", "FTT:H": "HTTI"}    
    models = ["FTT:P", "FTT:H", "FTT:Tr", "FTT:Fr"]
    shares_vars = {"FTT:P": "MEWG", "FTT:Tr": "TEWK", "FTT:Fr": "ZEWK", "FTT:H": "HEWG"} 

    return titles, fig_dir, tech_titles, models, shares_vars
    


def get_output(name_pickle_file, scenario):
    
    project_root_absolute_path = define_path()
    
    pickle_path = os.path.join(project_root_absolute_path, 'Output', name_pickle_file) 
    
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    
    # Extract the results for scenario (f.i. S0)
    output = results[scenario]   
    
    
    return output

def save_fig(fig, fig_dir, title):
    '''This function saves the figures in both svg and png format, at 300 dpi. '''
    
    # Save the graph as an editable svg file
    output_file = os.path.join(fig_dir, f'{title}.svg')
    output_file2 = os.path.join(fig_dir, f'{title}.png')
    output_file3 = os.path.join(fig_dir, f'{title}.pdf')

    fig.savefig(output_file, format="svg", bbox_inches='tight')
    fig.savefig(output_file2, format="png", bbox_inches='tight')
    fig.savefig(output_file3, format="pdf", bbox_inches='tight')

    
def save_data(df, fig_dir, title):
    '''Saved the dataframe with all the data for the figure'''
    
    # Go one directory back and add "Figures_data"
    data_dir = os.path.join(os.path.dirname(fig_dir), "Figures_data")
    
    # Create the new directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Specify the full file path
    file_path = os.path.join(data_dir, f"{title}.csv")
    
    # Save the DataFrame to the new CSV file
    df.to_csv(file_path, index=False)
    
    