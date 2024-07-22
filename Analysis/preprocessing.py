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
    
    
    # Attempt to import again
    try:
        from SourceCode.support.titles_functions import load_titles
        print("Import successful")
    except ModuleNotFoundError as e:
        print(f"Import failed: {e}")
        # Troubleshooting step 4: Use an absolute path for verification
        sys.path.append(r"C:\Users\fjmn202\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone_laptop_repos\FTT_StandAlone\SourceCode")
        sys.path.append(r"C:\Users\fjmn202\OneDrive - University of Exeter\Documents\GitHub\FTT_StandAlone\FTT_StandAlone\SourceCode")
        from support.titles_functions import load_titles
    # Local library imports
    
    # Import classification titles from utilities
    titles = load_titles()
    
    project_root_absolute_path = define_path()
    
    fig_dir = os.path.join(project_root_absolute_path, "Output", "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Get names of the technologies of interest
    tech_titles = {"FTT:P": "T2TI", "FTT:Tr": "VTTI", "FTT:Fr": "FTTI", "FTT:H": "HTTI"}    
    
    models = ["FTT:P", "FTT:H", "FTT:Tr", "FTT:Fr"]

    
    return titles, fig_dir, tech_titles, models
    


def get_output(name_pickle_file, scenario):
    
    project_root_absolute_path = define_path()
    
    pickle_path = os.path.join(project_root_absolute_path, 'Output', name_pickle_file) 
    
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    
    # Extract the results for scenario (f.i. S0)
    output = results[scenario]   
    
    
    return output