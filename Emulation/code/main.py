import os
import pandas as pd
import sys

# Set root directory
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)

from emulation_code.utils import load_config
from emulation_code.compare_scenarios_2 import compare_scenarios, export_compare
from emulation_code.ambition_vary_3 import process_ambition_variation
from SourceCode.support.titles_functions import load_titles
titles = load_titles()

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
    config = load_config(config_path)
    
    # Load variables from config
    base_master_path = config['base_master_path']
    amb_master_path = config['ambitious_master_path']
    scen_levels_path = config['scen_levels_path']
    comparison_path = config['comparison_path']
    carbon_price_path = config['carbon_price_path']
    output_dir = config['output_dir']
    vars_to_compare = config['vars_to_compare']
    scenarios = config['scenarios']
    amb_scenario = scenarios['ambitious']
    base_scenario = scenarios['base']
    region_groups = config['region_groups']
    params = config['params']
    general_vars = config['general_vars']
    cost_matrix = config['cost_matrix_var']
    cost_matrix_structure = config['cost_matrix_structure']
    updates_config = config['updates_config']
    
    run_compare_scenarios = config.get('run_compare_scenarios', True)
    run_ambition_vary = config.get('run_ambition_vary', True)
    

    # Compare scenarios
    if run_compare_scenarios:
        comparison_results = compare_scenarios(base_master_path, amb_master_path, vars_to_compare)
        export_compare(comparison_results, comparison_path)
        print(f"Comparison results saved to {comparison_path}")
        
    
    # Process ambition variation
    if run_ambition_vary:
        processed_data = process_ambition_variation(base_master_path, scen_levels_path, 
                                                    comparison_path, scenarios, region_groups, 
                                                    params, general_vars, output_dir,
                                                    cost_matrix, cost_matrix_structure, 
                                                    updates_config, titles, carbon_price_path)
        print("Processed Ambition Variation Data")
        
#%%
if __name__ == "__main__":
    main()

