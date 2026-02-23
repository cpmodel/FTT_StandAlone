#%%
import os
import pandas as pd
import sys

# Set root directory
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
os.chdir(root_dir)

from Emulation.code.simulation_code.utils import load_config
from Emulation.code.simulation_code.scenario_generator_1 import scen_generator
from Emulation.code.simulation_code.compare_scenarios_2 import compare_scenarios, export_compare
from Emulation.code.simulation_code.ambition_vary_3 import process_ambition_variation
from Emulation.code.simulation_code.run_emulation_5 import main as run_simulations
from Emulation.code.simulation_code.output_manipulation_6 import main as output_mainuplation

from SourceCode.support.titles_functions import load_titles
titles = load_titles()

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')
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
    params = config['params']
    ranges = config['ranges']
    Nscens = config['Nscens']
    regions = config['regions']
    region_groups = config['region_groups']
    params = config['params']
    general_vars = config['general_vars']
    cost_matrix = config['cost_matrix_var']
    cost_matrix_structure = config['cost_matrix_structure']
    updates_config = config['updates_config']
    scens_run = config['scens_run']
    round_decimals = config['round_decimals']
    output_path = config['output_path']


    run_compare_scenarios = config.get('run_compare_scenarios', True)
    run_generate_scenarios = config.get('run_generate_scenarios', True)
    run_ambition_vary = config.get('run_ambition_vary', True)
    run_run_simulations = config.get('run_run_simulations', True)
    run_output_manipulation = config.get('run_output_manipulation', True)


    # Compare scenarios
    if run_compare_scenarios:
        comparison_results = compare_scenarios(base_master_path, amb_master_path, vars_to_compare)
        export_compare(comparison_results, comparison_path)
        print(f"Comparison results saved to {comparison_path}")
        
    if run_generate_scenarios:
        # Generate scenarios with ambitious scen code
        scen_code = scenarios['ambitious']
        scen_generator(regions, params, Nscens, scen_code, ranges, round_decimals = 3, output_path=scen_levels_path)
        print(f"Scenarios saved to {scen_levels_path}")
    
    # Process ambition variation
    if run_ambition_vary:
        process_ambition_variation(base_master_path, scen_levels_path, 
                                                    comparison_path, scenarios, region_groups, 
                                                    params, general_vars, output_dir,
                                                    cost_matrix, cost_matrix_structure, 
                                                    updates_config, titles, carbon_price_path, scens_run)
        print("Processed Ambition Variation Data")

    # Run simulations with data varied between boundary scenarios
    if run_run_simulations:
        run_simulations(scens_run)
        print("Simulations completed")

    # Manipulate the output data into csv files for emulation
    if run_output_manipulation:
        output_mainuplation(scen_levels_path, output_path, scens_run)
        print(f"Output processed and saved to {output_path}")
    

        
#%%

if __name__ == "__main__":
    main()

