import os
import sys

# Set root directory
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
os.chdir(root_dir)

from Emulation.code.simulation_code.utils import load_config
from Emulation.code.simulation_code.scenario_generator_1 import scen_generator


def main():
    config_path = os.path.join((os.path.dirname(os.path.dirname(__file__))), 'config', 'config.json')
    config = load_config(config_path)
    
    regions = config['regions']
    params = config['params']
    ranges = config['ranges']
    Nscens = config['Nscens']
    scen_code = config['scenarios']['ambitious']
    round_decimals = config['round_decimals']
    scen_levels_path = config['scen_levels_path']
    
    combined_df = scen_generator(regions, params, Nscens, scen_code, ranges, round_decimals = 3, output_path=scen_levels_path)
    print(f"Scenarios saved to {scen_levels_path}")

if __name__ == "__main__":
    main()