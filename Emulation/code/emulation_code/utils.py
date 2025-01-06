import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Replace placeholders with actual variable names
    scen_code = config.get('scen_code', 'default')
    # config['scen_levels_path'] = config['scen_levels_path'].format(scen_code=scen_code)
    # config['comparison_path'] = config['comparison_output_path'].format(scen_code=scen_code)


    return config