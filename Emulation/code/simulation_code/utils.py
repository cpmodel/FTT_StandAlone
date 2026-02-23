import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)


    return config