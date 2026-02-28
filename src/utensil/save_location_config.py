import yaml

def save_location_config(target_loc, key_name, file_path):
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file) or {}
    
    if target_loc not in config:
        config[target_loc] = {}

    config[target_loc][key_name] = file_path

    with open('config.yaml', 'w') as file:
        yaml.safe_dump(config, file)
    
    return