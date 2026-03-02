import yaml

def load_params():
    with open("params.yaml", "rb") as f:
        return yaml.safe_load(f)