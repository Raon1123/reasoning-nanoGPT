import yaml

def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)