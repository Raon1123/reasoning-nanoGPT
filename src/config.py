import os
import yaml
import argparse
from typing import Dict, Any

def load_config(config_path: str = None, cli_args: list = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and override with CLI arguments.
    Supports hierarchical configuration and dot-notation CLI args (e.g. --model.n_layer).
    """
    config = {}
    
    # Default config path
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')

    # Load from YAML
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found. Using defaults/CLI only.")

    # Parse CLI arguments to override config
    parser = argparse.ArgumentParser(description='NanoGPT Training')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    # Helper to flatten config for CLI arg generation (optional, but good for help)
    # For now, we allow any --key.subkey value
    
    args, unknown = parser.parse_known_args(cli_args)
    
    # Process unknown args which should be in --key value or --key=value format
    # We expect --section.key value
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith('--'):
            key = arg[2:]
            value = None
            if '=' in key:
                key, value = key.split('=', 1)
            else:
                if i + 1 < len(unknown) and not unknown[i+1].startswith('--'):
                    value = unknown[i+1]
                    i += 1
                else:
                    value = 'true' # Assume boolean flag if no value follows
            
            # Handle type conversion (simple)
            if value.lower() == 'true': value = True
            elif value.lower() == 'false': value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass # keep as string
            
            # Update config with dot notation
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            
        i += 1

    return config

def get_default_config_path():
    return os.path.join(os.path.dirname(__file__), '../config/config.yaml')
