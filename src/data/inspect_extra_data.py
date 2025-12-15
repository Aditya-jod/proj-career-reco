import pandas as pd
import yaml
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def inspect_data():
    config = load_config()
    datasets_config = config['datasets']
    
    # List of datasets we haven't looked at closely yet
    targets = ['student_reco', 'indian_colleges', 'world_universities']
    
    for key in targets:
        path = datasets_config[key]
        print(f"\n--- Inspecting {key} ---")
        try:
            df = pd.read_csv(path)
            print(f"Columns: {df.columns.tolist()}")
            print(f"First row: {df.iloc[0].to_dict()}")
        except Exception as e:
            print(f"Error loading {key}: {e}")

if __name__ == "__main__":
    inspect_data()
