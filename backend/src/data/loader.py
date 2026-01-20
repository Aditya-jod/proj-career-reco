import os
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"

def load_config(config_path: str | None = None):
    """
    Load configuration from YAML file.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {path}")
        return None

def load_raw_data(config):
    """
    Load raw datasets defined in the config.
    Returns a dictionary containing the dataframes.
    """
    datasets_config = config['datasets']
    
    datasets = {}
    
    try:
        print("Loading datasets...")
        
        # 1. Career Path Data
        datasets['career_path'] = pd.read_csv(datasets_config['career_path'])
        datasets['career_path'].columns = datasets['career_path'].columns.str.strip()
        print(f"Loaded Career Path Data: {datasets['career_path'].shape}")

        # 2. Student/Recommendation Data 
        datasets['student_reco'] = pd.read_csv(datasets_config['student_reco'])
        datasets['student_reco'].columns = datasets['student_reco'].columns.str.strip()
        print(f"Loaded Student Reco Data: {datasets['student_reco'].shape}")
        
        # Load the second student dataset
        try:
            datasets['student_reco_2'] = pd.read_csv(datasets_config['student_reco_2'])
            datasets['student_reco_2'].columns = datasets['student_reco_2'].columns.str.strip()
            print(f"Loaded Student Reco Data 2: {datasets['student_reco_2'].shape}")
        except Exception as e:
            print(f"Warning: Could not load student_reco_2: {e}")
            datasets['student_reco_2'] = None

        # 3. Job Descriptions
        datasets['job_descriptions'] = pd.read_csv(datasets_config['job_descriptions'])
        print(f"Loaded Job Descriptions: {datasets['job_descriptions'].shape}")

        # 4. Colleges & Universities Data
        datasets['indian_colleges'] = pd.read_csv(datasets_config['indian_colleges'])
        datasets['world_universities'] = pd.read_csv(datasets_config['world_universities'])
        
        print("All primary datasets loaded successfully.")
        return datasets
        
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
if __name__ == "__main__":
    config = load_config()
    
    if config:
        data = load_raw_data(config)
        if data:
            print("\nSuccess! Data loaded.")
            print(f"Keys available: {list(data.keys())}")
