import pandas as pd
import yaml
import os

def load_config(config_path="config/config.yaml"):
    """
    Load configuration from YAML file.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return None

def load_raw_data(config):
    """
    Load raw datasets defined in the config.
    Returns a dictionary containing the dataframes.
    """
    datasets_config = config['datasets']
    
    datasets = {}
    
    try:
        # Load datasets using absolute paths from config
        print("Loading datasets...")
        
        # 1. Career Path Data
        datasets['career_path'] = pd.read_csv(datasets_config['career_path'])
        print(f"Loaded Career Path Data: {datasets['career_path'].shape}")

        # 2. Student/Recommendation Data 
        datasets['student_reco'] = pd.read_csv(datasets_config['student_reco'])
        print(f"Loaded Student Reco Data: {datasets['student_reco'].shape}")
        
        # Load the second student dataset
        datasets['student_reco_2'] = pd.read_csv(datasets_config['student_reco_2'])

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