import logging
import os
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"

# Anchor used to resolve relative dataset paths in config.yaml.
# config.yaml lives at  backend/config/config.yaml  and its paths start with
# "../Dataset/…" — meaning they are relative to the *project root*
# (proj-career-reco/), NOT the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # …/proj-career-reco


def _resolve_path(raw: str) -> Path:
    """Return an absolute Path for *raw*.

    If *raw* is already absolute, return it unchanged.
    Otherwise resolve it against _PROJECT_ROOT so the server can be launched
    from any working directory.
    """
    p = Path(raw)
    return p if p.is_absolute() else (_PROJECT_ROOT / raw).resolve()

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
        logger.error("Config file not found at %s", path)
        return None

def load_raw_data(config):
    """
    Load raw datasets defined in the config.
    Returns a dictionary containing the dataframes.
    """
    datasets_config = config['datasets']
    
    datasets = {}
    
    try:
        logger.info("Loading datasets...")
        
        # 1. Career Path Data
        datasets['career_path'] = pd.read_csv(_resolve_path(datasets_config['career_path']), skipinitialspace=True)
        datasets['career_path'].columns = datasets['career_path'].columns.str.strip()
        logger.info("Loaded Career Path Data: %s", datasets['career_path'].shape)

        # 2. Student/Recommendation Data
        datasets['student_reco'] = pd.read_csv(_resolve_path(datasets_config['student_reco']), skipinitialspace=True)
        datasets['student_reco'].columns = datasets['student_reco'].columns.str.strip()
        logger.info("Loaded Student Reco Data: %s", datasets['student_reco'].shape)

        # Load the second student dataset
        try:
            datasets['student_reco_2'] = pd.read_csv(_resolve_path(datasets_config['student_reco_2']), skipinitialspace=True)
            datasets['student_reco_2'].columns = datasets['student_reco_2'].columns.str.strip()
            logger.info("Loaded Student Reco Data 2: %s", datasets['student_reco_2'].shape)
        except Exception as e:
            logger.warning("Could not load student_reco_2: %s", e)
            datasets['student_reco_2'] = None

        # 3. Job Descriptions
        datasets['job_descriptions'] = pd.read_csv(_resolve_path(datasets_config['job_descriptions']), skipinitialspace=True)
        logger.info("Loaded Job Descriptions: %s", datasets['job_descriptions'].shape)

        # 4. Colleges & Universities Data
        datasets['indian_colleges'] = pd.read_csv(_resolve_path(datasets_config['indian_colleges']), skipinitialspace=True)
        datasets['world_universities'] = pd.read_csv(_resolve_path(datasets_config['world_universities']), skipinitialspace=True)
        
        logger.info("All primary datasets loaded successfully.")
        return datasets
        
    except FileNotFoundError as e:
        logger.error("Error loading datasets: %s", e)
        return None
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        return None
    
if __name__ == "__main__":
    config = load_config()
    
    if config:
        data = load_raw_data(config)
        if data:
            print("\nSuccess! Data loaded.")
            print(f"Keys available: {list(data.keys())}")
