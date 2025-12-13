import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.loader import load_config, load_raw_data
from src.data.preprocessing import preprocess_dataframe
from src.features.build_features import FeatureBuilder
from src.models.clustering import CareerCluster

def main():
    config = load_config()
    if config is None:
        return
    
    datasets = load_raw_data(config)
    career_df = datasets['career_path']
    
    print("\nColumns in Career Data:", career_df.columns.tolist())
    
    text_column = 'Career' 
    
    if text_column not in career_df.columns:
        print(f"\nError: Column '{text_column}' not found.")
        print(f"Please update 'text_column' in main.py to one of: {career_df.columns.tolist()}")
        return

    # 3. Preprocessing
    print("\n--- Step 3: Cleaning Text ---")
    career_df = preprocess_dataframe(career_df, [text_column])
    
    # 4. Feature Engineering (TF-IDF)
    print("\n--- Step 4: Vectorizing ---")
    builder = FeatureBuilder()
    vectors = builder.fit_transform(career_df[text_column])
    
    # 5. Clustering
    print("\n--- Step 5: Clustering ---")
    cluster_model = CareerCluster(n_clusters=5)
    labels = cluster_model.fit_predict(vectors)

    career_df['Cluster_ID'] = labels
    
    print("\n--- Results: Sample Careers in Each Cluster ---")
    for cluster_id in range(5):
        print(f"\nCluster {cluster_id}:")
        sample = career_df[career_df['Cluster_ID'] == cluster_id][text_column].head(2)
        for text in sample:
            print(f" - {text[:100]}...")

if __name__ == "__main__":
    main()