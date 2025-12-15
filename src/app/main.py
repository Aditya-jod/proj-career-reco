import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.loader import load_config, load_raw_data
from src.data.preprocessing import preprocess_dataframe, clean_text
from src.features.build_features import FeatureBuilder
from src.models.recommender import CareerRecommender

def main():
    # 1. Load Configuration
    config = load_config()
    if config is None:
        return
    
    # 2. Load Data
    datasets = load_raw_data(config)
    
    # --- SWITCH TO THE BIG DATASET ---
    job_df = datasets['job_descriptions']
    
    print(f"\nOriginal Job Data Size: {job_df.shape}")
    
    # --- SAMPLING ---
    # 1.6 Million is too big for a laptop to process in seconds.
    # Let's take a random sample of 20,000 jobs.
    sample_size = 20000
    print(f"Sampling {sample_size} random jobs for performance...")
    job_df = job_df.sample(n=sample_size, random_state=42)
    
    # --- COMBINE COLUMNS ---
    # We create a rich profile for each job
    cols_to_combine = ['Job Title', 'skills', 'Job Description', 'Responsibilities']
    
    print(f"Combining columns {cols_to_combine}...")
    
    # .astype(str) prevents the "int" error we saw before
    job_df['Content'] = job_df[cols_to_combine].fillna('').astype(str).agg(' '.join, axis=1)
    
    # 3. Preprocessing
    print("\n--- Step 3: Cleaning Data ---")
    job_df = preprocess_dataframe(job_df, ['Content'])
    
    # 4. Feature Engineering
    print("\n--- Step 4: Building Model ---")
    builder = FeatureBuilder(max_features=5000) # Limit vocabulary to top 5000 words
    
    # 5. Initialize Recommender
    print("\n--- Step 5: Initializing Recommender System ---")
    # We use 'Content' for matching, but we will display 'Job Title'
    recommender = CareerRecommender(builder, job_df, vector_column='Content')
    
    # 6. Interactive Loop
    print("\n" + "="*50)
    print("   CAREER PATH RECOMMENDER SYSTEM (PRO EDITION)")
    print("="*50)
    
    while True:
        print("\nTell me about yourself (skills, interests, hobbies).")
        print("Type 'exit' to quit.")
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            break
            
        cleaned_input = clean_text(user_input)
        print(f"\nProcessing: '{cleaned_input}'...")
        
        # Get recommendations
               # ... inside the loop in main.py ...
        recommendations = recommender.recommend(cleaned_input, top_k=5)
        
        print("\nTop 5 Recommended Jobs for you:")
        print("-" * 40)
        for i, row in recommendations.iterrows():
            idx = row['index']
            score = row['score']
            
            # Look up details using the index
            job_title = job_df.loc[idx, 'Job Title']
            company = job_df.loc[idx, 'Company']
            location = job_df.loc[idx, 'location']
            
            print(f"{i+1}. {job_title} at {company}")
            print(f"   Location: {location} | Score: {score:.2f}")
        print("-" * 40)

if __name__ == "__main__":
    main()