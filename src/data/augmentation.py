import pandas as pd
import numpy as np

def augment_cleaned_data(cleaned_df):
    """
    Augments the cleaned career recommender dataset with synthetic scores
    to match the schema required by the CareerPredictor.
    """
    print("Augmenting cleaned dataset with synthetic scores...")
    
    # 1. Map Target Column
    # We use 'Job_Title' as the ground truth for 'Primary_Career_Recommendation'
    # We might need to clean the job titles to match the categories in the main dataset,
    # but for now let's use them as is or map them if possible.
    
    df = cleaned_df.copy()
    
    # Rename Job_Title to Primary_Career_Recommendation if it exists, else use UG_Specialization
    if 'Job_Title' in df.columns:
        df['Primary_Career_Recommendation'] = df['Job_Title']
    else:
        # Fallback if Job_Title is missing or NA
        df['Primary_Career_Recommendation'] = df['UG_Specialization']
        
    # Drop rows where target is NaN
    df = df.dropna(subset=['Primary_Career_Recommendation'])
    
    # 2. Generate Synthetic Scores
    # The main dataset has these score columns
    score_columns = [
        'Mathematics_Score', 
        'Science_Score', 
        'Language_Arts_Score', 
        'Social_Studies_Score', 
        'Logical_Reasoning', 
        'Creativity', 
        'Communication', 
        'Leadership', 
        'Social_Skills'
    ]
    
    # We will generate random scores for now. 
    # In a real scenario, we would base this on the 'Skills' or 'Interests' column.
    # For example, if 'Skills' contains 'Python', Math score might be higher.
    
    for col in score_columns:
        # Generate normal distribution centered at 75 with std dev 15
        # This assumes people in these careers generally have good scores
        scores = np.random.normal(loc=75, scale=15, size=len(df))
        
        # Clip to 0-100 range
        scores = np.clip(scores, 0, 100)
        
        df[col] = scores
        
    # 3. Select only necessary columns
    final_columns = score_columns + ['Primary_Career_Recommendation']
    
    # Keep other metadata if useful for debugging
    # final_columns += ['Name', 'Gender', 'Skills', 'Interests']
    
    return df[final_columns]
