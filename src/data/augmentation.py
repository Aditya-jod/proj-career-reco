import pandas as pd
import numpy as np

def augment_cleaned_data(cleaned_df):
    """
    Augments the cleaned career recommender dataset with synthetic scores
    to match the schema required by the CareerPredictor.
    """
    print("Augmenting cleaned dataset with synthetic scores...")
    
    df = cleaned_df.copy()
    
    if 'Job_Title' in df.columns:
        df['Primary_Career_Recommendation'] = df['Job_Title']
    else:
        df['Primary_Career_Recommendation'] = df['UG_Specialization']
        
    df = df.dropna(subset=['Primary_Career_Recommendation'])
    
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

    
    for col in score_columns:
        scores = np.random.normal(loc=75, scale=15, size=len(df))
        
        scores = np.clip(scores, 0, 100)
        
        df[col] = scores
        
    final_columns = score_columns + ['Primary_Career_Recommendation']
    
    
    return df[final_columns]
