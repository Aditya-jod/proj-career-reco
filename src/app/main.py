import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.loader import load_config, load_raw_data
from src.data.preprocessing import clean_text
from src.features.build_features import FeatureBuilder
from src.models.recommender import CareerRecommender
from src.models.career_predictor import CareerPredictor
from src.models.university_recommender import UniversityRecommender

def get_user_scores():
    """Helper to get scores from user input"""
    print("\n--- Please enter your scores (0-100) ---")
    scores = {}
    # Default values 
    defaults = {
        'Mathematics_Score': 0, 'Science_Score': 0, 'Language_Arts_Score': 0,
        'Social_Studies_Score': 0, 'Logical_Reasoning': 5, 'Creativity': 5,
        'Communication': 5, 'Leadership': 5, 'Social_Skills': 5
    }
    
    for field in defaults:
        val = input(f"{field} (default {defaults[field]}): ")
        scores[field] = float(val) if val.strip() else defaults[field]
    return scores

def main():
    # 1. Load Configuration & Data
    config = load_config()
    datasets = load_raw_data(config)
    
    # 2. Initialize & Train Career Predictor
    print("\n" + "="*50)
    print("   PHASE 1: TRAINING AI MODELS")
    print("="*50)
    
    predictor = CareerPredictor()
    predictor.train(datasets['student_reco'])
    
    # 3. Initialize University Recommender
    uni_recommender = UniversityRecommender(
        datasets['indian_colleges'], 
        datasets['world_universities']
    )
    
    # 4. Initialize Job Recommender 
    job_df = datasets['job_descriptions'].sample(n=10000, random_state=42)
    
    # --- FIX: Remove Duplicates to ensure variety in recommendations ---
    job_df = job_df.drop_duplicates(subset=['Job Title'])
    
    cols_to_combine = ['Job Title', 'skills', 'Job Description', 'Responsibilities']
    job_df['Content'] = job_df[cols_to_combine].fillna('').astype(str).agg(' '.join, axis=1)
    
    builder = FeatureBuilder(max_features=5000)
    job_recommender = CareerRecommender(builder, job_df, vector_column='Content')

    # 5. Interactive Session
    print("\n" + "="*50)
    print("   CAREER PATH RECOMMENDER SYSTEM (FULL PIPELINE)")
    print("="*50)
    
    while True:
        print("\n--- New Session ---")
        
        # A. Get User Profile for Career Prediction
        user_scores = get_user_scores()
        
        # B. Predict Broad Career Field
        predicted_field, confidence = predictor.predict(user_scores)
        print(f"\n>>> AI PREDICTION: You are best suited for: {predicted_field} (Confidence: {confidence:.2%}) <<<")
        
        # C. Recommend Universities
        location = input("\nWhere do you want to study? (e.g., India, USA, UK): ")
        print(f"\nFinding top {predicted_field} universities in {location}...")
        
        unis = uni_recommender.recommend(predicted_field, location)
        if not unis.empty:
            print(unis.to_string(index=False))
        else:
            print("No specific universities found for this criteria.")
            
        # D. Recommend Specific Jobs
        print(f"\nNow, let's find specific job roles in {predicted_field}.")
        skills = input("Enter your specific skills/interests (e.g., python, drawing, management): ")
        
        query = f"{predicted_field} {skills}"
        cleaned_query = clean_text(query)
        
        print(f"Searching for jobs matching: '{cleaned_query}'...")
        # Fetch more results (20) so we can filter out duplicates and still show top 5
        recommendations = job_recommender.recommend(cleaned_query, top_k=20)
        
        print("\nTop 5 Recommended Job Roles:")
        print("-" * 40)
        
        seen_titles = set()
        count = 0
        
        for i, row in recommendations.iterrows():
            idx = row['index']
            job_title = job_df.loc[idx, 'Job Title']
            
            if job_title in seen_titles:
                continue
                
            seen_titles.add(job_title)
            count += 1
            
            print(f"{count}. {job_title} (Score: {row['score']:.2f})")
            
            if count >= 5:
                break
                
        print("-" * 40)
        
        if input("\nType 'exit' to quit or Enter to continue: ").lower() == 'exit':
            break

if __name__ == "__main__":
    main()