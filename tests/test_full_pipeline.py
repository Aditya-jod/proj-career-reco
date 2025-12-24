import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_config, load_raw_data
from src.data.preprocessing import clean_text
from src.features.build_features import FeatureBuilder
from src.models.recommender import CareerRecommender
from src.models.career_predictor import CareerPredictor
from src.models.university_recommender import UniversityRecommender

def test_pipeline():
    print(">>> INITIALIZING TEST PIPELINE <<<")
    
    # 1. Load Configuration & Data
    config = load_config()
    datasets = load_raw_data(config)
    
    # 2. Train Career Predictor
    print("\n[1/4] Training Career Predictor...")
    predictor = CareerPredictor()
    predictor.train(datasets['student_reco'])
    
    # 3. Initialize University Recommender
    print("\n[2/4] Initializing University Recommender...")
    uni_recommender = UniversityRecommender(
        datasets['indian_colleges'], 
        datasets['world_universities']
    )
    
    # 4. Initialize Job Recommender
    print("\n[3/4] Initializing Job Recommender...")
    job_df = datasets['job_descriptions'].sample(n=10000, random_state=42)
    job_df = job_df.drop_duplicates(subset=['Job Title'])
    
    cols_to_combine = ['Job Title', 'skills', 'Job Description', 'Responsibilities']
    job_df['Content'] = job_df[cols_to_combine].fillna('').astype(str).agg(' '.join, axis=1)
    
    builder = FeatureBuilder(max_features=5000)
    job_recommender = CareerRecommender(builder, job_df, vector_column='Content')

    # 5. Run Test Scenario
    print("\n[4/4] Running Test Scenario...")
    
    # Scenario: Student good at Math/Science, wants to study in USA, likes Python
    test_scores = {
        'Mathematics_Score': 95, 'Science_Score': 90, 'Language_Arts_Score': 60,
        'Social_Studies_Score': 60, 'Logical_Reasoning': 9, 'Creativity': 5,
        'Communication': 4, 'Leadership': 5, 'Social_Skills': 4
    }
    location_pref = "USA"
    skills_pref = "Python Data Science"
    
    print(f"\n--- Test Inputs ---")
    print(f"Scores: High Math/Science")
    print(f"Location: {location_pref}")
    print(f"Skills: {skills_pref}")
    
    # A. Predict
    predicted_field, confidence = predictor.predict(test_scores)
    print(f"\n>>> AI PREDICTION: {predicted_field} (Confidence: {confidence:.2%})")
    
    # B. Universities
    print(f"\n>>> UNIVERSITY RECOMMENDATIONS ({location_pref}) <<<")
    unis = uni_recommender.recommend(predicted_field, location_pref)
    if not unis.empty:
        print(unis.to_string(index=False))
    else:
        print("No universities found.")
        
    # C. Jobs
    print(f"\n>>> JOB RECOMMENDATIONS <<<")
    query = f"{predicted_field} {skills_pref}"
    cleaned_query = clean_text(query)
    print(f"Query: {cleaned_query}")
    
    recommendations = job_recommender.recommend(cleaned_query, top_k=20)
    
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

if __name__ == "__main__":
    test_pipeline()
