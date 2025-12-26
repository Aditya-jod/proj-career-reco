import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.loader import load_config, load_raw_data
from src.data.preprocessing import clean_text
from src.data.augmentation import augment_cleaned_data
from src.features.build_features import FeatureBuilder
from src.models.recommender import CareerRecommender
from src.models.career_predictor import CareerPredictor
from src.models.university_recommender import UniversityRecommender

def get_user_scores():
    """Helper to get scores from user input"""
    print("\n--- Please enter your scores ---")
    scores = {}
    
    # Group 1: Academic Scores (0-100)
    academic_cols = ['Mathematics_Score', 'Science_Score', 'Language_Arts_Score', 'Social_Studies_Score']
    print("\n[Academic Scores (0-100)]")
    for field in academic_cols:
        val = input(f"{field} (default 50): ")
        scores[field] = float(val) if val.strip() else 50.0
        
    # Group 2: Soft Skills (1-10)
    soft_skills = ['Logical_Reasoning', 'Creativity', 'Communication', 'Leadership', 'Social_Skills']
    print("\n[Soft Skills (1-10)]")
    for field in soft_skills:
        val = input(f"{field} (default 5): ")
        val = float(val) if val.strip() else 5.0
        
        # Auto-correct if user enters 0-100 scale for soft skills
        if val > 10:
            print(f"   -> Assuming {val} means {val/10} on a 1-10 scale.")
            val = val / 10
            
        scores[field] = val
        
    return scores

def main():
    config = load_config()
    datasets = load_raw_data(config)
    
    # 2. Initialize & Train Career Predictor
    print("\n" + "="*50)
    print("   PHASE 1: TRAINING AI MODELS")
    print("="*50)
    
    # --- DATA MERGING START ---
    # NOTE: Merging disabled to keep model size small (GitHub limit < 100MB)
    print("Training on primary dataset (Broad Career Fields)...")
    main_df = datasets['student_reco']
    predictor = CareerPredictor()
    predictor.train(main_df, verbose=False)
    
    print("Career Predictor Model trained successfully.")
    
    # 3. Initialize University Recommender
    uni_recommender = UniversityRecommender(
        datasets['indian_colleges'], 
        datasets['world_universities']
    )
    
    # 4. Initialize Job Recommender 
    job_df = datasets['job_descriptions'].sample(n=30000, random_state=42)
    
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
        # 1. Collect all user inputs up front
        user_scores = get_user_scores()
        skills = input("\nEnter your specific skills/interests (e.g., python, drawing, management): ")
        location = input("\nWhere do you want to study? (e.g., India, USA, UK): ")

        # 2. Predict Broad Career Field
        top_predictions = predictor.predict_top_k(user_scores, k=3)
        predicted_field, confidence = top_predictions[0]

        # 3. Recommend Specific Jobs (using interests or predicted field)
        if skills.strip():
            query = skills
        else:
            query = predicted_field
        cleaned_query = clean_text(query)
        recommendations = job_recommender.recommend(cleaned_query, top_k=20)

        # 4. Recommend Universities (using predicted field and location)
        unis = uni_recommender.recommend(predicted_field, location)

        # 5. Present the full roadmap
        print("\n" + "="*50)
        print("   YOUR CAREER ROADMAP")
        print("="*50)

        print(f"\nScores & Skills suggest: {predicted_field} (Confidence: {confidence:.2%})")
        if confidence < 0.50:
            print("   (Note: The AI is not 100% sure. Here are other strong matches:)")
            for i, (label, prob) in enumerate(top_predictions[1:], 1):
                print(f"   {i}. {label} ({prob:.2%})")
            choice = input("\n   Do you want to stick with the top prediction? (y/n): ").lower()
            if choice == 'n':
                print("   Please type the career field you prefer from the list above:")
                predicted_field = input("   > ").strip()
                unis = uni_recommender.recommend(predicted_field, location)
                if not skills.strip():
                    query = predicted_field
                    cleaned_query = clean_text(query)
                    recommendations = job_recommender.recommend(cleaned_query, top_k=20)

        print(f"\nTop universities in {location} for {predicted_field}:")
        if not unis.empty:
            print(unis.to_string(index=False))
        else:
            print("No specific universities found for this criteria.")

        print(f"\nTop job roles matching your interests ('{cleaned_query}'):")
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
