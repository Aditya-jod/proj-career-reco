import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.loader import load_config, load_raw_data
from src.models.career_predictor import CareerPredictor

def test_predictor():
    # 1. Load Data
    config = load_config()
    datasets = load_raw_data(config)
    student_df = datasets['student_reco']
    
    # 2. Initialize and Train
    predictor = CareerPredictor()
    predictor.train(student_df)
    
    # 3. Test Prediction
    print("\n--- Testing Prediction ---")
    # Create a dummy student who is good at Math/Science (Should be STEM)
    test_student = {
        'Mathematics_Score': 95, 
        'Science_Score': 90, 
        'Language_Arts_Score': 60,
        'Social_Studies_Score': 60, 
        'Logical_Reasoning': 9, 
        'Creativity': 5,
        'Communication': 4, 
        'Leadership': 5, 
        'Social_Skills': 4
    }
    
    field, conf = predictor.predict(test_student)
    print(f"Test Input: High Math/Science")
    print(f"Predicted Field: {field}")
    print(f"Confidence: {conf:.2%}")

if __name__ == "__main__":
    test_predictor()
