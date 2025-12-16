import pandas as pd
import os

file_path = "D:/Data Science advance projects/Career Path Recommender/Dataset/career recommendation dataset/career_recommendation_dataset.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(df[['Mathematics_Score', 'Logical_Reasoning', 'Creativity']].describe())
else:
    print("File not found")
