import pandas as pd
try:
    df = pd.read_csv('D:/Data Science advance projects/Career Path Recommender/Dataset/job dataset/job_descriptions.csv')
    print(f"Total Rows: {len(df)}")
    print(f"Unique Titles: {df['Job Title'].nunique()}")
    print("Top 10 Titles:")
    print(df['Job Title'].value_counts().head(10))
except Exception as e:
    print(e)
