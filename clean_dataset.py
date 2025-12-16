import pandas as pd
import os

def clean_dataset():
    input_path = r'D:\Data Science advance projects\Career Path Recommender\Dataset\career recommendation dataset\career_recommender.csv'
    print(f'Reading file: {input_path}')

    # Try reading with skipinitialspace=True to handle spaces before quotes
    try:
        print('Attempting to read with skipinitialspace=True...')
        df = pd.read_csv(input_path, skipinitialspace=True)
        
        print('Successfully read CSV with skipinitialspace=True')
        print(f'Shape: {df.shape}')
        print(f'Columns ({len(df.columns)}):')
        for i, col in enumerate(df.columns):
            print(f'{i}: {col}')
        
        # Standardize column names
        new_columns = [
            'Name', 
            'Gender', 
            'UG_Course', 
            'UG_Specialization', 
            'Interests', 
            'Skills', 
            'CGPA', 
            'Certifications', 
            'Certification_Title', 
            'Working', 
            'Job_Title', 
            'Masters'
        ]
        
        if len(df.columns) == len(new_columns):
            df.columns = new_columns
            print('\nRenamed columns successfully.')
            
            # Clean up the data
            # Remove any leading/trailing whitespace from string columns
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            
            # Save the cleaned dataset
            output_path = os.path.join(os.path.dirname(input_path), 'cleaned_career_recommender.csv')
            df.to_csv(output_path, index=False)
            print(f'\nSaved cleaned dataset to: {output_path}')
            
            # Show sample
            print('\nSample Data:')
            print(df[['Name', 'Skills', 'Certification_Title']].head())
            
        else:
            print(f'\nColumn count mismatch. Expected {len(new_columns)}, got {len(df.columns)}')
            
    except Exception as e:
        print(f'Error reading with skipinitialspace=True: {e}')

if __name__ == '__main__':
    clean_dataset()
