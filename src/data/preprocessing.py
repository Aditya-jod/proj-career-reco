import pandas as pd 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try: 
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    # Keep only letters, numbers and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    tokens = text.split()

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens 
        if word not in stop_words
    ]

    return " ".join(clean_tokens)


def preprocess_dataframe(df, columns_to_clean):
    df_clean = df.copy()
    
    for col in columns_to_clean:
        if col in df_clean.columns:
            print(f"Cleaning column: {col}...")
            df_clean[col] = df_clean[col].apply(clean_text)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    return df_clean

if __name__ == "__main__":
    sample_text = "I am a start-up founder and CEO!"
    print(f"Original Text: {sample_text}")
    print(f"Cleaned Text: {clean_text(sample_text)}")