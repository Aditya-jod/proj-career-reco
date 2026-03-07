import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for _corpus in ("corpora/stopwords", "corpora/wordnet"):
    try:
        nltk.data.find(_corpus)
    except LookupError:
        nltk.download(_corpus.split("/")[1], quiet=True)

# Creating these inside clean_text() would rebuild them for every single row,
# which is catastrophic at 1.6 M job description rows.
_STOP_WORDS: frozenset = frozenset(stopwords.words("english"))
_LEMMATIZER: WordNetLemmatizer = WordNetLemmatizer()


def clean_text(text: object) -> str:
    """Lowercase, strip punctuation, remove stop-words, lemmatize."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = text.split()

    clean_tokens = [
        _LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in _STOP_WORDS
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