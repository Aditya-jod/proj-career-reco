import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FeatureBuilder:
    def __init__(self, max_features=5000):
        "Initialize the TF-IDF Vectorizer."
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english'
        )
        self.feature_matrix = None

    def fit_transform(self, text_data):
        "Learn vocabulary and convert text data to TF-IDF matrix "
        print("Generating TF-IDF features...")
        self.feature_matrix = self.vectorizer.fit_transform(text_data)

        print(f"Feature Matrix shape: {self.feature_matrix.shape}")
        return self.feature_matrix
    
    def get_similarity(self, vector1, vector2):
        "Compute Cosine Similarity between two vectors."
        return cosine_similarity(vector1, vector2)