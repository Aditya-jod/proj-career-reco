import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

class CareerRecommender:
    def __init__(self, job_df, embedding_matrix, feature_builder=None):
        self.job_df = job_df.reset_index(drop=True)
        self.embedding_matrix = embedding_matrix
        self.feature_builder = feature_builder

    def recommend(self, query_text, top_k=10):
        if isinstance(query_text, str):
            query_texts = [query_text]
        else:
            query_texts = query_text

        if self.feature_builder is None:
            raise ValueError("Feature builder is required to encode queries.")

        query_embedding = self.feature_builder.encode(query_texts)

        scores = cosine_similarity(query_embedding, self.embedding_matrix).flatten()

        top_idx = scores.argsort()[::-1][:top_k]
        return self.job_df.iloc[top_idx].assign(score=scores[top_idx])
    