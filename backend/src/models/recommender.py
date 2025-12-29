import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

class CareerRecommender:
    def __init__(self, feature_builder, career_df, vector_column='Career'):
         self.builder = feature_builder
         self.career_df = career_df
         self.vector_column = vector_column

         print("Caching career vectors for recommendation...")
         self.career_vectors = self.builder.fit_transform(self.career_df[self.vector_column])
        
    def recommend(self, student_profile, top_k=5):
         student_vector = self.builder.vectorizer.transform([student_profile])
         similarities = cosine_similarity(student_vector, self.career_vectors)
         top_indices = similarities[0].argsort()[-top_k:][::-1]
         
         results = []
         for idx in top_indices:
              score = similarities[0][idx]
              original_idx = self.career_df.index[idx]
              results.append({"index": original_idx, "score": score})

         return pd.DataFrame(results)