import os
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity


class UniversityRecommender:
    def __init__(
        self,
        feature_builder,
        indian_df: pd.DataFrame,
        world_df: pd.DataFrame,
        embedding_path: str = "models/university_embeddings.npy"
    ):
        self.feature_builder = feature_builder
        self.embedding_path = embedding_path

        self.indian_df = indian_df.copy()
        self.world_df = world_df.copy()

        self._prepare_data()
        self._load_or_build_embeddings()

    def _prepare_data(self):
        self.indian_df["search_text"] = (
            self.indian_df["College Name"].fillna("") + " " +
            self.indian_df["University Name"].fillna("") + " " +
            self.indian_df["Specialised in"].fillna("") + " " +
            self.indian_df["College Type"].fillna("") + " " +
            self.indian_df["University Type"].fillna("") + " " +
            self.indian_df["Management"].fillna("")
        )

        self.indian_df["country"] = "India"

        indian_clean = self.indian_df[[
            "search_text",
            "College Name",
            "State",
            "District",
            "country",
            "Website"
        ]].rename(columns={"College Name": "name"})

        self.world_df["search_text"] = (
            self.world_df["name"].fillna("") + " " +
            self.world_df["country"].fillna("")
        )

        world_clean = self.world_df[[
            "search_text",
            "name",
            "country",
            "web_pages"
        ]].rename(columns={"web_pages": "Website"})

        self.unified_df = pd.concat([indian_clean, world_clean], ignore_index=True)
        self.unified_df["Website"] = self.unified_df["Website"].astype(str)

    def _load_or_build_embeddings(self):
        if os.path.exists(self.embedding_path):
            print("⚡ Loading cached university embeddings...")
            self.embedding_matrix = np.load(self.embedding_path)
        else:
            print("🚀 Computing university embeddings (one-time)...")
            self.embedding_matrix = self.feature_builder.encode(
                self.unified_df["search_text"].tolist()
            )
            os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
            np.save(self.embedding_path, self.embedding_matrix)
            print("✅ Embeddings cached")

    def recommend(
        self,
        query: str,
        country: Optional[str] = None,
        state: Optional[str] = None,
        top_k: int = 10
    ) -> pd.DataFrame:

        query_vec = self.feature_builder.encode(query)
        scores = cosine_similarity(query_vec, self.embedding_matrix).flatten()

        results = self.unified_df.copy()
        results["score"] = scores

        if country:
            results = results[results["country"].str.contains(country, case=False, na=False)]

        if state and "State" in results.columns:
            results = results[results["State"].str.contains(state, case=False, na=False)]

        return (
            results
            .sort_values("score", ascending=False)
            .head(top_k)[[
                "name",
                "country",
                "State",
                "District",
                "Website",
                "score"
            ]]
            .reset_index(drop=True)
        )
