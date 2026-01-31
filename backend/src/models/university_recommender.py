import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.university_ranker import (
    UniversityContext,
    UniversityRankerModel,
    UniversityRankerTrainer,
)
from src.models.country_utils import country_pattern


logger = logging.getLogger(__name__)


@dataclass
class UniversityDatasetBuilder:
    """Normalizes raw Indian/world college feeds into a unified dataframe."""

    indian_df: pd.DataFrame
    world_df: pd.DataFrame

    def build(self) -> pd.DataFrame:
        try:
            indian = self.indian_df.copy()
            world = self.world_df.copy()

            indian["search_text"] = (
                indian["College Name"].fillna("")
                + " "
                + indian["University Name"].fillna("")
                + " "
                + indian["Specialised in"].fillna("")
                + " "
                + indian["College Type"].fillna("")
                + " "
                + indian["University Type"].fillna("")
                + " "
                + indian["Management"].fillna("")
            )
            indian["country"] = "India"
            indian_clean = indian[
                [
                    "search_text",
                    "College Name",
                    "State",
                    "District",
                    "country",
                    "Website",
                ]
            ].rename(columns={"College Name": "name"})

            world["search_text"] = world["name"].fillna("") + " " + world["country"].fillna("")
            world_clean = world[
                ["search_text", "name", "country", "web_pages"]
            ].rename(columns={"web_pages": "Website"})

            unified = pd.concat([indian_clean, world_clean], ignore_index=True)
            unified["Website"] = unified["Website"].astype(str)
            return unified
        except Exception as exc:
            raise ValueError("Failed to build university dataset") from exc


class EmbeddingCache:
    """Manages loading and persisting sentence embeddings."""

    def __init__(self, feature_builder, cache_path: str):
        self.feature_builder = feature_builder
        self.cache_path = cache_path

    def load_or_build(self, texts: List[str]) -> np.ndarray:
        try:
            if os.path.exists(self.cache_path):
                print("⚡ Loading cached university embeddings...")
                return np.load(self.cache_path)

            print("🚀 Computing university embeddings (one-time)...")
            embeddings = self.feature_builder.encode(texts)
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.save(self.cache_path, embeddings)
            print("✅ Embeddings cached")
            return embeddings
        except Exception as exc:
            raise RuntimeError("Unable to load or build university embeddings") from exc

class UniversityRecommender:
    """Ranks universities using semantic similarity between query and metadata."""

    def __init__(
        self,
        feature_builder,
        indian_df: pd.DataFrame,
        world_df: pd.DataFrame,
        embedding_path: str = "models/university_embeddings.npy",
        student_df: Optional[pd.DataFrame] = None,
        ranker_model_path: str = "models/university_ranker.pkl",
        train_ranker: bool = True,
    ):
        self.feature_builder = feature_builder
        dataset_builder = UniversityDatasetBuilder(indian_df, world_df)
        self.unified_df = dataset_builder.build()
        cache = EmbeddingCache(feature_builder, embedding_path)
        self.embedding_matrix = cache.load_or_build(self.unified_df["search_text"].tolist())
        self.ranker: Optional[UniversityRankerModel] = None

        if ranker_model_path:
            self.ranker = UniversityRankerModel(ranker_model_path)
            if (
                train_ranker
                and (not self.ranker.is_ready())
                and student_df is not None
                and not student_df.empty
            ):
                logger.info("Training university ranker model from available datasets...")
                trainer = UniversityRankerTrainer()
                trainer.train(
                    universities=self.unified_df,
                    career_labels=student_df.get("Primary_Career_Recommendation", []),
                    save_path=ranker_model_path,
                )
                self.ranker = UniversityRankerModel(ranker_model_path)

    def recommend(
        self,
        query: str,
        country: Optional[str] = None,
        state: Optional[str] = None,
        top_k: int = 10,
        skills_text: str = "",
    ) -> pd.DataFrame:
        candidate_df = self.unified_df
        pattern = None
        country_enforced = False

        if country:
            pattern = country_pattern(country)
            filtered_candidates = candidate_df[
                candidate_df["country"].str.contains(pattern, case=False, na=False)
            ]
            if filtered_candidates.empty:
                print(
                    "⚠️ No universities found for the requested country; showing global matches instead."
                )
            else:
                candidate_df = filtered_candidates
                country_enforced = True

        if self.ranker and self.ranker.is_ready():
            context = UniversityContext(
                career_text=query,
                skills_text=skills_text,
                preferred_country=country,
                preferred_state=state,
            )
            ranked = self.ranker.rank(candidate_df, context, top_k=top_k)
            if pattern and not country_enforced:
                filtered = ranked[ranked["country"].str.contains(pattern, case=False, na=False)]
                if not filtered.empty:
                    ranked = filtered
            ranked = ranked.rename(columns={"ml_score": "score"})
            return ranked[["name", "country", "State", "District", "Website", "score"]].reset_index(
                drop=True
            )

        try:
            query_vec = self.feature_builder.encode(query)
            scores = cosine_similarity(query_vec, self.embedding_matrix).flatten()
        except Exception as exc:
            raise RuntimeError("Unable to score university recommendations") from exc

        results = self.unified_df.copy()
        results["score"] = scores

        if country:
            pattern = country_pattern(country)
            filtered = results[results["country"].str.contains(pattern, case=False, na=False)]
            if filtered.empty:
                print(
                    "⚠️ No universities found for the requested country; showing global matches instead."
                )
            else:
                results = filtered

        if state and "State" in results.columns:
            results = results[results["State"].str.contains(state, case=False, na=False)]

        return (
            results.sort_values("score", ascending=False)
            .head(top_k)[["name", "country", "State", "District", "Website", "score"]]
            .reset_index(drop=True)
        )

    @staticmethod
    def _country_pattern(country: str) -> str:
        """Deprecated helper retained for backward compatibility."""
        return country_pattern(country)
