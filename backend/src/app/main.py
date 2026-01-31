import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.loader import load_config, load_raw_data
from src.data.preprocessing import clean_text
from src.features.build_features import FeatureBuilder
from src.models.career_predictor import CareerPredictor
from src.models.career_recommender import CareerRecommender
from src.models.university_recommender import UniversityRecommender


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# USER INPUT HANDLERS
def collect_user_profile() -> dict:
    """
    Collect user academic scores and interests.
    """
    print("\n--- Academic Scores (0–100) ---")

    def _get_score(name, default):
        val = input(f"{name} (default {default}): ").strip()
        if not val:
            return int(default)
        try:
            return int(float(val))
        except ValueError:
            print("Invalid input. Please enter a numeric score.")
            return _get_score(name, default)

    profile = {
        "Mathematics_Score": _get_score("Mathematics", 0),
        "Science_Score": _get_score("Science", 0),
        "Language_Arts_Score": _get_score("Language Arts", 0),
        "Social_Studies_Score": _get_score("Social Studies", 0),
    }

    def _get_skill(name, default):
        val = input(f"{name} (default {default}): ").strip()
        if not val:
            return int(default)
        try:
            return int(float(val))
        except ValueError:
            print("Invalid input. Please enter a numeric score.")
            return _get_skill(name, default)

    profile.update(
        {
            "Logical_Reasoning": _get_skill("Logical Reasoning", 0),
            "Creativity": _get_skill("Creativity", 0),
            "Communication": _get_skill("Communication", 0),
            "Leadership": _get_skill("Leadership", 0),
            "Social_Skills": _get_skill("Social Skills", 0),
        }
    )

    profile["skills_text"] = input(
        "\nDescribe your skills & interests (free text): "
    ).strip()

    profile["preferred_location"] = input(
        "\nPreferred study location (India / USA / UK / etc.): "
    ).strip()

    return profile


# SYSTEM INITIALIZATION
def initialize_systems(datasets):
    """
    Initialize ML models and recommenders.
    """
    logger.info("Initializing Career Predictor...")
    career_predictor = CareerPredictor()
    career_predictor.load_or_train(datasets["student_reco"], verbose=False)

    logger.info("Initializing University Recommender...")
    feature_builder = FeatureBuilder()
    university_recommender = UniversityRecommender(
        feature_builder=feature_builder,
        indian_df=datasets["indian_colleges"],
        world_df=datasets["world_universities"],
        student_df=datasets.get("student_reco"),
    )

    logger.info("Initializing Job Recommender...")
    cache_dir = Path("models/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    job_df_path = cache_dir / "job_df.parquet"
    job_emb_path = cache_dir / "job_embeddings.npy"

    if job_df_path.exists() and job_emb_path.exists():
        job_df = pd.read_parquet(job_df_path)
        job_embeddings = np.load(job_emb_path)
    else:
        job_df = (
            datasets["job_descriptions"]
            .drop_duplicates(subset=["Job Title"])
            .reset_index(drop=True)
            .copy()
        )
        job_df["job_idx"] = job_df.index
        job_df["content"] = job_df[
            ["Job Title", "skills", "Job Description", "Responsibilities"]
        ].fillna("").agg(" ".join, axis=1)

        job_feature_builder = FeatureBuilder()
        job_embeddings = job_feature_builder.encode(
            job_df["content"].tolist(), batch_size=128
        )
        job_df.to_parquet(job_df_path, index=False)
        np.save(job_emb_path, job_embeddings)

    job_feature_builder = FeatureBuilder()
    job_recommender = CareerRecommender(
        job_df=job_df,
        embedding_matrix=job_embeddings,
        feature_builder=job_feature_builder,
    )

    return career_predictor, university_recommender, job_recommender, job_df

# PIPELINE
def run_pipeline():
    config = load_config()
    datasets = load_raw_data(config)

    career_predictor, uni_recommender, job_recommender, job_df = initialize_systems(datasets)

    print("\n" + "=" * 60)
    print("🎓 CAREER PATH RECOMMENDATION SYSTEM")
    print("=" * 60)

    while True:
        profile = collect_user_profile()

        # ---------------- Career Prediction ----------------
        predictions = career_predictor.predict_top_k(profile, k=3)
        top_career, confidence = predictions[0]

        print(f"\n🔮 Predicted Career Field: {top_career} ({confidence:.2%})")

        # ---------------- Job Recommendations ----------------
        query = profile["skills_text"] or top_career
        query = clean_text(query)

        jobs = job_recommender.recommend(query, top_k=10)

        # ---------------- University Recommendations ----------------
        universities = uni_recommender.recommend(
            query=top_career,
            country=profile["preferred_location"],
            top_k=10,
            skills_text=profile.get("skills_text", ""),
        )

        print("\n🏫 Recommended Universities:")
        if not universities.empty:
            print(universities.to_string(index=False))
        else:
            print("No matching universities found.")

        print("\n💼 Suggested Job Roles:")
        seen = set()
        count = 0
        for _, row in jobs.iterrows():
            title = job_df.loc[row["job_idx"], "Job Title"]
            if title not in seen:
                seen.add(title)
                count += 1
                print(f"{count}. {title}")
            if count == 5:
                break

        if input("\nPress Enter to continue or type 'exit' to quit: ").lower() == "exit":
            break


if __name__ == "__main__":
    run_pipeline()
