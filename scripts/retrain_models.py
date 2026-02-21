"""
Retrain all ML models from scratch.
Run this after cloning the repo — models are NOT stored in git.

Usage:
    cd <project-root>
    python scripts/retrain_models.py
"""
import sys
import os

sys.path.append(os.path.abspath("backend"))

from src.data.loader import load_config, load_raw_data
from src.features.build_features import FeatureBuilder
from src.models.career_predictor import CareerPredictor
from src.models.university_ranker import UniversityRankerTrainer
from src.models.university_recommender import UniversityDatasetBuilder

def main():
    print("Loading configuration and datasets...")
    config = load_config()
    data = load_raw_data(config)

    # ── 1. Career Predictor ──────────────────────────────────────
    print("\n[1/2] Training Career Predictor...")
    predictor = CareerPredictor(model_path="models/career_predictor.pkl")
    predictor.train(data["student_reco"], verbose=True)
    print("  ✓ Saved → models/career_predictor.pkl")

    # ── 2. University Ranker ─────────────────────────────────────
    print("\n[2/2] Training University Ranker...")
    builder = UniversityDatasetBuilder(
        data["indian_colleges"], data["world_universities"]
    )
    unified = builder.build()
    trainer = UniversityRankerTrainer()
    trainer.train(
        universities=unified,
        career_labels=data["student_reco"]["Primary_Career_Recommendation"],
        save_path="models/university_ranker.pkl",
        max_countries=30,
    )
    print("  ✓ Saved → models/university_ranker.pkl")

    print("\nAll models trained successfully.")
    print("University embeddings will be generated automatically on first run.")


if __name__ == "__main__":
    main()
