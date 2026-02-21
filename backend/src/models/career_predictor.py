import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.models.config import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT,
    RF_CLASS_WEIGHT,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_RANDOM_STATE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Career keyword map used to boost predictions from free-text skills input
# ---------------------------------------------------------------------------
CAREER_KEYWORD_MAP: Dict[str, List[str]] = {
    "STEM": [
        "python", "data", "programming", "coding", "algorithm", "analytics",
        "machine learning", "statistics", "r", "java", "software", "engineering",
        "physics", "chemistry", "biology", "computer", "ai", "ml", "deep learning",
        "mathematics", "calculus", "sql", "database", "network", "cybersecurity",
    ],
    "Healthcare": [
        "medical", "health", "nursing", "doctor", "patient", "clinical",
        "pharmacy", "therapy", "medicine", "hospital", "surgery", "diagnosis",
        "anatomy", "physiology", "biomedical", "dentistry", "radiology",
    ],
    "Business_Finance": [
        "business", "finance", "accounting", "management", "marketing",
        "sales", "economics", "investment", "trading", "entrepreneur",
        "strategy", "consulting", "product", "analyst", "budget", "revenue",
        "commerce", "supply chain", "operations",
    ],
    "Education": [
        "teaching", "education", "learning", "curriculum", "training",
        "coaching", "tutoring", "classroom", "pedagogy", "lesson plan",
        "school", "professor", "instructor",
    ],
    "Arts_Media": [
        "design", "art", "creative", "writing", "music", "photography",
        "video", "graphic", "animation", "media", "film", "content",
        "illustration", "storytelling", "journalism", "advertising",
    ],
    "Social_Services": [
        "social", "community", "welfare", "counseling", "nonprofit",
        "volunteer", "psychology", "mental health", "social work",
        "advocacy", "rehabilitation", "case management",
    ],
    "Government_Law": [
        "law", "legal", "policy", "government", "politics", "justice",
        "compliance", "regulation", "public administration", "civil service",
        "legislation", "judiciary", "attorney", "paralegal",
    ],
    "Trades_Manufacturing": [
        "mechanical", "electrical", "plumbing", "carpentry", "manufacturing",
        "construction", "trades", "welding", "automotive", "hvac",
        "industrial", "machinist", "technician",
    ],
}


class SkillsTextBooster:
    """
    Adjusts career prediction probabilities using keyword signals from
    free-text skills input.  Applied after model inference so no retraining
    is required when skills_text is available.
    """

    def __init__(self, keyword_map: Optional[Dict[str, List[str]]] = None, weight: float = 0.4):
        self.keyword_map = keyword_map or CAREER_KEYWORD_MAP
        # weight controls how much the text signal shifts the final score
        # 0.0 = pure model, 1.0 = pure text match
        self.weight = weight

    def boost(
        self,
        class_probs: np.ndarray,
        class_labels: np.ndarray,
        skills_text: str,
    ) -> np.ndarray:
        """Return adjusted probability array.  class_labels must align with class_probs."""
        if not skills_text or not skills_text.strip():
            return class_probs

        tokens = set(re.findall(r"[a-z]+", skills_text.lower()))
        text_scores = np.zeros(len(class_labels))

        for i, label in enumerate(class_labels):
            # Normalise stored label to match CAREER_KEYWORD_MAP keys  
            clean_label = label.replace(" ", "_").replace("/", "_")
            keywords = self.keyword_map.get(clean_label, [])
            if not keywords:
                # try case-insensitive key lookup
                for k, v in self.keyword_map.items():
                    if k.lower() == clean_label.lower():
                        keywords = v
                        break
            if keywords:
                keyword_tokens = set(
                    word for kw in keywords for word in kw.lower().split()
                )
                matches = len(tokens & keyword_tokens)
                text_scores[i] = matches / max(1, len(keyword_tokens))

        if text_scores.sum() == 0:
            return class_probs

        # Normalise text scores to a probability distribution
        text_probs = text_scores / text_scores.sum()

        # Blend
        blended = (1 - self.weight) * class_probs + self.weight * text_probs
        return blended / blended.sum()  # renormalise


@dataclass
class CareerDatasetBuilder:
    df: pd.DataFrame
    feature_columns: list
    target_column: str = TARGET_COLUMN

    def build(self) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            X = self.df[self.feature_columns].copy()
            y = self.df[self.target_column].apply(self._clean_label)
            return X, y
        except KeyError as exc:
            missing = set(self.feature_columns + [self.target_column]) - set(self.df.columns)
            raise ValueError(f"Missing required columns: {missing}") from exc

    @staticmethod
    def _clean_label(label):
        if isinstance(label, str):
            clean = re.sub(r"[\[\]\"']", "", label)
            return clean.strip()
        return label


class ModelStorage:
    def __init__(self, path: str):
        self.path = path

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def save_model(self, model, encoder, features):
        """Save the model, encoder, and features to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        joblib.dump({"model": model, "encoder": encoder, "features": features}, self.path)
        logger.info("Career predictor model saved to %s", self.path)

    def load_model(self):
        """Load the model, encoder, and features from disk."""
        if not self.exists():
            raise FileNotFoundError(f"No saved model found at {self.path}")
        logger.info("Loading career predictor model from %s", self.path)
        data = joblib.load(self.path)
        logger.info("Career predictor model loaded successfully")
        return data["model"], data["encoder"], data["features"]

class CareerPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RF_RANDOM_STATE,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            class_weight=RF_CLASS_WEIGHT,
            n_jobs=-1,
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_columns = FEATURE_COLUMNS
        self.storage = ModelStorage(model_path)
        self.booster = SkillsTextBooster()

    def load_or_train(self, df, verbose=True):
        """Load a saved model if available, otherwise train a new one."""
        if self.storage.exists():
            self._load_model()
        else:
            self.train(df, verbose=verbose)

    def train(self, df, verbose=True):
        try:
            if verbose:
                logger.info("\n%s", "=" * 40)
                logger.info("   TRAINING CAREER PREDICTOR MODEL")
                logger.info("%s", "=" * 40)
                logger.info("Preparing training data...")

            dataset = CareerDatasetBuilder(df, self.feature_columns)
            X, y = dataset.build()

            if verbose:
                logger.info("Unique classes found: %s", y.unique())

            y_encoded = self.label_encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=TEST_SIZE, random_state=TRAIN_RANDOM_STATE
            )

            if verbose:
                logger.info("Fitting Random Forest Classifier...")
            self.model.fit(X_train, y_train)
            self._mark_trained()

            if verbose:
                logger.info("\n--- Model Evaluation Results ---")
                accuracy = self.model.score(X_test, y_test)
                logger.info("Accuracy: %.2f%%", accuracy * 100)

                y_pred = self.model.predict(X_test)
                logger.info("\nClassification Report:\n%s", classification_report(
                    y_test, y_pred, target_names=self.label_encoder.classes_
                ))

            self._save_model()
        except Exception as exc:
            raise RuntimeError("Failed to train career predictor") from exc

    def predict(self, user_input, skills_text: str = ""):
        try:
            input_df = self._prepare_input(user_input)

            probs = self.model.predict_proba(input_df)[0]
            probs = self.booster.boost(probs, self.label_encoder.classes_, skills_text)

            pred_idx = int(np.argmax(probs))
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])

            return pred_label, confidence
        except Exception as exc:
            raise RuntimeError("Failed to generate career prediction") from exc

    def predict_top_k(self, user_input, k=3, skills_text: str = ""):
        """Returns the top k predictions with their probabilities."""
        try:
            input_df = self._prepare_input(user_input)

            probs = self.model.predict_proba(input_df)[0]
            probs = self.booster.boost(probs, self.label_encoder.classes_, skills_text)

            top_k_indices = np.argsort(probs)[-k:][::-1]

            results = []
            for idx in top_k_indices:
                label = self.label_encoder.inverse_transform([idx])[0]
                prob = float(probs[idx])
                results.append((label, prob))

            return results
        except Exception as exc:
            raise RuntimeError("Failed to compute top career predictions") from exc
    
    def _prepare_input(self, user_input: dict) -> pd.DataFrame:
        self._ensure_model_loaded()
        try:
            missing = [col for col in self.feature_columns if col not in user_input]
            if missing:
                raise ValueError(f"Missing required input fields: {set(missing)}")

            invalid_types = {
                col: type(user_input[col]).__name__
                for col in self.feature_columns
                if not isinstance(user_input[col], int)
            }
            if invalid_types:
                issues = ", ".join(f"{col} ({dtype})" for col, dtype in invalid_types.items())
                raise TypeError(
                    "Subject scores must be integers. Invalid values detected for: "
                    + issues
                )

            ordered_input = {col: user_input[col] for col in self.feature_columns}
            return pd.DataFrame([ordered_input])
        except (ValueError, TypeError):
            raise
        except Exception as exc:
            raise RuntimeError("Failed to prepare input for prediction") from exc

    def _ensure_model_loaded(self) -> None:
        """Ensure the model is trained or loaded from disk."""
        if not self.is_trained:
            if self.storage.exists():
                self._load_model()
            else:
                raise RuntimeError("Model is not trained yet.")

    def _mark_trained(self) -> None:
        """Mark the model as successfully trained."""
        self.is_trained = True

    def _save_model(self) -> None:
        """Save the trained model to disk."""
        self.storage.save_model(self.model, self.label_encoder, self.feature_columns)

    def _load_model(self) -> None:
        """Load the model from disk."""
        self.model, self.label_encoder, self.feature_columns = self.storage.load_model()
        self._mark_trained()
