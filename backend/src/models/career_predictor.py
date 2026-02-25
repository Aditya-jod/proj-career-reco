"""
CareerPredictor — Random Forest classifier trained on numeric academic scores.

Single Responsibility: provides the RF numeric / soft-skill signal only.
All SBERT text encoding lives in SBERTCareerClassifier.
All ensemble blending lives in CareerService.
(SRP / SOLID)
"""
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
            raise FileNotFoundError(
                f"No saved model found at {self.path}. "
                "Run 'python scripts/retrain_models.py' to train models first."
            )
        logger.info("Loading career predictor model from %s", self.path)
        data = joblib.load(self.path)
        logger.info("Career predictor model loaded successfully")
        return data["model"], data["encoder"], data["features"]

class CareerPredictor:
    """
    Random Forest career predictor trained on numeric academic scores.

    Single Responsibility: RF numeric / soft-skill signal only.
    Blending with SBERT semantic signal is the responsibility of CareerService.
    """

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self.model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RF_RANDOM_STATE,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            class_weight=RF_CLASS_WEIGHT,  # type: ignore[arg-type]  # validated str literal
            n_jobs=-1,
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_columns = FEATURE_COLUMNS
        self.storage = ModelStorage(model_path)

    def load(self) -> None:
        """Load a pre-trained model from disk.

        Raises FileNotFoundError with an actionable message when the model
        hasn't been trained yet.  Train first with::

            python scripts/retrain_models.py
        """
        self._load_model()

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
            results = self.predict_top_k(user_input, k=1, skills_text=skills_text)
            return results[0]
        except Exception as exc:
            raise RuntimeError("Failed to generate career prediction") from exc

    def predict_proba(self, user_input: dict) -> Tuple[np.ndarray, List[str]]:
        """Return (probability_array, label_list) for ALL career classes.

        Used by CareerService to blend with SBERT probabilities.
        """
        try:
            input_df = self._prepare_input(user_input)
            probs = self.model.predict_proba(input_df)[0]   # (n_classes,)
            labels: List[str] = list(self.label_encoder.classes_)
            return probs, labels
        except Exception as exc:
            raise RuntimeError("Failed to compute RF career probabilities") from exc

    def predict_top_k(
        self,
        user_input: dict,
        k: int = 3,
        skills_text: str = "",
    ) -> List[Tuple[str, float]]:
        """Return top-k RF-only career predictions.

        skills_text is accepted for API compatibility but is intentionally
        ignored here — blending is CareerService's concern.
        """
        try:
            probs, labels = self.predict_proba(user_input)
            top_indices = np.argsort(probs)[-k:][::-1]
            return [(labels[i], float(probs[i])) for i in top_indices]
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
        """Ensure the model is loaded; raise clearly if not trained yet."""
        if not self.is_trained:
            if self.storage.exists():
                self._load_model()
            else:
                raise RuntimeError(
                    "Career predictor model not found. "
                    "Run 'python scripts/retrain_models.py' first."
                )

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
