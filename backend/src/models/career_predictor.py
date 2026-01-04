import os
import re
from dataclasses import dataclass
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class CareerDatasetBuilder:
    df: pd.DataFrame
    feature_columns: list
    target_column: str = "Primary_Career_Recommendation"

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

    def save(self, model, encoder, features):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        joblib.dump({"model": model, "encoder": encoder, "features": features}, self.path)
        print(f"Model saved successfully to {self.path}")

    def load(self):
        if not self.exists():
            raise FileNotFoundError(f"No saved model found at {self.path}")
        print(f"Loading saved model from {self.path}...")
        data = joblib.load(self.path)
        print("Model loaded successfully.")
        return data["model"], data["encoder"], data["features"]

class CareerPredictor:
    def __init__(self, model_path="models/career_predictor.pkl"):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trianed = False
        self.feature_columns = [
            "Mathematics_Score",
            "Science_Score",
            "Language_Arts_Score",
            "Social_Studies_Score",
            "Logical_Reasoning",
            "Creativity",
            "Communication",
            "Leadership",
            "Social_Skills",
        ]
        self.storage = ModelStorage(model_path)

    def load_or_train(self, df, verbose=True):
        """Load a saved model if available, otherwise train a new one."""
        if self.storage.exists():
            self._load_model()
        else:
            self.train(df, verbose=verbose)

    def train(self, df, verbose=True):
        try:
            if verbose:
                print("\n" + "=" * 40)
                print("   TRAINING CAREER PREDICTOR MODEL")
                print("=" * 40)
                print("Preparing training data...")

            dataset = CareerDatasetBuilder(df, self.feature_columns)
            X, y = dataset.build()

            if verbose:
                print(f"Unique classes found: {y.unique()}")

            y_encoded = self.label_encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )

            if verbose:
                print("Fitting Random Forest Classifier...")
            self.model.fit(X_train, y_train)
            self.is_trianed = True

            if verbose:
                print("\n--- Model Evaluation Results ---")
                accuracy = self.model.score(X_test, y_test)
                print(f"Accuracy: {accuracy:.2%}")

                y_pred = self.model.predict(X_test)
                print("\nClassification Report:")
                print(
                    classification_report(
                        y_test, y_pred, target_names=self.label_encoder.classes_
                    )
                )

            self._save_model()
        except Exception as exc:
            raise RuntimeError("Failed to train career predictor") from exc

    def predict(self, user_input):
        try:
            input_df = self._prepare_input(user_input)

            pred_idx = self.model.predict(input_df)[0]
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]

            probs = self.model.predict_proba(input_df)[0]
            confidence = float(np.max(probs))

            return pred_label, confidence
        except Exception as exc:
            raise RuntimeError("Failed to generate career prediction") from exc

    def predict_top_k(self, user_input, k=3):
        """Returns the top k predictions with their probabilities."""
        try:
            input_df = self._prepare_input(user_input)

            probs = self.model.predict_proba(input_df)[0]
            top_k_indices = np.argsort(probs)[-k:][::-1]

            results = []
            for idx in top_k_indices:
                label = self.label_encoder.inverse_transform([idx])[0]
                prob = probs[idx]
                results.append((label, prob))

            return results
        except Exception as exc:
            raise RuntimeError("Failed to compute top career predictions") from exc
    
    def _prepare_input(self, user_input: dict) -> pd.DataFrame:
        if not self.is_trianed:
            if self.storage.exists():
                self._load_model()
            else:
                raise RuntimeError("Model is not trained yet.")
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

    def _save_model(self):
        self.storage.save(self.model, self.label_encoder, self.feature_columns)

    def _load_model(self):
        self.model, self.label_encoder, self.feature_columns = self.storage.load()
        self.is_trianed = True
