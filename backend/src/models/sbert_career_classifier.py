"""
SBERTCareerClassifier — supervised career field classifier.

A LogisticRegression head trained on SBERT embeddings of labeled
skills_text -> career_field pairs.  The classifier is trained offline
via ``scripts/retrain_models.py`` and loaded at API startup.

The encoder (FeatureBuilder) is injected via the constructor.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.features.build_features import FeatureBuilder
from src.models.config import SBERT_CLASSIFIER_PATH

logger = logging.getLogger(__name__)


class SBERTCareerClassifier:
    """
    Classifies career field using a LogisticRegression head on SBERT embeddings.

    Trained offline via ``scripts/retrain_models.py``.
    Loaded at API startup via ``load()``.

    Parameters
    ----------
    encoder : FeatureBuilder
        The shared Sentence-BERT encoder instance.
    model_path : str | Path, optional
        Disk location of the trained classifier artefact.
    """

    def __init__(
        self,
        encoder: FeatureBuilder,
        model_path: str | Path = SBERT_CLASSIFIER_PATH,
    ) -> None:
        self._encoder = encoder
        self._model_path = Path(model_path)
        self._clf: LogisticRegression | None = None
        self._label_encoder: LabelEncoder | None = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load a trained LogisticRegression classifier from disk.

        Raises ``FileNotFoundError`` if the model file does not exist.
        Run ``python scripts/retrain_models.py`` to train first.
        """
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Trained SBERT classifier not found at {self._model_path}. "
                "Run 'python scripts/retrain_models.py' to train the model first."
            )

        data = joblib.load(self._model_path)
        self._clf = data["model"]
        self._label_encoder = data["label_encoder"]
        assert self._label_encoder is not None, "Loaded artefact missing label_encoder"
        logger.info(
            "SBERT classifier loaded from %s (%d classes)",
            self._model_path,
            len(self._label_encoder.classes_),
        )

    def save(self) -> None:
        """Persist the trained classifier to disk."""
        if self._clf is None or self._label_encoder is None:
            raise RuntimeError("Cannot save — classifier has not been trained yet.")
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._clf, "label_encoder": self._label_encoder},
            self._model_path,
        )
        logger.info("SBERT classifier saved to %s", self._model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: List[str],
        labels: List[str],
        *,
        max_iter: int = 1000,
        C: float = 1.0,
    ) -> None:
        """Train a LogisticRegression classifier on SBERT embeddings.

        Parameters
        ----------
        texts : List[str]
            Raw skills / interests / academic text for each sample.
        labels : List[str]
            Corresponding career field label for each sample.
        max_iter : int
            Maximum solver iterations.
        C : float
            Inverse regularisation strength.
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")

        logger.info("Encoding %d training samples with SBERT…", len(texts))
        X = self._encoder.encode(texts, normalize=True)

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(labels)

        logger.info(
            "Training LogisticRegression — %d classes, %d samples…",
            len(self._label_encoder.classes_),
            int(len(y)),  # type: ignore[arg-type]  # ndarray is Sized
        )
        self._clf = LogisticRegression(
            max_iter=max_iter,
            C=C,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        self._clf.fit(X, y)
        logger.info("Classifier training complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """Return (probability_array, label_list).

        Raises ``RuntimeError`` if the classifier has not been loaded/trained.
        """
        if self._clf is None or self._label_encoder is None:
            raise RuntimeError(
                "Classifier not loaded. Call load() or train() first."
            )
        query_emb = self._encoder.encode([text], normalize=True)
        probs = self._clf.predict_proba(query_emb)[0]
        labels: List[str] = [str(c) for c in self._label_encoder.classes_]
        return probs, labels
