"""
CareerService — single-responsibility orchestrator for career prediction.

Wraps SBERTCareerClassifier and returns the top-k career predictions.
The classifier uses a trained LogisticRegression head on SBERT embeddings.

Usage::

    sbert_clf = SBERTCareerClassifier(encoder=feature_builder)
    sbert_clf.load()   # loads trained classifier from disk

    service = CareerService(classifier=sbert_clf)
    top3 = service.predict_top_k(user_input, k=3, skills_text=text)
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from src.models.sbert_career_classifier import SBERTCareerClassifier

logger = logging.getLogger(__name__)


class CareerService:
    """
    Predicts career fields by delegating to the SBERT classifier.

    Parameters
    ----------
    classifier : SBERTCareerClassifier
        A trained classifier exposing ``predict_proba(text) -> (probs, labels)``.
    """

    def __init__(self, classifier: SBERTCareerClassifier) -> None:
        self._clf = classifier
        logger.info("CareerService ready.")

    def predict_top_k(
        self,
        user_input: dict,
        k: int = 3,
        skills_text: str = "",
    ) -> List[Tuple[str, float]]:
        """Return top-k (career_field, confidence) pairs.

        Parameters
        ----------
        user_input : dict
            Numeric academic scores (currently unused by the text
            classifier but accepted for future blending).
        k : int
            Number of top predictions to return.
        skills_text : str
            Free-text description of skills/interests/stream.  This is the
            primary input to the SBERT classifier.

        Returns
        -------
        List[Tuple[str, float]]
            Top-k pairs sorted by descending confidence.  Confidences are
            normalised to sum to 1.
        """
        if not skills_text:
            logger.warning(
                "predict_top_k called without skills_text -- results may be poor"
            )
            skills_text = "general career guidance"

        probs, labels = self._clf.predict_proba(skills_text)

        # Normalise to a valid probability distribution
        total = probs.sum()
        if total > 0:
            probs = probs / total

        top_indices = np.argsort(probs)[-k:][::-1]
        return [(labels[i], float(probs[i])) for i in top_indices]
