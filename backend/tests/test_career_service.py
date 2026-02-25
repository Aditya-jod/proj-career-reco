"""
Unit tests for CareerService and SBERTCareerClassifier.

These tests use a lightweight stub encoder so they run in < 1 second
without downloading or loading a real SBERT model (~80 MB).

Run:
    cd backend
    python -m pytest tests/test_career_service.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.config import CAREER_DESCRIPTIONS
from src.services.career_service import CareerService


# ---------------------------------------------------------------------------
# Stub encoder (same interface as FeatureBuilder)
# ---------------------------------------------------------------------------
class _StubEncoder:
    """Returns deterministic random embeddings so tests are reproducible."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self._rng = np.random.RandomState(42)

    def encode(self, texts, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self._rng.randn(len(texts), self._dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Stub classifier (same interface as SBERTCareerClassifier)
# ---------------------------------------------------------------------------
class _StubClassifier:
    """Minimal classifier that returns fixed probabilities for testing."""

    def __init__(self, labels: list[str] | None = None) -> None:
        self._labels = labels or list(CAREER_DESCRIPTIONS.keys())

    def predict_proba(self, text: str):
        # Return higher prob for the first label
        probs = np.array([0.3] + [0.1] * (len(self._labels) - 1))
        return probs, self._labels


# ---------------------------------------------------------------------------
# Tests — CareerService
# ---------------------------------------------------------------------------
class TestCareerService:
    """Tests for CareerService.predict_top_k."""

    @pytest.fixture
    def service(self) -> CareerService:
        return CareerService(classifier=_StubClassifier())  # type: ignore[arg-type]

    def test_returns_k_results(self, service: CareerService) -> None:
        results = service.predict_top_k({}, k=3, skills_text="data science")
        assert len(results) == 3

    def test_results_sorted_descending(self, service: CareerService) -> None:
        results = service.predict_top_k({}, k=5, skills_text="programming")
        confidences = [conf for _, conf in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_returns_tuples_of_str_float(self, service: CareerService) -> None:
        results = service.predict_top_k({}, k=2, skills_text="art design")
        for label, conf in results:
            assert isinstance(label, str)
            assert isinstance(conf, float)

    def test_top_result_is_highest_prob_label(self, service: CareerService) -> None:
        results = service.predict_top_k({}, k=1, skills_text="test")
        # Our stub puts 0.3 on the first label
        expected_labels = list(CAREER_DESCRIPTIONS.keys())
        assert results[0][0] == expected_labels[0]

    def test_confidences_sum_to_one(self, service: CareerService) -> None:
        results = service.predict_top_k(
            {}, k=len(CAREER_DESCRIPTIONS), skills_text="test all"
        )
        total = sum(conf for _, conf in results)
        assert abs(total - 1.0) < 1e-6

    def test_empty_skills_text_still_returns(self, service: CareerService) -> None:
        """When skills_text is empty the service should gracefully fall back."""
        results = service.predict_top_k({}, k=3, skills_text="")
        assert len(results) == 3

    def test_k_greater_than_labels(self, service: CareerService) -> None:
        """Requesting more results than labels should not crash."""
        results = service.predict_top_k(
            {}, k=100, skills_text="over-request"
        )
        assert len(results) <= len(CAREER_DESCRIPTIONS)


# ---------------------------------------------------------------------------
# Tests — SBERTCareerClassifier
# ---------------------------------------------------------------------------
class TestSBERTClassifier:
    """Tests for SBERTCareerClassifier.train / predict_proba."""

    @pytest.fixture
    def trained_classifier(self):
        from src.models.sbert_career_classifier import SBERTCareerClassifier

        clf = SBERTCareerClassifier(encoder=_StubEncoder())  # type: ignore[arg-type]
        # Minimal training data — 3 samples per class
        texts = []
        labels = []
        for label in list(CAREER_DESCRIPTIONS.keys())[:3]:
            for _ in range(3):
                texts.append(f"sample text for {label}")
                labels.append(label)
        clf.train(texts, labels)
        return clf

    def test_raises_before_load_or_train(self) -> None:
        from src.models.sbert_career_classifier import SBERTCareerClassifier

        clf = SBERTCareerClassifier(encoder=_StubEncoder())  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="not loaded"):
            clf.predict_proba("test")

    def test_raises_on_missing_model_file(self, tmp_path) -> None:
        from src.models.sbert_career_classifier import SBERTCareerClassifier

        clf = SBERTCareerClassifier(
            encoder=_StubEncoder(),  # type: ignore[arg-type]
            model_path=tmp_path / "nonexistent.pkl",
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            clf.load()

    def test_predict_after_training(self, trained_classifier) -> None:
        probs, labels = trained_classifier.predict_proba("engineering")
        assert len(probs) == len(labels)
        assert len(labels) == 3  # only 3 classes in training data

    def test_probs_sum_to_one(self, trained_classifier) -> None:
        probs, _ = trained_classifier.predict_proba("test")
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_save_and_load(self, trained_classifier, tmp_path) -> None:
        from src.models.sbert_career_classifier import SBERTCareerClassifier

        save_path = tmp_path / "test_model.pkl"
        trained_classifier._model_path = save_path
        trained_classifier.save()

        loaded = SBERTCareerClassifier(encoder=_StubEncoder(), model_path=save_path)  # type: ignore[arg-type]
        loaded.load()
        probs, labels = loaded.predict_proba("engineering")
        assert len(probs) == len(labels) == 3
