import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.country_utils import country_matches

logger = logging.getLogger(__name__)

# Default local storage for the trained ranker
DEFAULT_MODEL_PATH = Path("models/university_ranker.pkl")


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return set(tokens)


@dataclass
class UniversityContext:
    career_text: str
    skills_text: str = ""
    preferred_country: Optional[str] = None
    preferred_state: Optional[str] = None
    keywords: set[str] = field(init=False, default_factory=set)

    def __post_init__(self):
        combined = " ".join(part for part in [self.career_text, self.skills_text] if part)
        self.keywords = _tokenize(combined)


class UniversityFeatureEngineer:
    KEYWORD_BOOST = {"health", "medical", "engineering", "business", "finance", "science"}

    def build_features(self, university_row: pd.Series, context: UniversityContext) -> dict:
        search_text = str(university_row.get("search_text", ""))
        specialization_tokens = _tokenize(search_text)
        name_tokens = _tokenize(str(university_row.get("name", "")))

        keyword_overlap = len(context.keywords & specialization_tokens)
        keyword_overlap_ratio = (
            keyword_overlap / max(1, len(context.keywords)) if context.keywords else 0.0
        )

        name_overlap = len(context.keywords & name_tokens)
        name_overlap_ratio = name_overlap / max(1, len(context.keywords)) if context.keywords else 0.0

        country = str(university_row.get("country", ""))
        state = str(university_row.get("State", ""))
        district = str(university_row.get("District", ""))
        website = str(university_row.get("Website", ""))

        country_match = country_matches(context.preferred_country, country)
        state_match = self._matches(context.preferred_state, state)

        features = {
            "specialization_match": 1.0 if keyword_overlap > 0 else 0.0,
            "specialization_overlap": keyword_overlap_ratio,
            "name_overlap": name_overlap_ratio,
            "country_match": 1.0 if country_match else 0.0,
            "state_match": 1.0 if state_match else 0.0,
            "has_state": 0.0 if state in {"", "nan", "None"} else 1.0,
            "has_district": 0.0 if district in {"", "nan", "None"} else 1.0,
            "has_website": 0.0 if website in {"", "nan", "None"} else 1.0,
            "search_text_len": len(search_text),
            "name_len": len(str(university_row.get("name", ""))),
            "country_is_india": 1.0 if country.lower().strip() == "india" else 0.0,
            "keyword_boost": self._keyword_boost(specialization_tokens),
        }
        return features

    @staticmethod
    def _matches(preference: Optional[str], candidate: str) -> bool:
        if not preference or not candidate:
            return False
        return preference.strip().lower() in candidate.strip().lower()

    def _keyword_boost(self, specialization_tokens: set[str]) -> float:
        overlap = len(self.KEYWORD_BOOST & specialization_tokens)
        return overlap / max(1, len(self.KEYWORD_BOOST))

class FeatureMatrixBuilder:
    """Converts raw university rows into a feature matrix for the ranker."""

    def __init__(self, engineer: Optional[UniversityFeatureEngineer] = None):
        self.engineer = engineer or UniversityFeatureEngineer()

    def build(self, universities: pd.DataFrame, context: UniversityContext) -> pd.DataFrame:
        feature_rows = [self.engineer.build_features(row, context) for _, row in universities.iterrows()]
        return pd.DataFrame(feature_rows)


class BaseLabelStrategy(ABC):
    """Defines how synthetic labels are produced for training."""

    @abstractmethod
    def score(self, features: dict) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class HeuristicLabelStrategy(BaseLabelStrategy):
    """Default rule-based scoring to avoid brittle hardcoded logic elsewhere."""

    def score(self, features: dict) -> float:
        score = 0.1
        score += 0.5 * features["specialization_match"]
        score += 0.3 * features["specialization_overlap"]
        score += 0.2 * features["country_match"]
        score += 0.1 * features["state_match"]
        score += 0.1 * features["name_overlap"]
        score += 0.1 * features["keyword_boost"]
        return float(min(score, 1.0))


class UniversityRankerModel:
    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        feature_matrix_builder: Optional[FeatureMatrixBuilder] = None,
    ):
        self.model_path = Path(model_path)
        self.model: Optional[RandomForestRegressor] = None
        self.feature_columns: list[str] = []
        self.matrix_builder = feature_matrix_builder or FeatureMatrixBuilder()
        self._load_if_available()

    def _load_if_available(self):
        if not self.model_path.exists():
            return
        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        logger.info("University ranker loaded from %s", self.model_path)

    def is_ready(self) -> bool:
        return self.model is not None

    def rank(
        self,
        universities: pd.DataFrame,
        context: UniversityContext,
        top_k: int = 10,
    ) -> pd.DataFrame:
        if not self.is_ready():
            raise RuntimeError("University ranker model is not loaded")

        feature_df = self.matrix_builder.build(universities, context)
        feature_df = feature_df[self.feature_columns]
        predictions = self.model.predict(feature_df)

        ranked = universities.copy()
        ranked["ml_score"] = predictions
        ranked = ranked.sort_values("ml_score", ascending=False).head(top_k)
        return ranked


class UniversityRankerTrainer:
    def __init__(
        self,
        feature_matrix_builder: Optional[FeatureMatrixBuilder] = None,
        label_strategy: Optional[BaseLabelStrategy] = None,
        random_state: int = 42,
    ):
        self.matrix_builder = feature_matrix_builder or FeatureMatrixBuilder()
        self.label_strategy = label_strategy or HeuristicLabelStrategy()
        self.rng = np.random.default_rng(random_state)

    def train(
        self,
        universities: pd.DataFrame,
        career_labels: Iterable[str],
        save_path: Path | str = DEFAULT_MODEL_PATH,
        max_careers: int = 20,
        max_countries: int = 10,
        samples_per_context: int = 300,
    ) -> None:
        careers = self._prepare_list(career_labels, max_careers)
        countries = self._prepare_list(universities["country"].dropna().unique(), max_countries)

        if not careers or not countries:
            raise ValueError("Insufficient data to train university ranker")

        rows = []
        for career in careers:
            for country in countries:
                context = UniversityContext(career_text=career, preferred_country=country)
                sample_df = universities.sample(
                    n=min(samples_per_context, len(universities)),
                    random_state=int(self.rng.integers(0, 1_000_000)),
                )
                features_df = self.matrix_builder.build(sample_df, context)
                for feature_map in features_df.to_dict(orient="records"):
                    label = self.label_strategy.score(feature_map)
                    rows.append({**feature_map, "target": label})

        training_df = pd.DataFrame(rows)
        feature_columns = [col for col in training_df.columns if col != "target"]
        X = training_df[feature_columns]
        y = training_df["target"]

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": feature_columns}, save_path)
        logger.info("University ranker trained and saved to %s", save_path)

    def _prepare_list(self, values: Iterable[str], limit: int) -> list[str]:
        cleaned = [str(v).strip() for v in values if isinstance(v, str) and str(v).strip()]
        unique = list(dict.fromkeys(cleaned))
        if len(unique) <= limit:
            return unique
        indices = self.rng.choice(len(unique), size=limit, replace=False)
        return [unique[i] for i in indices]
