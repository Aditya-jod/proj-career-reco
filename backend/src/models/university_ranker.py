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


# Patterns in college names that indicate coaching / skill centres
# rather than actual universities.  Used to penalise low-quality entries.
_COACHING_PATTERNS = re.compile(
    r"\b(career\s+(computer|academy|dimension|institute)|"
    r"skill\s*(initiate|development|institute)|"
    r"computer\s+education|"
    r"coaching\s+centre|"
    r"tutorial|"
    r"cats\b.*career\s+academy)\b",
    re.IGNORECASE,
)

# Name patterns for elite / nationally-recognised institutions.
# Used to boost IITs, NITs, IIITs, IISc, BITS, top private universities
# **and** world-class global universities that appear in the world_universities
# dataset (they lack the Indian quality columns).
_ELITE_NAME_PATTERNS = re.compile(
    r"\b("
    r"Indian Institute of Technology|"
    r"National Institute of Technology|"
    r"Indian Institute of Information Technology|"
    r"Indian Institute of Science|"
    r"Indian Statistical Institute|"
    r"Birla Institute of Technology|"
    r"Vellore Institute of Technology|"
    r"Anna University|"
    r"Jadavpur University|"
    r"University of Delhi|"
    r"Jawaharlal Nehru University|"
    r"Banaras Hindu University|"
    r"Aligarh Muslim University|"
    r"BITS\s+Pilani|"
    r"\bIIT\b|"
    r"\bNIT\b|"
    r"\bIIIT\b|"
    r"\bIISc\b|"
    r"\bISI\b"
    r")\b",
    re.IGNORECASE,
)


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
        name = str(university_row.get("name", ""))
        name_tokens = _tokenize(name)

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

        # ── Quality signals ──────────────────────────────────────────────
        # Specialisation — a college with a declared specialisation in the
        # area of interest is worth far more than one with "No" specialism.
        specialisation_raw = str(university_row.get("Specialised in", "")).strip()
        has_real_specialisation = (
            1.0 if specialisation_raw and specialisation_raw.lower() not in ("no", "nan", "none", "")
            else 0.0
        )

        # College type quality — constituent / university colleges are
        # generally higher-quality than random affiliated colleges.
        # World universities (no College Type) are actual universities by
        # definition, so they get a base quality bonus.
        college_type = str(university_row.get("College Type", "")).strip().lower()
        is_world_university = (not college_type) or college_type in ("nan", "none")
        is_university_college = (
            1.0 if (
                "university" in college_type
                or "autonomous" in college_type
                or is_world_university  # world_universities entries are real universities
            ) else 0.0
        )

        # University type quality — Central University / IIT / NIT / deemed
        # are stronger signals than generic state universities.
        # Also match elite institution names directly (covers IITs from
        # world_universities that have no University Type metadata).
        uni_type = str(university_row.get("University Type", "")).strip().lower()
        is_premier = (
            1.0 if (
                any(k in uni_type for k in ("central", "national importance", "deemed"))
                or _ELITE_NAME_PATTERNS.search(name)
            ) else 0.0
        )

        # Coaching-centre penalty — small skill/career/computer institutes
        # should score lower.
        is_coaching = 1.0 if _COACHING_PATTERNS.search(name) else 0.0

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
            "name_len": len(name),
            "country_is_india": 1.0 if country.lower().strip() == "india" else 0.0,
            "keyword_boost": self._keyword_boost(specialization_tokens),
            # Quality features
            "has_real_specialisation": has_real_specialisation,
            "is_university_college": is_university_college,
            "is_premier": is_premier,
            "is_coaching": is_coaching,
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
        score += 0.15 * features["specialization_match"]
        score += 0.10 * features["specialization_overlap"]
        score += 0.10 * features["country_match"]
        score += 0.05 * features["state_match"]
        score += 0.05 * features["name_overlap"]
        score += 0.05 * features["keyword_boost"]
        # Quality bonuses / penalties — weighted heavily so that premier
        # institutions consistently outrank small unspecialised colleges.
        score += 0.15 * features["has_real_specialisation"]
        score += 0.15 * features["is_university_college"]
        score += 0.25 * features["is_premier"]
        score -= 0.40 * features["is_coaching"]
        return float(max(0.0, min(score, 1.0)))


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
        if not self.is_ready() or self.model is None:
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
