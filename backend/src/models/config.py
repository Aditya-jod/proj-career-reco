"""
Configuration constants for all ML model components.

All tunable values are loaded from ``backend/config/config.yaml`` at import
time.  Hardcoded defaults are used only if the YAML file is missing or a
key is absent — this lets the system function out-of-the-box while allowing
one-file tweaks for production or A/B experiments.

NOTE: CAREER_DESCRIPTIONS is kept here ONLY as a fallback for the
SBERT classifier when MongoDB has not been seeded yet.  The canonical
source of career metadata is the ``careers`` collection in MongoDB,
populated by ``scripts/seed_careers.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # …/proj-career-reco

_cfg: Optional[Dict[str, Any]] = None

try:
    import yaml

    _CONFIG_PATH = _PROJECT_ROOT / "backend" / "config" / "config.yaml"
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r") as _f:
            _cfg = yaml.safe_load(_f)
        logger.debug("Loaded config from %s", _CONFIG_PATH)
    else:
        logger.warning("config.yaml not found at %s — using built-in defaults", _CONFIG_PATH)
except Exception as _exc:  # pragma: no cover
    logger.warning("Failed to load config.yaml: %s — using built-in defaults", _exc)

_datasets_cfg: Dict[str, Any] = (_cfg or {}).get("datasets", {})
_model_cfg: Dict[str, Any] = (_cfg or {}).get("model", {})
_rf_cfg: Dict[str, Any] = _model_cfg.get("random_forest", {})

# ---------------------------------------------------------------------------
# Model storage paths
# ---------------------------------------------------------------------------
MODEL_PATH = str(_PROJECT_ROOT / "models" / "career_predictor.pkl")
SBERT_CLASSIFIER_PATH = str(_PROJECT_ROOT / "models" / "sbert_career_classifier.pkl")

# Rich career field descriptions used by SBERTCareerClassifier.
# Written as natural-language paragraphs so the sentence-transformer
# produces maximally discriminative vectors.
CAREER_DESCRIPTIONS: Dict[str, str] = {
    "Healthcare": (
        "Medicine, healthcare, and life sciences. Careers include doctor, physician, "
        "surgeon, nurse, dentist, pharmacist, physiotherapist, radiologist, pediatrician, "
        "cardiologist, neurologist, oncologist, psychiatrist, and medical researcher. "
        "Academic streams: Science PCB, biology, chemistry, anatomy, physiology. "
        "Activities: patient care, clinical diagnosis, hospital work, surgery, therapy, "
        "biomedical research, public health, nutrition, and pharmaceutical sciences."
    ),
    "STEM": (
        "Science, technology, engineering, and mathematics. Careers include software engineer, "
        "data scientist, AI researcher, mechanical engineer, electrical engineer, civil engineer, "
        "aerospace engineer, chemical engineer, mathematician, physicist, and biotechnologist. "
        "Academic streams: Science PCM, computer science, physics, mathematics, calculus. "
        "Skills: programming, coding, Python, Java, machine learning, deep learning, "
        "data analysis, algorithms, robotics, cloud computing, cybersecurity, statistics."
    ),
    "Business_Finance": (
        "Business, finance, economics, and commerce. Careers include entrepreneur, "
        "chartered accountant, financial analyst, investment banker, marketing manager, "
        "product manager, business consultant, supply chain manager, HR manager, "
        "stock trader, and operations director. "
        "Academic streams: Commerce, economics, business studies, accountancy. "
        "Skills: accounting, budgeting, market analysis, sales, strategy, MBA, finance."
    ),
    "Arts_Media": (
        "Creative arts, design, and media. Careers include graphic designer, filmmaker, "
        "journalist, animator, photographer, fashion designer, architect, musician, "
        "content creator, advertising professional, interior designer, and game designer. "
        "Academic streams: Arts, humanities, fine arts, performing arts, literature. "
        "Skills: design, illustration, storytelling, video editing, photography, "
        "creative writing, music, theatre, cinematography, and branding."
    ),
    "Education": (
        "Teaching, training, and academic instruction. Careers include school teacher, "
        "university professor, curriculum developer, educational researcher, corporate trainer, "
        "academic counselor, e-learning developer, and edtech specialist. "
        "Skills: pedagogy, curriculum design, classroom management, tutoring, coaching, "
        "lesson planning, educational technology, and student assessment."
    ),
    "Social_Services": (
        "Social work, community welfare, and counseling. Careers include social worker, "
        "psychologist, counsellor, NGO worker, community organizer, mental health therapist, "
        "rehabilitation specialist, child welfare officer, and human rights advocate. "
        "Academic streams: Social science, sociology, psychology, humanities. "
        "Skills: empathy, counseling, community service, advocacy, mental health support."
    ),
    "Government_Law": (
        "Law, government, and public administration. Careers include lawyer, advocate, "
        "judge, IAS officer, IPS officer, civil servant, diplomat, policy analyst, "
        "politician, compliance officer, and public prosecutor. "
        "Academic streams: Law, political science, public administration. "
        "Exams and paths: UPSC, civil services, bar council, judiciary. "
        "Skills: legal reasoning, legislation, public policy, governance, diplomacy."
    ),
    "Trades_Manufacturing": (
        "Skilled trades, manufacturing, and vocational careers. Careers include electrician, "
        "plumber, carpenter, welder, machinist, automotive technician, HVAC technician, "
        "industrial engineer, and construction manager. "
        "Academic streams: ITI, polytechnic, diploma, vocational training. "
        "Skills: mechanical work, electrical installation, fabrication, tool operation, "
        "construction, maintenance, welding, and manufacturing processes."
    ),
}

# Display-friendly titles for each career field
CAREER_TITLES: Dict[str, str] = {
    "Healthcare": "Healthcare & Life Sciences",
    "STEM": "Science, Technology, Engineering & Mathematics",
    "Business_Finance": "Business & Finance",
    "Arts_Media": "Arts, Design & Media",
    "Education": "Education & Training",
    "Social_Services": "Social Services & Counseling",
    "Government_Law": "Government & Law",
    "Trades_Manufacturing": "Trades & Manufacturing",
}

FEATURE_COLUMNS: List[str] = _datasets_cfg.get("feature_columns", [
    "Mathematics_Score",
    "Science_Score",
    "Language_Arts_Score",
    "Social_Studies_Score",
    "Logical_Reasoning",
    "Creativity",
    "Communication",
    "Leadership",
    "Social_Skills",
])

TARGET_COLUMN: str = _datasets_cfg.get(
    "target_column", "Primary_Career_Recommendation"
)

RF_N_ESTIMATORS: int = _rf_cfg.get("n_estimators", 300)
RF_RANDOM_STATE: int = _rf_cfg.get("random_state", 42)
RF_MAX_DEPTH: Optional[int] = _rf_cfg.get("max_depth", None)
RF_MIN_SAMPLES_SPLIT: int = _rf_cfg.get("min_samples_split", 5)
RF_CLASS_WEIGHT: str = _rf_cfg.get("class_weight", "balanced")  # type: ignore[assignment]  # always "balanced" or "balanced_subsample"

TEST_SIZE: float = _model_cfg.get("test_size", 0.2)
TRAIN_RANDOM_STATE: int = _model_cfg.get("train_random_state", 42)
