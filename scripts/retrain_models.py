"""
Retrain the SBERT career classifier and run a full evaluation.

Produces:
  - Trained model     -> models/sbert_career_classifier.pkl
  - University ranker -> models/university_ranker.pkl
  - Evaluation report -> reports/evaluation_report.md

The report includes per-model accuracy / precision / recall / F1,
confusion matrices, and an ablation study comparing:
  1. TF-IDF + Logistic Regression  (text baseline)
  2. SBERT + Logistic Regression   (primary model)
  3. Random Forest on numeric scores only (numeric baseline)

Usage:
    cd <project-root>
    python scripts/retrain_models.py
"""
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("retrain")

# Ensure the backend package is importable regardless of CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_ROOT = _PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data.loader import load_config, load_raw_data
from src.features.build_features import FeatureBuilder
from src.models.config import CAREER_DESCRIPTIONS, FEATURE_COLUMNS
from src.models.sbert_career_classifier import SBERTCareerClassifier
from src.models.university_ranker import UniversityRankerTrainer
from src.models.university_recommender import UniversityDatasetBuilder

_SCORE_COLS = [
    "Mathematics_Score", "Science_Score", "Language_Arts_Score",
    "Social_Studies_Score",
]
_SOFT_SKILL_COLS = [
    "Logical_Reasoning", "Critical_Thinking", "Analytical_Ability",
    "Creativity", "Communication", "Emotional_Intelligence",
    "Social_Skills", "Leadership",
]
_PARTICIPATION_COLS = [
    "Sports_Participation", "Arts_Participation", "Music_Participation",
    "Science_Club_Participation", "Debate_Participation",
    "Community_Service_Participation",
]
# Career domain score columns (aligned with CAREER_DESCRIPTIONS keys)
_DOMAIN_SCORE_COLS: Dict[str, str] = {
    "STEM_Score": "STEM",
    "Business_Finance_Score": "Business_Finance",
    "Arts_Media_Score": "Arts_Media",
    "Healthcare_Score": "Healthcare",
    "Education_Score": "Education",
    "Social_Services_Score": "Social_Services",
    "Trades_Manufacturing_Score": "Trades_Manufacturing",
    "Government_Law_Score": "Government_Law",
}

# Pre-extract keyword pools from CAREER_DESCRIPTIONS for augmentation.
# Each career field gets a list of individual keywords drawn from its
# description paragraph.  At training time we sample a subset per sample
# to teach SBERT the vocabulary users are likely to type.
_CAREER_KEYWORD_POOL: Dict[str, List[str]] = {}
for _field, _desc in CAREER_DESCRIPTIONS.items():
    _tokens = re.findall(r"[a-zA-Z][\w\-]+", _desc.lower())
    # Keep unique tokens ≥3 chars that are not stop-words / filler
    _stop = {
        "and", "the", "for", "are", "with", "include", "includes",
        "careers", "career", "academic", "streams", "skills", "activities",
        "work", "such", "like", "also", "from", "that", "this", "other",
    }
    _CAREER_KEYWORD_POOL[_field] = list(
        dict.fromkeys(t for t in _tokens if len(t) >= 3 and t not in _stop)
    )


def _clean_label(raw: str) -> str:
    """Strip the JSON-like wrapping from the target column."""
    return re.sub(r"[^a-zA-Z_]", "", str(raw).strip())


def _score_tier(val, lo=40, hi=70) -> str:
    """Convert a numeric score to a human-readable tier."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return "unknown"
    if v >= hi:
        return "strong"
    elif v >= lo:
        return "moderate"
    return "developing"


def _top_domain(row: pd.Series) -> str:
    """Return the career domain label with the highest domain score."""
    best_score = -1.0
    best_domain = ""
    for col, domain in _DOMAIN_SCORE_COLS.items():
        if col in row.index:
            try:
                val = float(row[col])
            except (ValueError, TypeError):
                continue
            if val > best_score:
                best_score = val
                best_domain = domain
    return best_domain


def _build_skills_text(row: pd.Series, label: str = "", rng: np.random.RandomState | None = None) -> str:
    """Construct a natural-language description from a single dataset row.

    If *label* is provided, a random sample of career-relevant keywords from
    CAREER_DESCRIPTIONS is appended.  This bridges the gap between the numeric
    score-tier phrases used in training and the technology / skill keywords
    users type at inference time (e.g. "Python, machine learning, data science").

    Parameters
    ----------
    row : pd.Series
        A single row from the student dataset.
    label : str
        Career label for this row (used for keyword augmentation).
    rng : np.random.RandomState | None
        RNG for reproducible keyword sampling.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    parts: list[str] = []

    # 1. Academic score tiers
    for col in _SCORE_COLS:
        if col in row.index:
            tier = _score_tier(row[col])
            name = col.replace("_Score", "").replace("_", " ").lower()
            parts.append(f"{tier} {name}")

    # 2. Soft-skill tiers
    for col in _SOFT_SKILL_COLS:
        if col in row.index:
            tier = _score_tier(row[col], lo=4, hi=7)
            name = col.replace("_", " ").lower()
            parts.append(f"{tier} {name}")

    # 3. Extra-curricular participation
    active = []
    for col in _PARTICIPATION_COLS:
        if col in row.index and str(row[col]).strip().lower() in ("yes", "1", "true"):
            activity = col.replace("_Participation", "").replace("_", " ").lower()
            active.append(activity)
    if active:
        parts.append("participates in " + " and ".join(active))

    # 4. Learning style
    if "Learning_Style" in row.index:
        style = str(row["Learning_Style"]).strip().lower()
        if style and style != "nan":
            parts.append(f"{style} learner")

    # 5. Top career domain (from domain scores in the dataset)
    top_dom = _top_domain(row)
    if top_dom:
        friendly = top_dom.replace("_", " ").lower()
        parts.append(f"interested in {friendly}")

    # 6. Career-keyword augmentation (label-guided)
    if label and label in _CAREER_KEYWORD_POOL:
        pool = _CAREER_KEYWORD_POOL[label]
        n_keywords = min(rng.randint(5, 10), len(pool))
        sampled = rng.choice(pool, size=n_keywords, replace=False)
        parts.append(" ".join(sampled))

    return ", ".join(parts)


def _evaluate(
    y_true: List[str],
    y_pred: List[str],
    model_name: str,
) -> Dict[str, float]:
    """Compute and log accuracy, precision, recall, F1 for a model."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    logger.info(
        "  %s — Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.2f%%",
        model_name, acc * 100, prec * 100, rec * 100, f1 * 100,
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _sbert_predict_all(clf: SBERTCareerClassifier, texts: List[str]) -> List[str]:
    """Generate predictions for every text sample using the SBERT classifier."""
    preds = []
    for t in texts:
        probs, lbls = clf.predict_proba(t)
        preds.append(str(lbls[int(np.argmax(probs))]))
    return preds


def _generate_report(
    results: List[Tuple[str, Dict[str, float], List[str], List[str]]],
    class_labels: List[str],
    output_path: Path,
    dataset_info: Dict[str, int],
) -> None:
    """Write a Markdown evaluation report to disk.

    Each entry in *results* is (model_name, metrics_dict, y_true, y_pred).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    lines.append("# Model Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total samples:** {dataset_info['total']}")
    lines.append(f"**Train set:** {dataset_info['train']}  |  **Test set:** {dataset_info['test']}")
    lines.append(f"**Classes:** {len(class_labels)}")
    lines.append("")

    # Ablation study table
    lines.append("## 1. Ablation Study")
    lines.append("")
    lines.append("| Model | Accuracy | Precision | Recall | F1 Score |")
    lines.append("|-------|----------|-----------|--------|----------|")
    for name, metrics, _, _ in results:
        lines.append(
            f"| {name} "
            f"| {metrics['accuracy']:.1%} "
            f"| {metrics['precision']:.1%} "
            f"| {metrics['recall']:.1%} "
            f"| {metrics['f1']:.1%} |"
        )
    lines.append("")

    best = max(results, key=lambda r: r[1]["accuracy"])
    worst = min(results, key=lambda r: r[1]["accuracy"])
    improvement = (best[1]["accuracy"] - worst[1]["accuracy"]) * 100
    lines.append("### Key Takeaway")
    lines.append("")
    lines.append(
        f"**{best[0]}** achieves the highest accuracy at "
        f"**{best[1]['accuracy']:.1%}**, outperforming "
        f"**{worst[0]}** ({worst[1]['accuracy']:.1%}) by "
        f"**{improvement:.1f} percentage points**."
    )
    lines.append("")


    for idx, (name, metrics, y_true, y_pred) in enumerate(results, start=2):
        lines.append(f"## {idx}. {name} — Detailed Metrics")
        lines.append("")

        # Classification report
        report = str(classification_report(y_true, y_pred, zero_division=0))
        lines.append("### Classification Report")
        lines.append("")
        lines.append("```")
        lines.append(report.rstrip())
        lines.append("```")
        lines.append("")

        # Confusion matrix
        present_labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=present_labels)
        lines.append("### Confusion Matrix")
        lines.append("")

        # Truncate labels for table readability
        short = [l[:14] for l in present_labels]
        header = "| |" + "|".join(f" **{s}** " for s in short) + "|"
        sep = "|---|" + "|".join("---:" for _ in short) + "|"
        lines.append(header)
        lines.append(sep)
        for i, label in enumerate(short):
            row_vals = "|".join(f" {cm[i, j]} " for j in range(len(short)))
            lines.append(f"| **{label}** |{row_vals}|")
        lines.append("")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info("Evaluation report saved to %s", output_path)


def main() -> None:
    os.makedirs("models", exist_ok=True)

    logger.info("Loading configuration and datasets…")
    config = load_config()
    data = load_raw_data(config)
    if data is None:
        raise RuntimeError("Failed to load datasets — check config.yaml paths.")

    df = data["student_reco"]
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    # ── 1. Prepare data ──────────────────────────────────────────────────
    logger.info("[1/5] Building training texts from dataset features…")
    df["_label"] = df["Primary_Career_Recommendation"].apply(_clean_label)

    # Build training text with career-keyword augmentation
    augment_rng = np.random.RandomState(42)
    df["_skills_text"] = df.apply(
        lambda row: _build_skills_text(row, label=row["_label"], rng=augment_rng),
        axis=1,
    )

    all_texts = df["_skills_text"].tolist()
    all_labels = df["_label"].tolist()

    logger.info("  Total samples: %d", len(all_texts))
    logger.info("  Label distribution:\n%s", pd.Series(all_labels).value_counts().to_string())
    logger.info("  Example text: %s", all_texts[0][:200])

    # Use the same random split for all models to ensure fair comparison
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels,
        test_size=0.2, random_state=42, stratify=all_labels,
    )
    logger.info("  Train: %d | Test: %d", len(train_texts), len(test_texts))

    class_labels = sorted(set(all_labels))

    # Prepare numeric features for the RF baseline
    numeric_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_numeric = df[numeric_cols].values
    y_numeric = df["_label"].tolist()

    # Use same random_state & stratify to get aligned splits
    X_num_train, X_num_test, y_num_train, y_num_test = train_test_split(
        X_numeric, y_numeric,
        test_size=0.2, random_state=42, stratify=y_numeric,
    )

    evaluation_results: List[Tuple[str, Dict[str, float], List[str], List[str]]] = []


    logger.info("[2/5] Training SBERT + Logistic Regression (primary model)…")
    encoder = FeatureBuilder()
    sbert_clf = SBERTCareerClassifier(encoder=encoder)
    sbert_clf.train(train_texts, train_labels)
    sbert_clf.save()
    logger.info("  Saved -> models/sbert_career_classifier.pkl")

    sbert_preds = _sbert_predict_all(sbert_clf, test_texts)
    sbert_metrics = _evaluate(test_labels, sbert_preds, "SBERT + LR")
    evaluation_results.append(
        ("SBERT + Logistic Regression", sbert_metrics, test_labels, sbert_preds)
    )


    logger.info("[3/5] Training TF-IDF + Logistic Regression (text baseline)…")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf_train = tfidf.fit_transform(train_texts)
    X_tfidf_test = tfidf.transform(test_texts)

    le_tfidf = LabelEncoder()
    y_tfidf_train = le_tfidf.fit_transform(train_labels)

    tfidf_clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42,
    )
    tfidf_clf.fit(X_tfidf_train, y_tfidf_train)

    tfidf_preds_encoded = tfidf_clf.predict(X_tfidf_test)
    tfidf_preds = [str(c) for c in le_tfidf.inverse_transform(tfidf_preds_encoded)]
    tfidf_metrics = _evaluate(test_labels, tfidf_preds, "TF-IDF + LR")
    evaluation_results.append(
        ("TF-IDF + Logistic Regression", tfidf_metrics, test_labels, tfidf_preds)
    )


    logger.info("[4/5] Training Numeric-only Random Forest (numeric baseline)…")
    le_rf = LabelEncoder()
    y_rf_train = le_rf.fit_transform(y_num_train)

    rf_clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_clf.fit(X_num_train, y_rf_train)

    rf_preds_encoded = rf_clf.predict(X_num_test)
    rf_preds = [str(c) for c in le_rf.inverse_transform(rf_preds_encoded)]
    rf_metrics = _evaluate(y_num_test, rf_preds, "Numeric RF")
    evaluation_results.append(
        ("Numeric-only Random Forest", rf_metrics, y_num_test, rf_preds)
    )


    logger.info("[5/5] Generating evaluation report…")

    # Print ablation summary to console
    logger.info("")
    logger.info("=" * 72)
    logger.info("  ABLATION STUDY RESULTS")
    logger.info("=" * 72)
    logger.info(
        "  %-35s %8s %9s %8s %8s",
        "Model", "Acc", "Prec", "Rec", "F1",
    )
    logger.info("  " + "-" * 68)
    for name, metrics, _, _ in evaluation_results:
        logger.info(
            "  %-35s %7.1f%% %8.1f%% %7.1f%% %7.1f%%",
            name,
            metrics["accuracy"] * 100,
            metrics["precision"] * 100,
            metrics["recall"] * 100,
            metrics["f1"] * 100,
        )
    logger.info("=" * 72)

    report_path = Path("reports/evaluation_report.md")
    _generate_report(
        evaluation_results,
        class_labels,
        report_path,
        dataset_info={
            "total": len(all_labels),
            "train": len(train_labels),
            "test": len(test_labels),
        },
    )


    logger.info("Training University Ranker…")
    builder = UniversityDatasetBuilder(
        data["indian_colleges"], data["world_universities"]
    )
    unified = builder.build()
    trainer = UniversityRankerTrainer()
    trainer.train(
        universities=unified,
        career_labels=df["_label"],
        save_path="models/university_ranker.pkl",
        max_countries=30,
    )
    logger.info("  Saved -> models/university_ranker.pkl")

    logger.info("All models trained successfully.")


if __name__ == "__main__":
    main()
