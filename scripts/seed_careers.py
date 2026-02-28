"""
Seed the MongoDB ``careers`` collection from real datasets.

**No manual keyword lists or field mappings.**  Job → career classification
and career-path field → career-class mapping are both driven by SBERT
semantic similarity against the career descriptions defined in config.py.

Extracts per-career-field metadata:
  • salary ranges   – aggregated from the job_descriptions dataset
  • top skills      – frequency-ranked from job skills text
  • growth / demand – derived from job-posting volume
  • pathway steps   – built from the career_path dataset
  • descriptions    – imported from config.py (single source of truth)

Run:
    python -m scripts.seed_careers          (from project root)
    python scripts/seed_careers.py          (direct)
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ── Resolve project root so imports work regardless of CWD ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_ROOT = _PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_ROOT))

from src.db.career_repository import delete_all_careers, upsert_career  # noqa: E402
from src.models.config import (                                          # noqa: E402
    CAREER_DESCRIPTIONS,
    CAREER_TITLES,
)

# ── Dataset paths (relative to project root) ────────────────────────────────
_DATASET_ROOT = _PROJECT_ROOT.parent / "Dataset"
_JOB_CSV = _DATASET_ROOT / "job dataset" / "job_descriptions.csv"
_CAREER_PATH_CSV = _DATASET_ROOT / "career path data" / "career_path_in_all_field.csv"
_CAREER_RECO_CSV = (
    _DATASET_ROOT
    / "career recommendation dataset"
    / "career_recommendation_dataset.csv"
)


# ── Lazy-loaded SBERT model ────────────────────────────────────────────────
_sbert_model = None


def _get_sbert():
    """Load the SBERT model once on first use."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        print("   Loading SBERT model (all-MiniLM-L6-v2) …")
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


# ── SBERT-based job classification ──────────────────────────────────────────

def _classify_jobs_sbert(
    titles: pd.Series,
    roles: pd.Series,
    descriptions: Dict[str, str],
    min_similarity: float = 0.15,
) -> pd.Series:
    """
    Classify jobs into career fields using SBERT cosine similarity.

    Instead of manual keyword lists, each job's ``title + role`` text is
    embedded and compared against the 8 career-class description embeddings.
    The best-matching career is assigned if similarity >= *min_similarity*.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    model = _get_sbert()
    career_ids = list(descriptions.keys())
    career_texts = list(descriptions.values())

    # Encode career descriptions (8 vectors)
    career_embeddings = model.encode(career_texts, show_progress_bar=False)

    # Combine title + role for richer job context
    texts = (titles.fillna("") + " " + roles.fillna("")).str.strip().tolist()

    # Encode job texts in batches
    print(f"   Encoding {len(texts):,} job texts with SBERT …")
    job_embeddings = model.encode(texts, batch_size=512, show_progress_bar=True)

    # Cosine similarity: (N_jobs, 8)
    print("   Computing cosine similarities …")
    sims = cosine_similarity(job_embeddings, career_embeddings)

    # Assign best match (or None if below threshold)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    results = pd.Series([None] * len(texts), dtype=object, index=titles.index)
    for i in range(len(texts)):
        if best_sim[i] >= min_similarity:
            results.iloc[i] = career_ids[best_idx[i]]

    # Log distribution
    assigned = results.dropna()
    dist = assigned.value_counts()
    print(f"   SBERT classified {len(assigned):,} / {len(texts):,} jobs:")
    for field, count in dist.items():
        print(f"      {field:<25s}  {count:>6,}")

    return results


# ── SBERT-based career-path field mapping ───────────────────────────────────

def _auto_map_fields(
    dataset_fields: List[str],
    descriptions: Dict[str, str],
) -> Dict[str, str]:
    """
    Automatically map dataset field names (e.g. "Engineering", "Medicine")
    to our career classes (e.g. "STEM", "Healthcare") using SBERT similarity.

    No manual ``_CAREER_PATH_FIELD_MAP`` dictionary needed.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    model = _get_sbert()
    career_ids = list(descriptions.keys())
    career_texts = list(descriptions.values())

    career_embeddings = model.encode(career_texts, show_progress_bar=False)
    field_embeddings = model.encode(dataset_fields, show_progress_bar=False)

    sims = cosine_similarity(field_embeddings, career_embeddings)  # (F, 8)

    mapping: Dict[str, str] = {}
    print("   Auto-mapped career-path fields:")
    for i, field_name in enumerate(dataset_fields):
        best_j = int(sims[i].argmax())
        best_score = sims[i][best_j]
        mapped = career_ids[best_j]
        mapping[field_name] = mapped
        print(f"      {field_name:<20s} → {mapped:<25s} (sim={best_score:.3f})")

    return mapping


# ── Helper functions ────────────────────────────────────────────────────────

def _parse_salary(s: str) -> tuple[int | None, int | None]:
    """Parse strings like '$59K-$99K' → (59000, 99000)."""
    matches = re.findall(r"\$(\d+)K", str(s))
    if len(matches) >= 2:
        return int(matches[0]) * 1000, int(matches[1]) * 1000
    if len(matches) == 1:
        v = int(matches[0]) * 1000
        return v, v
    return None, None


def _extract_skills(skills_series: pd.Series, top_n: int = 15) -> List[str]:
    """
    Frequency-rank skill phrases from a series of skill strings.

    The dataset uses a mix of formats:
      - Comma-separated:  "HTML, CSS, JavaScript"
      - Space-separated phrases starting with caps:
        "Teaching pedagogy Classroom management Curriculum development"
      - Parenthesized tools: "(e.g., React, Angular)"
    """
    counter: Counter = Counter()
    for raw in skills_series.dropna():
        text = str(raw).strip()
        if not text:
            continue

        # Strategy: first try splitting by commas or newlines
        # Then for remaining chunks, split on sentence-case boundaries
        fragments = re.split(r"[,\n;]+", text)
        for frag in fragments:
            frag = frag.strip()
            if not frag:
                continue
            # Split long fragments on uppercase-word boundaries
            # e.g. "Teaching pedagogy Classroom management" → ["Teaching pedagogy", "Classroom management"]
            sub_skills = re.split(r"(?<=[a-z]) (?=[A-Z])", frag)
            for sk in sub_skills:
                sk = sk.strip().rstrip(".")
                sk = re.sub(r"\s*\([^)]*\)?\s*", " ", sk).strip()
                # Also remove trailing "(e.g" fragments
                sk = re.sub(r"\s*\(e\.g\.?$", "", sk).strip()
                if len(sk) < 3 or len(sk) > 50:
                    continue
                if sk.lower() in _GENERIC_WORDS:
                    continue
                # Skip if more than 5 words
                if len(sk.split()) > 5:
                    continue
                counter[sk] += 1

    # Deduplicate case-insensitively
    result = []
    seen_lower: set = set()
    for skill, count in counter.most_common(top_n * 5):
        low = skill.lower().strip()
        if low in seen_lower or low in _GENERIC_WORDS:
            continue
        # Skip single-char or very short
        if len(low) < 3:
            continue
        seen_lower.add(low)
        result.append(skill)
        if len(result) >= top_n:
            break
    return result


_GENERIC_WORDS = {
    "and", "the", "with", "for", "etc", "or", "a", "in", "of", "to",
    "skills", "ability", "knowledge", "experience", "e.g.", "including",
    "such as", "other", "various", "general", "basic", "advanced",
}


def _build_pathway(
    career_path_df: pd.DataFrame,
    field: str,
    field_mapping: Dict[str, str],
) -> List[Dict[str, str]]:
    """
    Build a 4-stage pathway from career_path dataset aggregates.

    Uses the SBERT-derived *field_mapping* to find which dataset fields
    correspond to the given career class.
    """
    mapped_fields = [k for k, v in field_mapping.items() if v == field]
    subset = career_path_df[
        career_path_df["Field"].str.strip().isin(mapped_fields)
    ] if not career_path_df.empty else pd.DataFrame()

    if subset.empty:
        return [
            {"stage": "Foundation", "detail": "Build core academic skills and domain knowledge."},
            {"stage": "Specialization", "detail": "Focus on your chosen area through coursework and projects."},
            {"stage": "Professional Entry", "detail": "Gain practical experience through internships and entry-level roles."},
            {"stage": "Career Growth", "detail": "Advance through certifications, networking, and leadership."},
        ]

    # Compute median/mean stats for the pathway
    avg_gpa = round(subset["GPA"].mean(), 2) if "GPA" in subset.columns else 3.5
    avg_internships = round(subset["Internships"].mean(), 1) if "Internships" in subset.columns else 1
    avg_projects = round(subset["Projects"].mean(), 1) if "Projects" in subset.columns else 2
    top_careers = subset["Career"].str.strip().value_counts().head(5).index.tolist()
    has_certs = (subset.get("Industry_Certifications", pd.Series([0])).mean() > 0.3)

    careers_str = ", ".join(top_careers[:4])
    cert_note = " Industry certifications are common." if has_certs else ""

    return [
        {
            "stage": "Foundation",
            "detail": (
                f"Build a strong academic foundation (avg GPA ~{avg_gpa}). "
                f"Explore areas like {careers_str}."
            ),
        },
        {
            "stage": "Specialization",
            "detail": (
                f"Complete ~{int(round(avg_projects))} projects and "
                f"field-specific coursework to deepen expertise."
            ),
        },
        {
            "stage": "Professional Entry",
            "detail": (
                f"Target ~{int(round(avg_internships))} internship(s) to gain "
                f"industry exposure.{cert_note}"
            ),
        },
        {
            "stage": "Career Growth",
            "detail": (
                "Advance through leadership roles, continuous learning, "
                "networking, and professional development."
            ),
        },
    ]


def _compute_growth(job_count: int, total_jobs: int) -> tuple[str, str]:
    """Return (growth_rate, growth_description) based on job market share."""
    share = job_count / total_jobs if total_jobs else 0
    if share > 0.15:
        return "high", f"High demand — {share:.1%} of job postings"
    if share > 0.08:
        return "moderate-to-high", f"Growing demand — {share:.1%} of job postings"
    if share > 0.03:
        return "moderate", f"Steady demand — {share:.1%} of job postings"
    return "emerging", f"Niche / emerging — {share:.1%} of job postings"


# ── Main seed function ──────────────────────────────────────────────────────

def seed(*, sample_size: int = 200_000, min_similarity: float = 0.15) -> None:
    """
    Extract career metadata from datasets and upsert into MongoDB.

    Parameters
    ----------
    sample_size : int
        Number of rows to sample from the job dataset (1.6 M rows).
    min_similarity : float
        Minimum SBERT cosine similarity for a job to be assigned to a career.
    """
    print("🔄  Loading datasets …")

    # ── Load jobs (sampled for speed) ────────────────────────────────────────
    if _JOB_CSV.exists():
        jobs_df = pd.read_csv(_JOB_CSV, usecols=[
            "Job Title", "Role", "Salary Range", "skills",
        ])
        if len(jobs_df) > sample_size:
            jobs_df = jobs_df.sample(sample_size, random_state=42)
        print(f"   Jobs loaded: {len(jobs_df):,} rows (sampled from full dataset)")
    else:
        print(f"   ⚠  Job dataset not found at {_JOB_CSV}")
        jobs_df = pd.DataFrame(columns=["Job Title", "Role", "Salary Range", "skills"])

    # ── Load career-path data ────────────────────────────────────────────────
    if _CAREER_PATH_CSV.exists():
        career_path_df = pd.read_csv(_CAREER_PATH_CSV)
        career_path_df.columns = [c.strip() for c in career_path_df.columns]
        print(f"   Career-path data loaded: {len(career_path_df):,} rows")
    else:
        print(f"   ⚠  Career-path dataset not found at {_CAREER_PATH_CSV}")
        career_path_df = pd.DataFrame()

    # ── Auto-map career-path fields via SBERT ────────────────────────────────
    if not career_path_df.empty:
        unique_fields = career_path_df["Field"].str.strip().unique().tolist()
        print(f"\n🔄  Auto-mapping {len(unique_fields)} career-path fields via SBERT …")
        field_mapping = _auto_map_fields(unique_fields, CAREER_DESCRIPTIONS)
    else:
        field_mapping = {}

    # ── Classify each job via SBERT similarity ───────────────────────────────
    print("\n🔄  Classifying jobs into career fields via SBERT …")
    if len(jobs_df) > 0:
        jobs_df["career_field"] = _classify_jobs_sbert(
            jobs_df["Job Title"],
            jobs_df["Role"],
            CAREER_DESCRIPTIONS,
            min_similarity=min_similarity,
        )
    else:
        jobs_df["career_field"] = pd.Series(dtype=object)

    classified = jobs_df.dropna(subset=["career_field"])
    total_classified = len(classified)
    print(f"\n   Total classified: {total_classified:,} / {len(jobs_df):,} jobs")

    # ── Aggregate per career field ───────────────────────────────────────────
    print("\n🔄  Computing salary, skills, growth per field …")
    delete_all_careers()

    for field_id in CAREER_DESCRIPTIONS:
        field_jobs = classified[classified["career_field"] == field_id]

        # Salary aggregation
        salaries = field_jobs["Salary Range"].apply(lambda s: _parse_salary(s))
        lo_vals = [s[0] for s in salaries if s[0] is not None]
        hi_vals = [s[1] for s in salaries if s[1] is not None]

        if lo_vals and hi_vals:
            median_lo = int(sorted(lo_vals)[len(lo_vals) // 2])
            median_hi = int(sorted(hi_vals)[len(hi_vals) // 2])
            salary_display = f"${median_lo // 1000}K – ${median_hi // 1000}K"
        else:
            median_lo = None
            median_hi = None
            salary_display = "Data not available"

        # Skills extraction
        skills = _extract_skills(field_jobs["skills"], top_n=15) if len(field_jobs) > 0 else []

        # Growth
        growth_rate, growth_desc = _compute_growth(len(field_jobs), total_classified)

        # Pathway (uses SBERT-derived field_mapping)
        pathway = _build_pathway(career_path_df, field_id, field_mapping)

        career_doc = {
            "career_id": field_id,
            "title": CAREER_TITLES.get(field_id, field_id),
            "description": CAREER_DESCRIPTIONS[field_id],
            "salary_display": salary_display,
            "salary_lo": median_lo,
            "salary_hi": median_hi,
            "growth_rate": growth_rate,
            "growth_description": growth_desc,
            "skills": skills,
            "pathway": pathway,
            "job_count": len(field_jobs),
        }

        upsert_career(career_doc)
        print(
            f"   ✅ {field_id:<25s}  salary={salary_display:<16s}  "
            f"skills={len(skills):<3d}  jobs={len(field_jobs):>6,}"
        )

    print("\n✅  Seed complete — all career metadata stored in MongoDB.")


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    seed()
