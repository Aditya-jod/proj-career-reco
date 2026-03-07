"""
MongoDB repository for career metadata.

All career-field information (descriptions, salary ranges, skills, growth,
pathway steps) lives in the ``careers`` collection and is served dynamically
by the API — no hardcoded display data anywhere.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from src.db.mongo import get_db

_COLLECTION = "careers"



def _col():
    """Return the ``careers`` collection handle."""
    return get_db()[_COLLECTION]


def _ensure_indexes() -> None:
    """Create indexes once (idempotent)."""
    col = _col()
    col.create_index("career_id", unique=True)
    col.create_index("title")


def upsert_career(career: Dict[str, Any]) -> None:
    """Insert or replace a career document keyed on ``career_id``."""
    career.setdefault("updated_at", _dt.datetime.utcnow().isoformat())
    _col().update_one(
        {"career_id": career["career_id"]},
        {"$set": career},
        upsert=True,
    )


def get_all_careers() -> List[Dict[str, Any]]:
    """Return every career document (without Mongo ``_id``)."""
    return list(_col().find({}, {"_id": 0}))


def get_career(career_id: str) -> Optional[Dict[str, Any]]:
    """Return a single career or ``None``."""
    return _col().find_one({"career_id": career_id}, {"_id": 0})


def get_career_descriptions() -> Dict[str, str]:
    """Return ``{career_id: description}`` mapping for the SBERT classifier."""
    return {
        doc["career_id"]: doc.get("description", "")
        for doc in _col().find({}, {"_id": 0, "career_id": 1, "description": 1})
    }


def get_career_metadata(career_id: str) -> Optional[Dict[str, Any]]:
    """Return salary, growth, skills, pathway for a specific career."""
    doc = _col().find_one(
        {"career_id": career_id},
        {
            "_id": 0,
            "career_id": 1,
            "title": 1,
            "salary_display": 1,
            "growth_description": 1,
            "growth_rate": 1,
            "skills": 1,
            "pathway": 1,
        },
    )
    return doc


def delete_all_careers() -> int:
    """Wipe the collection — used by the seed script before re-seeding."""
    result = _col().delete_many({})
    return result.deleted_count


# Run index creation on import
_ensure_indexes()
