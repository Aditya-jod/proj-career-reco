"""
MongoDB singleton helper.

One MongoClient is created for the whole application lifetime and reused on
every request.  Call close_db() on application shutdown (the FastAPI lifespan
handler does this automatically).

Usage
-----
from src.db.mongo import get_db, close_db

db   = get_db()               # returns the default database
col  = get_db()["users"]      # shorthand for a specific collection
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

from pymongo import MongoClient
from pymongo.database import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton — one client per process, connection pool reused.
# ---------------------------------------------------------------------------
_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    """Return (or lazily create) the module-level MongoClient singleton."""
    global _client
    if _client is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        _client = MongoClient(uri)
        logger.info("MongoDB client created — URI: %s", uri.split("@")[-1])  # hide creds
    return _client


def get_db(name: Optional[str] = None) -> Database:
    """Return the application database handle (reuses the singleton client)."""
    db_name = name or os.getenv("MONGODB_DB_NAME", "career_recommender")
    return _get_client()[db_name]


def close_db() -> None:
    """Close the singleton client.  Call once on application shutdown."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB client closed.")


# ---------------------------------------------------------------------------
# Backwards-compat shim so existing code that imports get_database still works
# ---------------------------------------------------------------------------
def get_database(name: Optional[str] = None) -> Database:
    """Deprecated: use get_db() instead."""
    return get_db(name)


@contextmanager
def mongo_connection(name: Optional[str] = None) -> Generator[Database, None, None]:
    """Context manager: yields a database handle and closes the client on exit.

    Intended for one-off scripts; within the FastAPI app prefer get_db()."""
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
    try:
        yield client[name or os.getenv("MONGODB_DB_NAME", "career_recommender")]
    finally:
        client.close()
