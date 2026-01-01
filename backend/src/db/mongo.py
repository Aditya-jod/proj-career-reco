from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator, Optional

from pymongo import MongoClient
from pymongo.database import Database


def _get_mongo_uri() -> str:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        uri = "mongodb://localhost:27017/"
    return uri


def get_mongo_client() -> MongoClient:
    uri = _get_mongo_uri()
    return MongoClient(uri)


def get_database(name: Optional[str] = None) -> Database:
    """Return a Database handle using the provided name or env default."""
    db_name = name or os.getenv("MONGODB_DB_NAME", "career_recommender")
    client = get_mongo_client()
    return client[db_name]


@contextmanager
def mongo_connection(name: Optional[str] = None) -> Generator[Database, None, None]:
    """Context manager that yields a database connection and closes it afterwards."""
    client = get_mongo_client()
    try:
        yield client[name or os.getenv("MONGODB_DB_NAME", "career_recommender")]
    finally:
        client.close()
