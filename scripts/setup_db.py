"""
MongoDB Schema Setup Script
============================
Creates collections, validation schemas, and indexes for the
Career Path Recommender database.

Run once (or re-run safely) with:
    backend\\venv\\Scripts\\python.exe scripts/setup_db.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / "backend" / ".env")
except ImportError:
    pass          # dotenv optional here; rely on shell env

from pymongo import ASCENDING, MongoClient
from pymongo.errors import CollectionInvalid

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:44551/")
DB_NAME   = os.getenv("MONGODB_DB_NAME", "career_recommender")



USER_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["name", "email", "password_hash", "created_at"],
        "properties": {
            "name":          {"bsonType": "string"},
            "email":         {"bsonType": "string"},
            "password_hash": {"bsonType": "string"},
            "created_at":    {"bsonType": "date"},
        },
    }
}

RECOMMENDATION_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "profile", "results", "created_at"],
        "properties": {
            "user_id":    {"bsonType": "string"},
            "profile":    {"bsonType": "object"},
            "results":    {"bsonType": "object"},
            "created_at": {"bsonType": "date"},
        },
    }
}


def create_or_update(db, name: str, validator: dict) -> None:
    """Create collection with validator, or update validator if it exists."""
    existing = db.list_collection_names()
    if name not in existing:
        try:
            db.create_collection(name, validator=validator)
            print(f"  ✅ Created   '{name}'")
        except CollectionInvalid:
            print(f"  ⚠️  Already   exists '{name}'")
    else:
        db.command("collMod", name, validator=validator)
        print(f"  🔄 Updated   '{name}'")


def setup() -> None:
    print(f"\n🔗 Connecting to {MONGO_URI}  db={DB_NAME} …")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

    client.admin.command("ping")
    print("   Connected ✓\n")

    db = client[DB_NAME]

    print("📁 Setting up collections …")

    create_or_update(db, "users", USER_SCHEMA)

    # Sessions (no strict schema — flexible JWT info)
    if "sessions" not in db.list_collection_names():
        db.create_collection("sessions")
        print("  ✅ Created   'sessions'")
    else:
        print("  🔄 Exists    'sessions'")

    create_or_update(db, "recommendations", RECOMMENDATION_SCHEMA)

    print("\n🗂️  Creating indexes …")

    db["users"].create_index([("email", ASCENDING)], unique=True, name="idx_users_email")
    print("  ✅ users.email  (unique)")

    db["sessions"].create_index([("user_id", ASCENDING)],  name="idx_sessions_user_id")
    db["sessions"].create_index([("expires_at", ASCENDING)], expireAfterSeconds=0,
                                 name="idx_sessions_ttl")
    print("  ✅ sessions.user_id  (regular)")
    print("  ✅ sessions.expires_at  (TTL)")

    db["recommendations"].create_index(
        [("user_id", ASCENDING), ("created_at", ASCENDING)],
        name="idx_reco_user_created",
    )
    print("  ✅ recommendations.(user_id, created_at)")

    print("\n👤 Ensuring demo admin user …")
    try:
        import importlib.util, pathlib
        auth_path = PROJECT_ROOT / "backend" / "src" / "auth" / "auth.py"
        spec = importlib.util.spec_from_file_location("auth", auth_path)
        auth_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auth_mod)
        hash_password = auth_mod.hash_password
        demo_email = "admin@careerpath.local"
        if not db["users"].find_one({"email": demo_email}):
            db["users"].insert_one({
                "name": "Admin",
                "email": demo_email,
                "password_hash": hash_password("admin1234"),
                "created_at": datetime.now(timezone.utc),
                "role": "admin",
            })
            print(f"  ✅ Created demo user: {demo_email} / admin1234")
        else:
            print(f"  ℹ️  Demo user already exists: {demo_email}")
    except Exception as exc:
        print(f"  ⚠️  Could not create demo user: {exc}")

    print("\n🎉 Database setup complete!\n")
    client.close()


if __name__ == "__main__":
    setup()
