"""
Auth service layer — separates business logic from HTTP transport (SRP).

api.py route handlers should only call AuthService methods; they must not
contain raw DB queries, password hashing, or token creation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from pymongo.database import Database

from src.auth.auth import create_access_token, hash_password, verify_password

logger = logging.getLogger(__name__)


# ── Data transfer objects ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuthResult:
    """Returned by AuthService on success. Immutable."""
    token: str
    user_id: str
    name: str


# ── Repository (SRP: only responsible for user persistence) ──────────────────

class UserRepository:
    """
    Encapsulates all MongoDB access for the users collection.

    Single Responsibility: knows how to read/write users — nothing else.
    Dependency Inversion: receives `db` via constructor so callers can inject
    any compatible Database (real or test double).
    """

    def __init__(self, db: Database) -> None:
        self._col = db["users"]

    def find_by_email(self, email: str) -> Optional[dict]:
        return self._col.find_one({"email": email.lower()})

    def email_exists(self, email: str) -> bool:
        return self._col.count_documents({"email": email.lower()}, limit=1) > 0

    def create(self, name: str, email: str, password_hash: str) -> str:
        """Insert a new user document. Returns the string _id."""
        result = self._col.insert_one({
            "name": name,
            "email": email.lower(),
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc),
        })
        return str(result.inserted_id)


# ── Service (SRP: orchestrates auth business rules) ───────────────────────────

class AuthService:
    """
    Single Responsibility: auth business rules (register / login).

    Does not know about HTTP, FastAPI, or MongoDB internals —
    those are handled by the caller and UserRepository respectively.
    Open/Closed: extend by subclassing or composing, not by modifying.
    """

    def __init__(self, repo: UserRepository) -> None:
        self._repo = repo

    def register(self, name: str, email: str, password: str) -> AuthResult:
        """
        Create a new user account.
        Raises ValueError on duplicate email.
        """
        if self._repo.email_exists(email):
            raise ValueError("Email already registered")

        hashed = hash_password(password)
        user_id = self._repo.create(name, email, hashed)
        token = create_access_token(user_id, email.lower())
        return AuthResult(token=token, user_id=user_id, name=name)

    def login(self, email: str, password: str) -> AuthResult:
        """
        Verify credentials and return auth tokens.
        Raises ValueError on invalid credentials.
        """
        user = self._repo.find_by_email(email)
        if not user or not verify_password(password, user["password_hash"]):
            raise ValueError("Invalid email or password")

        user_id = str(user["_id"])
        token = create_access_token(user_id, email.lower())
        return AuthResult(token=token, user_id=user_id, name=user.get("name", ""))
