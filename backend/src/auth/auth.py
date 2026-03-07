"""
Authentication helpers: password hashing, JWT creation/verification.

Security: No fallback JWT secret.  If JWT_SECRET_KEY is not set the
application fails loudly at import time rather than running with a
known-insecure key.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

_secret = os.getenv("JWT_SECRET_KEY")
if not _secret:
    raise RuntimeError(
        "JWT_SECRET_KEY environment variable is not set. "
        "Add it to backend/.env or export it before starting the server."
    )

SECRET_KEY: str = _secret
ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_HOURS: int = int(os.getenv("JWT_EXPIRE_HOURS", "48"))

_bearer_scheme = HTTPBearer()



def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())



def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=EXPIRE_HOURS)
    payload = {"sub": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None



async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict:
    """Verify the JWT from the Authorization header and return the payload.

    Raises 401 if the token is missing, expired, or invalid.
    """
    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload
