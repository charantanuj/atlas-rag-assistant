from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=8, max_length=200)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return base64.urlsafe_b64encode(salt + digest).decode()


def password_matches(password: str, encoded: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(encoded.encode())
        return hmac.compare_digest(raw[16:], hashlib.pbkdf2_hmac("sha256", password.encode(), raw[:16], 310_000))
    except (ValueError, TypeError):
        return False


def issue_token(username: str, secret: str, ttl_seconds: int = 28800) -> str:
    payload = f"{username}|{int(time.time()) + ttl_seconds}"
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}|{signature}".encode()).decode()


def verify_token(token: str, secret: str) -> str | None:
    try:
        payload, expiry, signature = base64.urlsafe_b64decode(token.encode()).decode().rsplit("|", 2)
        if int(expiry) < int(time.time()):
            return None
        expected = hmac.new(secret.encode(), f"{payload}|{expiry}".encode(), hashlib.sha256).hexdigest()
        return payload if hmac.compare_digest(signature, expected) else None
    except (ValueError, TypeError):
        return None


def require_auth(authorization: str | None = Header(default=None), *, username: str, password_hash_value: str, secret: str) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    identity = verify_token(authorization.split(" ", 1)[1], secret)
    if identity != username:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return identity


def login(username: str, password: str, *, expected_username: str, expected_hash: str, secret: str) -> LoginResponse:
    if username != expected_username or not password_matches(password, expected_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return LoginResponse(access_token=issue_token(username, secret))
