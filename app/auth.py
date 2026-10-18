"""Supabase JWT verification for FastAPI routes and websockets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings

log = logging.getLogger(__name__)
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class User:
    id: Optional[str]
    email: Optional[str] = None
    role: str = "anonymous"

    @property
    def is_anonymous(self) -> bool:
        return self.id is None


ANONYMOUS = User(id=None)


_ASYMMETRIC = ["ES256", "RS256"]
_jwks_clients: dict[str, jwt.PyJWKClient] = {}


def _jwks_client(settings: Settings) -> jwt.PyJWKClient:
    url = settings.supabase_url.rstrip("/") + "/auth/v1/.well-known/jwks.json"
    if url not in _jwks_clients:
        _jwks_clients[url] = jwt.PyJWKClient(url, cache_keys=True, lifespan=3600)
    return _jwks_clients[url]


def decode_token(token: str, settings: Settings) -> User:
    """Validate a Supabase access token (audience 'authenticated').

    New Supabase projects sign tokens with an asymmetric key (ES256/RS256), verified against the
    project's public JWKS. Older projects use the shared HS256 JWT secret.
    """
    try:
        alg = jwt.get_unverified_header(token).get("alg")
    except jwt.InvalidTokenError as exc:
        log.info("Rejected token: %s", exc)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    if alg in _ASYMMETRIC:
        if not settings.supabase_url:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "SUPABASE_URL is not configured")
        key, algorithms = None, _ASYMMETRIC
    else:
        if not settings.supabase_jwt_secret:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "SUPABASE_JWT_SECRET is not configured")
        key, algorithms = settings.supabase_jwt_secret, ["HS256"]
    try:
        if key is None:
            key = _jwks_client(settings).get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            key,
            algorithms=algorithms,
            audience="authenticated",
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.PyJWKClientError as exc:
        log.warning("Could not get a signing key: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Could not verify the token right now")
    except jwt.InvalidTokenError as exc:
        log.info("Rejected token: %s", exc)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    return User(id=claims["sub"], email=claims.get("email"), role=claims.get("role", "authenticated"))


def resolve_user(token: Optional[str], settings: Settings) -> User:
    if token:
        return decode_token(token, settings)
    if settings.auth_required:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return ANONYMOUS


async def current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> User:
    # Use the settings the app was built with (not the global cache) so config is consistent.
    settings: Settings = request.app.state.ctx.settings
    return resolve_user(creds.credentials if creds else None, settings)
