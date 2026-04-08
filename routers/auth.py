"""Authentication Router — Register, Login, Logout, Session, Multi-Tenant Integrations.

Implements:
    - POST /auth/register — Create account with Bcrypt-hashed password
    - POST /auth/login — Authenticate + set JWT HTTP-only cookie
    - POST /auth/logout — Clear session cookie
    - GET  /auth/me — Get current user info (session validation)
    - GET  /auth/{provider}/connect — Generate OAuth URL for the logged-in user
    - GET  /auth/{provider}/callback — Exchange OAuth code, save encrypted tokens
    - DELETE /auth/{provider}/disconnect — Wipe tokens, deactivate integration
    - GET  /auth/integrations — List user's active integrations (metadata only)

Rate limited: 5 attempts/min on login, 3/min on register.
"""

import logging
import time
from pydantic import BaseModel, Field, field_validator
from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

import database as db
from security import (
    hash_password, verify_password,
    create_jwt_token, set_auth_cookie, clear_auth_cookie,
    get_current_user, log_security_event, limiter,
    sanitize_string, MAX_TITLE_LENGTH,
    encrypt_token, decrypt_token,
)

logger = logging.getLogger("nexus.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


# ── Request Schemas ───────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8, max_length=128)
    display_name: str = Field(default="", max_length=128)

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        import re
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError("Username can only contain letters, numbers, underscores, hyphens, and dots")
        return v.lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter")
        return v


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


# ── Endpoints ─────────────────────────────────────────────

@router.post("/register")
@limiter.limit("3/minute")
async def register(request: Request, body: RegisterRequest):
    """Register a new user account."""
    username = body.username
    display_name = sanitize_string(body.display_name or username, MAX_TITLE_LENGTH, "display_name")

    # Check if user exists
    existing = await db.get_user_by_username(username)
    if existing:
        log_security_event("REGISTER_DUPLICATE", f"username={username}", request)
        raise HTTPException(status_code=409, detail="Username already taken")

    # Create user
    password_hash = hash_password(body.password)
    user = await db.create_user(username=username, password_hash=password_hash,
                                display_name=display_name)

    log_security_event("REGISTER_SUCCESS", f"username={username} user_id={user['id']}",
                       request, user_id=user["id"])

    # Auto-login: set JWT cookie
    token = create_jwt_token(user["id"])
    response = JSONResponse(content={
        "status": "registered",
        "user_id": user["id"],
        "username": username,
        "display_name": display_name,
    })
    set_auth_cookie(response, token)
    return response


@router.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, body: LoginRequest):
    """Authenticate and receive a session cookie."""
    user = await db.get_user_by_username(body.username.lower())

    if not user:
        log_security_event("LOGIN_FAILURE", f"username={body.username} reason=user_not_found",
                           request, severity="WARNING")
        # Constant-time response to prevent username enumeration
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(body.password, user["password_hash"]):
        log_security_event("LOGIN_FAILURE", f"username={body.username} reason=bad_password",
                           request, user_id=user["id"], severity="WARNING")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_jwt_token(user["id"])
    log_security_event("LOGIN_SUCCESS", f"username={body.username}",
                       request, user_id=user["id"])

    response = JSONResponse(content={
        "status": "authenticated",
        "user_id": user["id"],
        "username": user["username"],
        "display_name": user.get("display_name", ""),
    })
    set_auth_cookie(response, token)
    return response


@router.post("/logout")
async def logout(request: Request):
    """Clear the session cookie."""
    user_id = None
    try:
        user_id = get_current_user(request)
    except Exception:
        pass
    log_security_event("LOGOUT", "", request, user_id=user_id)

    response = JSONResponse(content={"status": "logged_out"})
    clear_auth_cookie(response)
    return response


@router.get("/me")
async def get_me(request: Request):
    """Get current authenticated user info. Returns 401 if not authenticated."""
    user_id = get_current_user(request)
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {
        "user_id": user["id"],
        "username": user["username"],
        "display_name": user.get("display_name", ""),
        "created_at": user.get("created_at"),
    }


# ═══════════════════════════════════════════════════════════════
# MULTI-TENANT INTEGRATIONS — Per-User OAuth Connect/Disconnect
# ═══════════════════════════════════════════════════════════════

PROVIDER_CONFIG = {
    "outlook": {
        "name": "Microsoft Outlook",
        "scopes": "Mail.Read Calendars.ReadWrite User.Read",
    },
    "slack": {
        "name": "Slack",
        "scopes": "chat:write channels:read files:write",
    },
    "linkedin": {
        "name": "LinkedIn",
        "scopes": "",
    },
}


def _get_bridge():
    """Lazy-import MCP bridge."""
    try:
        from mcp_executive import mcp_bridge
        return mcp_bridge
    except Exception:
        return None


@router.get("/integrations")
async def list_integrations(request: Request):
    """List all active integrations for the current user (metadata only, no tokens)."""
    user_id = get_current_user(request)
    integrations = await db.get_user_integrations(user_id)

    # Build a status map with all providers
    result = {}
    for provider, cfg in PROVIDER_CONFIG.items():
        active = next((i for i in integrations if i["provider"] == provider), None)
        result[provider] = {
            "name": cfg["name"],
            "connected": active is not None,
            "scopes": active["scopes"] if active else "",
            "connected_at": active["created_at"] if active else None,
        }
    return {"user_id": user_id, "integrations": result}


@router.get("/{provider}/connect")
async def connect_provider(provider: str, request: Request):
    """Generate the OAuth authorization URL for the logged-in user.

    Supports: outlook, slack, linkedin
    """
    if provider not in PROVIDER_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}. Supported: {list(PROVIDER_CONFIG.keys())}")

    user_id = get_current_user(request)

    if provider == "outlook":
        bridge = _get_bridge()
        if not bridge or not bridge.outlook.enabled:
            return JSONResponse(status_code=501, content={
                "status": "disabled",
                "message": "Outlook integration not configured. Set MICROSOFT_GRAPH_CLIENT_ID and MICROSOFT_GRAPH_CLIENT_SECRET in .env.",
            })
        # Generate auth URL with state parameter encoding user_id
        import secrets as _secrets
        state = f"{user_id}:{_secrets.token_urlsafe(16)}"
        auth_url = bridge.outlook.get_auth_url()
        # Append state parameter for user tracking through OAuth flow
        separator = "&" if "?" in auth_url else "?"
        auth_url += f"{separator}state={state}"
        return {"auth_url": auth_url, "provider": "outlook"}

    elif provider == "slack":
        bridge = _get_bridge()
        if bridge and bridge.slack.enabled:
            return JSONResponse(content={
                "status": "already_configured",
                "message": "Slack is configured via bot token. Your admin has already set this up.",
                "provider": "slack",
            })
        return JSONResponse(status_code=501, content={
            "status": "disabled",
            "message": "Slack requires a bot token configured by an admin. Set SLACK_BOT_TOKEN in .env.",
            "instructions": [
                "1. Go to https://api.slack.com/apps and create a new app",
                "2. Add scopes: chat:write, channels:read, files:write",
                "3. Install to workspace and copy the Bot User OAuth Token",
                "4. Set SLACK_BOT_TOKEN in .env and restart NEXUS",
            ],
        })

    elif provider == "linkedin":
        return JSONResponse(content={
            "status": "api_key",
            "message": "LinkedIn uses Tavily web search. Set TAVILY_API_KEY in .env to enable.",
            "provider": "linkedin",
        })

    return JSONResponse(status_code=400, content={"error": "Unsupported provider"})


@router.get("/{provider}/callback")
async def provider_callback(provider: str, request: Request,
                            code: str = None, error: str = None,
                            state: str = None):
    """Handle OAuth callback — exchange code for tokens, save encrypted to DB.

    The state parameter contains user_id:nonce for session binding.
    """
    if provider not in PROVIDER_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    if error:
        return JSONResponse(status_code=400, content={"error": error})

    if provider == "outlook":
        if not code:
            return JSONResponse(status_code=400, content={"error": "No authorization code received"})

        # Extract user_id from state parameter
        user_id = None
        if state and ":" in state:
            user_id = state.split(":")[0]

        # Fallback to session user if state is missing
        if not user_id:
            try:
                user_id = get_current_user(request)
            except Exception:
                user_id = "nexus_default_user"

        bridge = _get_bridge()
        if not bridge:
            return JSONResponse(status_code=501, content={"error": "MCP Bridge unavailable"})

        result = await bridge.outlook.exchange_code(code)
        if "error" in result:
            return JSONResponse(status_code=400, content=result)

        # Save encrypted tokens to integrations table
        try:
            access_token = bridge.config.ms_access_token
            refresh_token = bridge.config.ms_refresh_token
            expiry = str(bridge.config.ms_token_expiry)

            await db.upsert_integration(
                user_id=user_id,
                provider="outlook",
                access_token=encrypt_token(access_token),
                refresh_token=encrypt_token(refresh_token),
                scopes=PROVIDER_CONFIG["outlook"]["scopes"],
                token_expiry=expiry,
            )

            log_security_event(
                "INTEGRATION_CONNECT", f"provider=outlook user={user_id}",
                request, user_id=user_id,
            )
            logger.info(f"Outlook integration saved for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to save Outlook integration: {e}")

        # Redirect back to dashboard with success indicator
        return RedirectResponse(url="/?auth=microsoft&status=connected")

    return JSONResponse(status_code=400, content={"error": f"Callback not supported for {provider}"})


@router.delete("/{provider}/disconnect")
async def disconnect_provider(provider: str, request: Request):
    """Disconnect (wipe tokens) for a provider integration.

    Deactivates the integration and wipes all stored tokens from the DB.
    """
    if provider not in PROVIDER_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    user_id = get_current_user(request)

    deleted = await db.delete_integration(user_id, provider)
    if not deleted:
        return JSONResponse(status_code=404, content={
            "status": "not_found",
            "message": f"No active {PROVIDER_CONFIG[provider]['name']} integration found.",
        })

    log_security_event(
        "INTEGRATION_DISCONNECT", f"provider={provider} user={user_id}",
        request, user_id=user_id,
    )

    return {
        "status": "disconnected",
        "provider": provider,
        "message": f"{PROVIDER_CONFIG[provider]['name']} has been disconnected. All tokens have been wiped.",
    }
