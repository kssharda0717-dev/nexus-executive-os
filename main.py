"""NEXUS — Autonomous Life Operating System.

Main FastAPI application with REST endpoints, WebSocket Mission Control,
MCP Executive Bridge (Outlook + Slack), proactive agent scheduler,
security middleware, and full agent orchestration.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from slowapi.errors import RateLimitExceeded

import database as db
from mission_control import mission_control
from routers import health, chat, data
from security import (
    limiter, rate_limit_exceeded_handler,
    SecurityHeadersMiddleware, ALLOWED_ORIGINS, NEXUS_ENV,
    log_security_event,
)

logger = logging.getLogger("nexus.main")

scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + Bridge + Scheduler."""
    await db.get_db()
    print("NEXUS database initialized")

    # FORCE BRIDGE INITIALIZATION
    try:
        from mcp_executive import mcp_bridge
        status = mcp_bridge.get_status()
        # This line is critical for your logs
        print(f"--- BRIDGE HANDSHAKE ---")
        print(f"Slack: {'ACTIVE ✅' if status['slack']['enabled'] else 'DISABLED ❌'}")
        print(f"Outlook: {'ACTIVE ✅' if status['outlook']['enabled'] else 'DISABLED ❌'}")
        print(f"------------------------")
    except Exception as e:
        print(f"Bridge Handshake Failed: {e}")

    # Start proactive scheduler
    try:
        from agents.proactive import run_daily_briefing, run_overdue_check
        scheduler.add_job(run_daily_briefing, "cron", hour=8, minute=0,
                          id="daily_briefing", replace_existing=True)
        scheduler.add_job(run_overdue_check, "interval", hours=4,
                          id="overdue_check", replace_existing=True)
        scheduler.start()
        print("Proactive scheduler started (daily briefing @ 8am, overdue check every 4h)")
    except Exception as e:
        print(f"Proactive scheduler setup skipped: {e}")

    yield

    scheduler.shutdown(wait=False)
    await db.close_db()
    print("NEXUS shutdown complete")


app = FastAPI(
    title="NEXUS",
    description="Autonomous Life Operating System — Multi-Agent AI Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if NEXUS_ENV != "production" else None,
    redoc_url=None,
)

# ── Security Middleware Stack (order matters: last added = first executed) ──

# 1. Security Headers (X-Content-Type-Options, X-Frame-Options, HSTS, etc.)
app.add_middleware(SecurityHeadersMiddleware)

# 2. CORS — restrictive in production, permissive in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if NEXUS_ENV == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# 3. Rate Limiter (slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# ── Routers ────────────────────────────────────────────
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(data.router)

# Auth router (register, login, logout, session check)
try:
    from routers.auth import router as auth_user_router
    app.include_router(auth_user_router)
    logger.info("Auth router registered")
except Exception as e:
    logger.warning(f"Auth router not loaded: {e}")

# MCP Executive Bridge — modular, zero-crash if credentials missing
try:
    from routers.mcp import router as mcp_router
    app.include_router(mcp_router)
    logger.info("MCP Executive Bridge router registered")
except Exception as e:
    logger.warning(f"MCP Executive Bridge router not loaded: {e}")

# ── Auth Shortcut Routes ─────────────────────────────
# These provide clean /auth/... URLs that the Neural Link UI links to.

@app.get("/auth/microsoft/login", tags=["auth"])
async def auth_microsoft_login(request: Request):
    """Redirect to Microsoft OAuth2 login — uses per-user integration system."""
    try:
        from security import get_current_user
        from mcp_executive import mcp_bridge

        if not mcp_bridge or not mcp_bridge.outlook.enabled:
            return JSONResponse(status_code=501, content={
                "status": "disabled",
                "message": "Outlook not configured. Set MICROSOFT_GRAPH_CLIENT_ID and MICROSOFT_GRAPH_CLIENT_SECRET in .env."
            })

        # Encode user_id in OAuth state for callback binding
        user_id = get_current_user(request)
        import secrets as _secrets
        state = f"{user_id}:{_secrets.token_urlsafe(16)}"
        auth_url = mcp_bridge.outlook.get_auth_url()
        separator = "&" if "?" in auth_url else "?"
        return RedirectResponse(url=f"{auth_url}{separator}state={state}")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/auth/microsoft/callback", tags=["auth"])
async def auth_microsoft_callback(request: Request, code: str = None,
                                  error: str = None, state: str = None):
    """Handle Microsoft OAuth2 callback — saves encrypted tokens per-user."""
    if error:
        return JSONResponse(status_code=400, content={"error": error})
    if not code:
        return JSONResponse(status_code=400, content={"error": "No authorization code received"})
    try:
        from mcp_executive import mcp_bridge
        from security import get_current_user, encrypt_token, log_security_event
        import database as _db

        if not mcp_bridge:
            return JSONResponse(status_code=501, content={"error": "MCP Bridge unavailable"})

        # Resolve user from state param or session
        user_id = None
        if state and ":" in state:
            user_id = state.split(":")[0]
        if not user_id:
            try:
                user_id = get_current_user(request)
            except Exception:
                user_id = "nexus_default_user"

        result = await mcp_bridge.outlook.exchange_code(code)
        if "error" in result:
            return JSONResponse(status_code=400, content=result)

        # Persist encrypted tokens to integrations table
        try:
            await _db.upsert_integration(
                user_id=user_id,
                provider="outlook",
                access_token=encrypt_token(mcp_bridge.config.ms_access_token),
                refresh_token=encrypt_token(mcp_bridge.config.ms_refresh_token),
                scopes="Mail.Read Calendars.ReadWrite User.Read",
                token_expiry=str(mcp_bridge.config.ms_token_expiry),
            )
            log_security_event("INTEGRATION_CONNECT", f"provider=outlook", request, user_id=user_id)
        except Exception as e:
            logger.error(f"Failed to persist Outlook tokens: {e}")

        return RedirectResponse(url="/?auth=microsoft&status=connected")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/auth/slack/install", tags=["auth"])
async def auth_slack_install():
    """Slack connection status and setup instructions."""
    try:
        from mcp_executive import mcp_bridge
        if mcp_bridge and mcp_bridge.slack.enabled:
            return JSONResponse(content={
                "status": "connected",
                "message": "Slack is connected and operational.",
                "channel": mcp_bridge.config.slack_channel,
            })
        return JSONResponse(status_code=501, content={
            "status": "disabled",
            "message": "Slack not configured. Set SLACK_BOT_TOKEN in your .env file.",
            "instructions": [
                "1. Go to https://api.slack.com/apps and create a new app",
                "2. Under OAuth & Permissions, add scopes: chat:write, channels:read, files:write",
                "3. Install to workspace and copy the Bot User OAuth Token",
                "4. Set SLACK_BOT_TOKEN=xoxb-... in your .env file",
                "5. Restart NEXUS",
            ]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/auth/status", tags=["auth"])
async def auth_status(request: Request):
    """Unified auth status endpoint — checks per-user integrations first, then global."""
    try:
        from mcp_executive import mcp_bridge
        from security import get_current_user
        import database as _db

        base = {"outlook": {"connected": False, "enabled": False},
                "slack": {"connected": False, "enabled": False}}

        if not mcp_bridge:
            return base

        status = mcp_bridge.get_status()
        base["outlook"]["enabled"] = status["outlook"].get("enabled", False)
        base["slack"]["enabled"] = status["slack"].get("enabled", False)
        base["slack"]["channel"] = status["slack"].get("channel")

        # Check per-user integrations from DB
        try:
            user_id = get_current_user(request)
            user_integrations = await _db.get_user_integrations(user_id)
            for integ in user_integrations:
                provider = integ.get("provider")
                if provider in base:
                    base[provider]["connected"] = True
                    base[provider]["user_connected"] = True
        except Exception:
            pass

        # Fall back to global token check (single-user/demo mode)
        if not base["outlook"]["connected"]:
            base["outlook"]["connected"] = status["outlook"].get("authenticated", False)
            base["outlook"]["token_valid"] = status["outlook"].get("token_valid", False)
        if not base["slack"]["connected"]:
            base["slack"]["connected"] = status["slack"].get("enabled", False)

        return base
    except Exception:
        return {"outlook": {"connected": False}, "slack": {"connected": False}}


# ── Static Files ──────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
EXPORTS_DIR = Path(__file__).parent / "data" / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Mission Control WebSocket ──────────────────────────

WS_HEARTBEAT_INTERVAL_S = 30  # Server-side ping every 30 seconds


@app.websocket("/ws/mission-control")
async def mission_control_ws(websocket: WebSocket):
    """Real-time stream of all agent thoughts, tool calls, and results.

    Includes server-side heartbeat (ping every 30s) to prevent connection
    drops during long Outlook API calls or complex agent operations.
    """
    await mission_control.connect(websocket)

    async def _heartbeat():
        """Send periodic server-side pings to keep the connection alive."""
        try:
            while True:
                await asyncio.sleep(WS_HEARTBEAT_INTERVAL_S)
                try:
                    await websocket.send_json({"type": "heartbeat", "ts": __import__("time").time()})
                except Exception:
                    break  # Connection closed — let the main loop handle it
        except asyncio.CancelledError:
            pass

    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        while True:
            recv_data = await websocket.receive_text()
            if recv_data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        heartbeat_task.cancel()
        mission_control.disconnect(websocket)


# ── Proactive Trigger Endpoints ────────────────────────

@app.post("/proactive/daily-briefing", tags=["proactive"])
async def trigger_daily_briefing():
    """Manually trigger a daily briefing (normally runs at 8am)."""
    try:
        from agents.proactive import run_daily_briefing
        result = await run_daily_briefing()
        return {"status": "ok", "response": result.get("response", "")[:500]}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/proactive/overdue-check", tags=["proactive"])
async def trigger_overdue_check():
    """Manually trigger an overdue task check (normally runs every 4h)."""
    try:
        from agents.proactive import run_overdue_check
        result = await run_overdue_check()
        return {"status": "ok", "response": result.get("response", "")[:500]}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── Root ───────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the NEXUS Command Center frontend."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/info")
async def api_info():
    """Programmatic API info (original root response)."""
    return {
        "name": "NEXUS",
        "tagline": "Your Autonomous Life Operating System",
        "version": "1.0.0",
        "agents": {
            "primary_agent": "Intent classification + Plan-and-Execute orchestrator",
            "planning_agent": "Week scheduling, calendar optimization, task prioritization",
            "learning_agent": "Curriculum generation, study scheduling, spaced repetition",
            "life_admin_agent": "Life events, checklists, errand orchestration",
            "proactive_monitor": "Background overdue detection, daily briefings, rescheduling",
        },
        "endpoints": {
            "chat": "POST /chat",
            "mission_control": "WS /ws/mission-control",
            "proactive": {
                "daily_briefing": "POST /proactive/daily-briefing",
                "overdue_check": "POST /proactive/overdue-check",
            },
            "health": "GET /health",
            "data": {
                "events": "/data/events",
                "tasks": "/data/tasks",
                "notes": "/data/notes",
            },
        },
    }
