"""Health and status endpoints."""

from fastapi import APIRouter
from database import get_db
from llm_router import llm_router

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "ok", "service": "nexus"}


@router.get("/health/db")
async def db_health():
    """Database connectivity check."""
    try:
        db = await get_db()
        cursor = await db.execute("SELECT COUNT(*) as c FROM tasks")
        row = await cursor.fetchone()
        tasks_count = row["c"]
        cursor = await db.execute("SELECT COUNT(*) as c FROM events")
        row = await cursor.fetchone()
        events_count = row["c"]
        cursor = await db.execute("SELECT COUNT(*) as c FROM notes")
        row = await cursor.fetchone()
        notes_count = row["c"]
        return {
            "status": "ok",
            "database": "connected",
            "counts": {
                "tasks": tasks_count,
                "events": events_count,
                "notes": notes_count
            }
        }
    except Exception as e:
        return {"status": "error", "database": str(e)}


@router.get("/health/llm")
async def llm_health():
    """LLM Router health — provider status, circuit breaker, active assignments."""
    status = llm_router.get_status()
    cb = status.get("circuit_breaker", {})
    all_available = all(p["available"] for p in status["providers"])

    if cb.get("state") == "OPEN":
        health = "circuit_breaker_open"
    elif all_available:
        health = "ok"
    else:
        health = "degraded"

    return {
        "status": health,
        **status,
    }
