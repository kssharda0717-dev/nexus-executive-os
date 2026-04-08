"""Chat endpoint — the main entry point for user interactions with NEXUS.

Implements:
    - Action Snapshot: Last 3 mutations injected into context so the LLM
      can resolve "that", "it", "the task" references.
    - Personalized Patterns: Top correction patterns from the learning loop
      injected into every session for user-specific behavior adaptation.
"""

import json
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from schemas import ChatRequest, ChatResponse
from mission_control import mission_control
from llm_router import AllProvidersExhausted
from security import get_current_user, limiter, log_security_event
import database as db

logger = logging.getLogger("nexus.chat")
router = APIRouter(tags=["chat"])


async def _build_context_prefix() -> str:
    """Build the context prefix with System Time + Action Snapshot + Personalized Patterns.

    This gives the LLM:
    - Temporal grounding (exact current date/time/day-of-week)
    - "Eyes" on its own recent mutations (Action Snapshot)
    - Learned user correction patterns (Personalized Patterns)
    """
    now = datetime.now()
    parts = []

    # ── System Time (Temporal Grounding Layer) ──
    day_of_week = now.strftime("%A")
    formatted_date = now.strftime("%B %d, %Y")
    formatted_time = now.strftime("%H:%M %p")
    parts.append(f"[SYSTEM_TIME: {day_of_week}, {formatted_date} | {formatted_time}]")
    parts.append("")

    # ── Action Snapshot (Short-Term Memory) ──
    try:
        snapshot = await db.get_action_snapshot(limit=3)
        if snapshot:
            parts.append("[ACTION SNAPSHOT — Last 3 mutations:]")
            for i, entry in enumerate(snapshot, 1):
                fields = []
                if entry.get("tool"):
                    fields.append(f"Tool: {entry['tool']}")
                if entry.get("id"):
                    fields.append(f"ID: {entry['id']}")
                if entry.get("title"):
                    fields.append(f"Title: {entry['title']}")
                if entry.get("due_date"):
                    fields.append(f"Date: {entry['due_date']}")
                if entry.get("start_time"):
                    fields.append(f"Start: {entry['start_time']}")
                if entry.get("status"):
                    fields.append(f"Status: {entry['status']}")
                parts.append(f"  {i}. {' | '.join(fields)}")
            parts.append("")
    except Exception as e:
        logger.debug(f"Action snapshot unavailable: {e}")

    # ── Personalized Patterns (Learning Loop) ──
    try:
        corrections = await db.get_top_corrections(limit=5)
        if corrections:
            parts.append("[PERSONALIZED PATTERNS — Learned from your corrections:]")
            for c in corrections:
                ctype = c.get("correction_type", "general")
                freq = c.get("frequency", 0)
                examples = (c.get("examples") or "")[:200]
                if ctype == "date":
                    parts.append(f"  - This user frequently corrects dates ({freq}x). Prioritize 'Update' over 'Add' for date-related follow-ups.")
                elif ctype == "title":
                    parts.append(f"  - This user frequently corrects titles ({freq}x). Confirm title before creating.")
                elif ctype == "priority":
                    parts.append(f"  - This user frequently adjusts priorities ({freq}x). Ask for priority when ambiguous.")
                else:
                    parts.append(f"  - Correction pattern ({ctype}, {freq}x): {examples[:100]}")
            parts.append("")
    except Exception as e:
        logger.debug(f"Correction patterns unavailable: {e}")

    return "\n".join(parts) if parts else ""


# ── Correction Keywords that signal the user is refining a previous action ──
_CORRECTION_KEYWORDS = {"no", "actually", "wait", "change", "fix", "wrong", "not", "instead", "correct"}


async def _detect_and_log_correction(user_message: str, tools_used: list[str],
                                      run_id: str):
    """Detect if this interaction was a correction and log it for the learning loop.

    Heuristic: If the user's message starts with a correction keyword AND
    the agent used an update/complete tool (not a create tool), this is
    a correction event worth learning from.
    """
    msg_lower = user_message.strip().lower()
    first_word = msg_lower.split()[0] if msg_lower.split() else ""

    # Check if message starts with a correction signal
    is_correction = first_word in _CORRECTION_KEYWORDS

    if not is_correction:
        return

    # Check if the agent used an UPDATE tool (not a CREATE)
    update_tools = [t for t in tools_used if "update" in t or "complete" in t]
    create_tools = [t for t in tools_used if "create" in t]

    if not update_tools or create_tools:
        return  # Not a correction — either created new or did nothing

    # Classify the correction type
    correction_type = "general"
    if any(kw in msg_lower for kw in ("date", "time", "when", "schedule", "day",
                                       "morning", "afternoon", "evening", "tomorrow",
                                       "monday", "tuesday", "wednesday", "thursday",
                                       "friday", "saturday", "sunday",
                                       "january", "february", "march", "april",
                                       "may", "june", "july", "august",
                                       "september", "october", "november", "december")):
        correction_type = "date"
    elif any(kw in msg_lower for kw in ("title", "name", "call it", "rename")):
        correction_type = "title"
    elif any(kw in msg_lower for kw in ("priority", "urgent", "high", "low")):
        correction_type = "priority"
    elif any(kw in msg_lower for kw in ("status", "done", "complete", "pending")):
        correction_type = "status"

    await db.log_correction(
        original_prompt=user_message[:500],
        detected_error=f"User corrected via {', '.join(update_tools)}",
        final_correction=f"Applied {', '.join(update_tools)} instead of create",
        correction_type=correction_type,
    )
    logger.info(f"Learning Loop: logged {correction_type} correction from run {run_id}")


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """Process a user message through the NEXUS agent system.

    Pipeline:
    1. Authenticates the user (JWT/cookie)
    2. Validates input length
    3. Builds context prefix (Action Snapshot + Personalized Patterns)
    4. Creates an agent run record
    5. Routes to the Primary Agent with enriched context
    6. Post-run: detects corrections and logs them for the learning loop
    7. Returns the final synthesized response
    """
    user_id = get_current_user(request)

    # Input length guard — prevent AI token drain attacks
    if len(chat_request.message) > 10_000:
        raise HTTPException(400, "Message too long (max 10,000 characters)")

    run = await db.create_agent_run(user_message=chat_request.message)
    run_id = run["id"]

    try:
        await mission_control.emit_narration(
            run_id, "primary_agent",
            f"Got it! Working on: \"{chat_request.message[:100]}\"..."
        )

        # ── Build enriched message with Action Snapshot + Patterns ──
        context_prefix = await _build_context_prefix()
        enriched_message = chat_request.message
        if context_prefix:
            enriched_message = f"{context_prefix}\n{chat_request.message}"

        from agents.primary import process_message
        result = await process_message(run_id, enriched_message)

        # Finalize the run
        await db.update_agent_run(
            run_id,
            status="completed",
            result=result.get("response", "")[:5000],
            completed_at=datetime.now().isoformat()
        )

        # Count steps
        db_conn = await db.get_db()
        cursor = await db_conn.execute(
            "SELECT COUNT(*) as c FROM agent_steps WHERE run_id = ?", (run_id,)
        )
        row = await cursor.fetchone()
        steps_count = row["c"]

        logger.info(f"Run {run_id} completed: {steps_count} steps, "
                     f"{len(result.get('tools_used', []))} tools used")

        # ── Learning Loop: Detect and log corrections ──
        try:
            await _detect_and_log_correction(
                chat_request.message,
                result.get("tools_used", []),
                run_id,
            )
        except Exception as e:
            logger.debug(f"Correction detection skipped: {e}")

        return ChatResponse(
            run_id=run_id,
            response=result.get("response", ""),
            steps_count=steps_count,
            tools_used=result.get("tools_used", [])
        )

    except AllProvidersExhausted as e:
        # ── 503: Structured degradation — NEVER expose raw traceback ──
        logger.warning(f"Run {run_id} — all LLM providers exhausted: {e}")
        await mission_control.emit_narration(
            run_id, "primary_agent",
            "I'm experiencing very high demand right now. "
            "Please try again in a moment — I'll be back to full strength shortly."
        )
        await db.update_agent_run(run_id, status="congested",
                                  result="All LLM providers temporarily exhausted")
        return JSONResponse(
            status_code=503,
            content={
                "run_id": run_id,
                "response": (
                    "NEXUS is experiencing high demand and all AI models are temporarily "
                    "at capacity. Your request has been noted. Please retry in 2-3 minutes."
                ),
                "steps_count": 0,
                "tools_used": [],
                "status": "congested",
                "retry_after_seconds": 180,
            },
            headers={"Retry-After": "180"},
        )

    except Exception as e:
        logger.exception(f"Run {run_id} failed: {e}")
        await mission_control.emit_error(run_id, "primary_agent", str(e))
        await db.update_agent_run(run_id, status="failed", result=str(e)[:2000])
        # Never expose raw exception details to the client
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. The error has been logged for investigation."
        )


class ResumeRequest(BaseModel):
    run_id: str


@router.post("/chat/resume", response_model=ChatResponse)
@limiter.limit("10/minute")
async def resume_chat(request: Request, resume_req: ResumeRequest):
    """Resume an interrupted agent plan from its last checkpoint.

    If a previous /chat request was interrupted (server crash, timeout, etc.),
    call this endpoint with the original run_id to pick up where it left off.
    """
    get_current_user(request)
    run_id = resume_req.run_id

    try:
        await mission_control.emit_narration(
            run_id, "primary_agent",
            "Resuming interrupted operation..."
        )

        from agents.primary import resume_plan
        result = await resume_plan(run_id)

        if result.get("response") == "No interrupted plan found to resume.":
            return ChatResponse(
                run_id=run_id,
                response=result["response"],
                steps_count=0,
                tools_used=[]
            )

        await db.update_agent_run(
            run_id,
            status="completed",
            result=result.get("response", "")[:5000],
            completed_at=datetime.now().isoformat()
        )

        db_conn = await db.get_db()
        cursor = await db_conn.execute(
            "SELECT COUNT(*) as c FROM agent_steps WHERE run_id = ?", (run_id,)
        )
        row = await cursor.fetchone()
        steps_count = row["c"]

        return ChatResponse(
            run_id=run_id,
            response=result.get("response", ""),
            steps_count=steps_count,
            tools_used=result.get("tools_used", [])
        )

    except AllProvidersExhausted as e:
        logger.warning(f"Resume {run_id} — all LLM providers exhausted: {e}")
        await mission_control.emit_narration(
            run_id, "primary_agent",
            "High demand during plan resume. Switching to resilient mode."
        )
        return JSONResponse(
            status_code=503,
            content={
                "run_id": run_id,
                "response": (
                    "NEXUS is experiencing high demand. Your plan resume has been "
                    "paused. Please retry in 2-3 minutes."
                ),
                "steps_count": 0,
                "tools_used": [],
                "status": "congested",
                "retry_after_seconds": 180,
            },
            headers={"Retry-After": "180"},
        )

    except Exception as e:
        logger.exception(f"Resume {run_id} failed: {e}")
        await mission_control.emit_error(run_id, "primary_agent", str(e))
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during plan resume."
        )
