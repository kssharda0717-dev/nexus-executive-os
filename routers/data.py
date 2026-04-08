"""Data endpoints — direct CRUD access to calendar, tasks, and notes.

These endpoints allow the frontend to display and manage data directly,
independent of the agent system.

Security:
    - All endpoints verify ownership (session_user == resource_owner)
    - Input lengths validated
    - Rate limited
"""

import asyncio
import re

from fastapi import APIRouter, HTTPException, Request
from schemas import (
    EventCreate, EventUpdate, Event,
    TaskCreate, TaskUpdate, Task,
    NoteCreate, NoteUpdate, Note,
)
import database as db
from mission_control import mission_control
from security import (
    get_current_user, limiter, log_security_event,
    sanitize_string, sanitize_id,
    MAX_TITLE_LENGTH, MAX_STRING_LENGTH, MAX_SEARCH_QUERY_LENGTH,
)

router = APIRouter(prefix="/data", tags=["data"])


def _check_ownership(resource: dict, user_id: str, resource_type: str = "Resource"):
    """Verify the authenticated user owns the resource. Raises 404 (not 403) to prevent IDOR enumeration."""
    if not resource:
        raise HTTPException(404, f"{resource_type} not found")
    resource_owner = resource.get("owner_id", "nexus_default_user")
    if resource_owner != user_id and resource_owner != "nexus_default_user":
        # Return 404 (not 403) so attackers can't enumerate valid IDs
        raise HTTPException(404, f"{resource_type} not found")


# ── Calendar ───────────────────────────────────────────

@router.get("/events", response_model=list[dict])
async def list_events(request: Request, start: str = None, end: str = None):
    user_id = get_current_user(request)
    return await db.list_events(start, end, owner_id=user_id)


@router.post("/events", response_model=dict)
@limiter.limit("30/minute")
async def create_event(request: Request, event: EventCreate):
    user_id = get_current_user(request)
    return await db.create_event(**event.model_dump(), owner_id=user_id)


@router.get("/events/{event_id}", response_model=dict)
async def get_event(request: Request, event_id: str):
    user_id = get_current_user(request)
    event_id = sanitize_id(event_id)
    result = await db.get_event(event_id)
    _check_ownership(result, user_id, "Event")
    return result


@router.patch("/events/{event_id}", response_model=dict)
async def update_event(request: Request, event_id: str, event: EventUpdate):
    user_id = get_current_user(request)
    event_id = sanitize_id(event_id)
    existing = await db.get_event(event_id)
    _check_ownership(existing, user_id, "Event")
    fields = {k: v for k, v in event.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "No fields to update")
    result = await db.update_event(event_id, **fields)
    if not result:
        raise HTTPException(404, "Event not found")
    return result


@router.delete("/events/{event_id}")
async def delete_event(request: Request, event_id: str):
    user_id = get_current_user(request)
    event_id = sanitize_id(event_id)
    existing = await db.get_event(event_id)
    _check_ownership(existing, user_id, "Event")
    success = await db.delete_event(event_id)
    if not success:
        raise HTTPException(404, "Event not found")
    return {"deleted": True}


@router.get("/events/free-slots/find")
async def find_free_slots(request: Request, start: str, end: str, duration_minutes: int = 60):
    get_current_user(request)
    return await db.find_free_slots(start, end, duration_minutes)


# ── Tasks ──────────────────────────────────────────────

@router.get("/tasks", response_model=list[dict])
async def list_tasks(request: Request, status: str = None, priority: str = None, due_before: str = None):
    user_id = get_current_user(request)
    return await db.list_tasks(status=status, priority=priority, due_before=due_before,
                               owner_id=user_id)


@router.post("/tasks", response_model=dict)
@limiter.limit("30/minute")
async def create_task(request: Request, task: TaskCreate):
    user_id = get_current_user(request)
    return await db.create_task(**task.model_dump(), owner_id=user_id)


@router.get("/tasks/{task_id}", response_model=dict)
async def get_task(request: Request, task_id: str):
    user_id = get_current_user(request)
    task_id = sanitize_id(task_id)
    result = await db.get_task(task_id)
    _check_ownership(result, user_id, "Task")
    return result


@router.patch("/tasks/{task_id}", response_model=dict)
async def update_task(request: Request, task_id: str, task: TaskUpdate):
    user_id = get_current_user(request)
    task_id = sanitize_id(task_id)
    existing = await db.get_task(task_id)
    _check_ownership(existing, user_id, "Task")
    fields = {k: v for k, v in task.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "No fields to update")
    result = await db.update_task(task_id, **fields)
    if not result:
        raise HTTPException(404, "Task not found")
    return result


@router.post("/tasks/{task_id}/complete", response_model=dict)
async def complete_task(request: Request, task_id: str):
    user_id = get_current_user(request)
    task_id = sanitize_id(task_id)
    existing = await db.get_task(task_id)
    _check_ownership(existing, user_id, "Task")
    result = await db.complete_task(task_id)
    if not result:
        raise HTTPException(404, "Task not found")
    asyncio.create_task(mission_control.emit_narration(
        "ui-action", "primary_agent",
        f"Task '{result.get('title', '')}' successfully archived."
    ))
    return result


@router.delete("/tasks/{task_id}")
async def delete_task(request: Request, task_id: str):
    """Delete a task permanently."""
    user_id = get_current_user(request)
    task_id = sanitize_id(task_id)
    task = await db.get_task(task_id)
    _check_ownership(task, user_id, "Task")
    task_title = task.get("title", "") if task else ""
    success = await db.delete_task(task_id)
    if not success:
        raise HTTPException(404, "Task not found")
    asyncio.create_task(mission_control.emit_narration(
        "ui-action", "primary_agent",
        f"Task '{task_title}' removed from your list."
    ))
    return {"deleted": True}


@router.get("/tasks/overdue/all", response_model=list[dict])
async def get_overdue_tasks(request: Request):
    user_id = get_current_user(request)
    return await db.get_overdue_tasks()


# ── Notes ──────────────────────────────────────────────

@router.get("/notes", response_model=list[dict])
async def list_notes(request: Request, tags: str = None):
    user_id = get_current_user(request)
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    return await db.list_notes(tags=tag_list, owner_id=user_id)


@router.post("/notes", response_model=dict)
@limiter.limit("30/minute")
async def create_note(request: Request, note: NoteCreate):
    user_id = get_current_user(request)
    return await db.create_note(**note.model_dump(), owner_id=user_id)


@router.get("/notes/{note_id}", response_model=dict)
async def get_note(request: Request, note_id: str):
    user_id = get_current_user(request)
    note_id = sanitize_id(note_id)
    result = await db.get_note(note_id)
    _check_ownership(result, user_id, "Note")
    return result


@router.patch("/notes/{note_id}", response_model=dict)
async def update_note(request: Request, note_id: str, note: NoteUpdate):
    user_id = get_current_user(request)
    note_id = sanitize_id(note_id)
    existing = await db.get_note(note_id)
    _check_ownership(existing, user_id, "Note")
    fields = {k: v for k, v in note.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "No fields to update")
    result = await db.update_note(note_id, **fields)
    if not result:
        raise HTTPException(404, "Note not found")
    return result


@router.get("/notes/search/query")
async def search_notes(request: Request, q: str):
    user_id = get_current_user(request)
    # Input length limit to prevent DoS via huge search strings
    if len(q) > MAX_SEARCH_QUERY_LENGTH:
        raise HTTPException(400, f"Search query too long (max {MAX_SEARCH_QUERY_LENGTH} chars)")
    return await db.search_notes(q, owner_id=user_id)


@router.post("/notes/{note_id}/append", response_model=dict)
async def append_to_note(request: Request, note_id: str, content: dict):
    user_id = get_current_user(request)
    note_id = sanitize_id(note_id)
    existing = await db.get_note(note_id)
    _check_ownership(existing, user_id, "Note")
    append_content = content.get("content", "")
    if len(append_content) > MAX_STRING_LENGTH:
        raise HTTPException(400, f"Content too long (max {MAX_STRING_LENGTH} chars)")
    result = await db.append_to_note(note_id, append_content)
    if not result:
        raise HTTPException(404, "Note not found")
    return result


# ── Email Inbox (Mental Peace) ────────────────────────

@router.get("/emails", response_model=list[dict])
async def list_emails(request: Request, category: str = None, unread_only: bool = False):
    get_current_user(request)
    return await db.list_emails(category=category, unread_only=unread_only)


@router.get("/emails/counts")
async def email_counts(request: Request):
    get_current_user(request)
    return await db.get_email_counts()


@router.post("/emails/{email_id}/read")
async def mark_email_read(request: Request, email_id: str):
    get_current_user(request)
    email_id = sanitize_id(email_id)
    success = await db.mark_email_read(email_id)
    if not success:
        raise HTTPException(404, "Email not found")
    return {"read": True}


@router.patch("/tasks/{task_id}/urgency")
async def update_task_urgency(request: Request, task_id: str, body: dict):
    """Update a task's urgency DNA (async/standard/critical)."""
    user_id = get_current_user(request)
    task_id = sanitize_id(task_id)
    existing = await db.get_task(task_id)
    _check_ownership(existing, user_id, "Task")
    urgency = body.get("urgency", "standard")
    if urgency not in ("async", "standard", "critical"):
        raise HTTPException(400, "urgency must be async, standard, or critical")
    result = await db.update_task(task_id, urgency=urgency)
    if not result:
        raise HTTPException(404, "Task not found")
    asyncio.create_task(mission_control.emit_narration(
        "ui-action", "primary_agent",
        f"Task urgency updated to {urgency.upper()}."))
    return result


@router.patch("/tasks/{task_id}/reminder")
async def update_task_reminder(request: Request, task_id: str, body: dict):
    """Set the manual reminder frequency override for a task."""
    user_id = get_current_user(request)
    task_id = sanitize_id(task_id)
    existing = await db.get_task(task_id)
    _check_ownership(existing, user_id, "Task")

    interval = body.get("reminder_interval")

    # Validate — strict format checking with length limits
    quick_presets = {"15m", "30m", "1h", "2h", "4h", "8h",
                     "daily", "weekdays", "weekly", "monthly", "suppress"}
    valid = False
    if interval is None:
        valid = True
    elif isinstance(interval, str) and len(interval) <= 200:
        if interval in quick_presets:
            valid = True
        elif re.match(r'^\d{1,4}[mhd]$', interval):
            valid = True
        elif interval.startswith('days:') and '@' in interval:
            valid = True
        elif interval.startswith('once:'):
            valid = True
        elif interval.startswith('cron:'):
            valid = True

    if not valid:
        raise HTTPException(400, "Invalid reminder_interval format")

    result = await db.update_task(task_id, reminder_interval=interval)
    if not result:
        raise HTTPException(404, "Task not found")
    label = interval.upper() if interval else "DEFAULT"
    asyncio.create_task(mission_control.emit_narration(
        "ui-action", "primary_agent",
        f"Reminder frequency set to {label}."))
    return result
