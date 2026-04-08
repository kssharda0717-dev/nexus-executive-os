"""NEXUS Database — async SQLite with full schema + ownership isolation."""

import aiosqlite
import json
import logging
from datetime import datetime
from uuid import uuid4
from config import DB_PATH

logger = logging.getLogger("nexus.database")
_db: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    """Get or create the singleton database connection."""
    global _db
    if _db is None:
        _db = await aiosqlite.connect(str(DB_PATH))
        _db.row_factory = aiosqlite.Row
        await _db.execute("PRAGMA journal_mode=WAL")
        await _db.execute("PRAGMA foreign_keys=ON")
        await init_schema(_db)
    return _db


async def close_db():
    """Close the database connection."""
    global _db
    if _db:
        await _db.close()
        _db = None


async def init_schema(db: aiosqlite.Connection):
    """Initialize all tables."""
    statements = [
        """CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            linked_task_id TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )""",
        """CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',
            priority TEXT DEFAULT 'medium',
            urgency TEXT DEFAULT 'standard',
            reminder_interval TEXT,
            due_date TEXT,
            estimated_minutes INTEGER,
            tags TEXT DEFAULT '[]',
            dependencies TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )""",
        """CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '[]',
            linked_task_ids TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )""",
        """CREATE TABLE IF NOT EXISTS agent_runs (
            id TEXT PRIMARY KEY,
            user_message TEXT NOT NULL,
            intent TEXT,
            status TEXT DEFAULT 'running',
            plan TEXT,
            result TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS agent_steps (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            step_type TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (run_id) REFERENCES agent_runs(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_events_time ON events(start_time, end_time)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(due_date)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority)",
        "CREATE INDEX IF NOT EXISTS idx_notes_tags ON notes(tags)",
        "CREATE INDEX IF NOT EXISTS idx_agent_steps_run ON agent_steps(run_id)",
        # ── Learning Loop: User Corrections ──
        """CREATE TABLE IF NOT EXISTS user_corrections (
            id TEXT PRIMARY KEY,
            original_prompt TEXT NOT NULL,
            detected_error TEXT NOT NULL,
            final_correction TEXT NOT NULL,
            correction_type TEXT DEFAULT 'general',
            created_at TEXT DEFAULT (datetime('now'))
        )""",
        "CREATE INDEX IF NOT EXISTS idx_corrections_type ON user_corrections(correction_type)",
        "CREATE INDEX IF NOT EXISTS idx_corrections_time ON user_corrections(created_at DESC)",
        # ── Email Inbox Simulation (Mental Peace) ──
        """CREATE TABLE IF NOT EXISTS email_inbox (
            id TEXT PRIMARY KEY,
            sender TEXT NOT NULL,
            subject TEXT NOT NULL,
            preview TEXT,
            category TEXT DEFAULT 'paperwork',
            tldr TEXT,
            linked_note_id TEXT,
            is_read INTEGER DEFAULT 0,
            received_at TEXT DEFAULT (datetime('now'))
        )""",
        "CREATE INDEX IF NOT EXISTS idx_email_category ON email_inbox(category)",
        "CREATE INDEX IF NOT EXISTS idx_email_received ON email_inbox(received_at DESC)",
        # ── Users & Auth ──
        """CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        )""",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)",
        # ── Multi-Tenant Integrations ──
        """CREATE TABLE IF NOT EXISTS integrations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            access_token TEXT NOT NULL,
            refresh_token TEXT DEFAULT '',
            scopes TEXT DEFAULT '',
            token_expiry TEXT DEFAULT '',
            extra_data TEXT DEFAULT '{}',
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )""",
        # One active connection per provider per user
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_integrations_user_provider ON integrations(user_id, provider) WHERE is_active = 1",
        "CREATE INDEX IF NOT EXISTS idx_integrations_user ON integrations(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_integrations_provider ON integrations(provider)",
    ]
    for stmt in statements:
        await db.execute(stmt)
    await db.commit()

    # ── Migrations: add columns to existing tables ──
    _migrations = [
        ("tasks", "urgency", "'standard'"),
        ("tasks", "reminder_interval", "NULL"),
        # Ownership columns for zero-trust data isolation
        ("events", "owner_id", "'nexus_default_user'"),
        ("tasks", "owner_id", "'nexus_default_user'"),
        ("notes", "owner_id", "'nexus_default_user'"),
        ("email_inbox", "owner_id", "'nexus_default_user'"),
        ("agent_runs", "owner_id", "'nexus_default_user'"),
    ]
    for table, col, default in _migrations:
        try:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT DEFAULT {default}")
            await db.commit()
        except Exception:
            pass  # Column already exists

    # Create indexes for owner_id lookups
    for table in ("events", "tasks", "notes", "email_inbox", "agent_runs"):
        try:
            await db.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_owner ON {table}(owner_id)")
            await db.commit()
        except Exception:
            pass


# ── Column Whitelist (SQL Injection Prevention) ───────
# Only these column names are allowed in dynamic UPDATE SET clauses.
# This prevents injection via malicious field names like "x); DROP TABLE--"

_ALLOWED_COLUMNS = {
    "events": {"title", "description", "start_time", "end_time",
               "linked_task_id", "updated_at", "owner_id"},
    "tasks": {"title", "description", "status", "priority", "urgency",
              "reminder_interval", "due_date", "estimated_minutes",
              "tags", "dependencies", "updated_at", "owner_id"},
    "notes": {"title", "content", "tags", "linked_task_ids", "updated_at", "owner_id"},
    "agent_runs": {"status", "plan", "result", "completed_at", "owner_id", "intent"},
}

import re as _re

def _validate_fields(table: str, fields: dict) -> dict:
    """Validate and whitelist column names for dynamic SQL.

    Raises ValueError on any column not in the whitelist or with
    non-alphanumeric characters (SQL injection vector).
    """
    allowed = _ALLOWED_COLUMNS.get(table, set())
    validated = {}
    for key, value in fields.items():
        if not _re.match(r'^[a-z_][a-z0-9_]*$', key):
            raise ValueError(f"Invalid column name rejected: {key!r}")
        if key not in allowed:
            raise ValueError(f"Column '{key}' not permitted for table '{table}'")
        validated[key] = value
    return validated


# ── Calendar Operations ────────────────────────────────

async def create_event(title: str, start_time: str, end_time: str,
                       description: str = None, linked_task_id: str = None,
                       owner_id: str = "nexus_default_user") -> dict:
    db = await get_db()
    event_id = str(uuid4())
    now = datetime.now().isoformat()
    await db.execute(
        "INSERT INTO events (id, title, description, start_time, end_time, linked_task_id, owner_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (event_id, title, description, start_time, end_time, linked_task_id, owner_id, now, now)
    )
    await db.commit()
    return {"id": event_id, "title": title, "description": description,
            "start_time": start_time, "end_time": end_time,
            "linked_task_id": linked_task_id, "created_at": now}


async def list_events(start: str = None, end: str = None,
                      owner_id: str = None) -> list[dict]:
    db = await get_db()
    query = "SELECT * FROM events WHERE 1=1"
    params = []
    if owner_id:
        query += " AND owner_id = ?"
        params.append(owner_id)
    if start and end:
        query += " AND start_time >= ? AND end_time <= ?"
        params.extend([start, end])
    elif start:
        query += " AND start_time >= ?"
        params.append(start)
    query += " ORDER BY start_time"
    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_event(event_id: str) -> dict | None:
    db = await get_db()
    cursor = await db.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def update_event(event_id: str, **fields) -> dict | None:
    db = await get_db()
    fields["updated_at"] = datetime.now().isoformat()
    fields = _validate_fields("events", fields)
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [event_id]
    await db.execute(f"UPDATE events SET {sets} WHERE id = ?", vals)
    await db.commit()
    return await get_event(event_id)


async def delete_event(event_id: str) -> bool:
    db = await get_db()
    cursor = await db.execute("DELETE FROM events WHERE id = ?", (event_id,))
    await db.commit()
    return cursor.rowcount > 0


async def find_free_slots(start: str, end: str, duration_minutes: int) -> list[dict]:
    """Find free time slots between start and end that fit the given duration."""
    events = await list_events(start, end)
    free_slots = []
    current = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    for event in events:
        event_start = datetime.fromisoformat(event["start_time"])
        if (event_start - current).total_seconds() >= duration_minutes * 60:
            free_slots.append({
                "start": current.isoformat(),
                "end": event_start.isoformat(),
                "duration_minutes": int((event_start - current).total_seconds() / 60)
            })
        current = max(current, datetime.fromisoformat(event["end_time"]))

    if (end_dt - current).total_seconds() >= duration_minutes * 60:
        free_slots.append({
            "start": current.isoformat(),
            "end": end_dt.isoformat(),
            "duration_minutes": int((end_dt - current).total_seconds() / 60)
        })

    return free_slots


# ── Task Operations ────────────────────────────────────

async def create_task(title: str, description: str = None,
                      status: str = "pending", priority: str = "medium",
                      urgency: str = "standard", due_date: str = None,
                      estimated_minutes: int = None,
                      tags: list[str] = None, dependencies: list[str] = None,
                      owner_id: str = "nexus_default_user") -> dict:
    db = await get_db()
    task_id = str(uuid4())
    now = datetime.now().isoformat()
    await db.execute(
        "INSERT INTO tasks (id, title, description, status, priority, urgency, due_date, "
        "estimated_minutes, tags, dependencies, owner_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, title, description, status, priority, urgency, due_date,
         estimated_minutes, json.dumps(tags or []), json.dumps(dependencies or []),
         owner_id, now, now)
    )
    await db.commit()
    return {"id": task_id, "title": title, "description": description,
            "status": status, "priority": priority, "urgency": urgency,
            "due_date": due_date, "estimated_minutes": estimated_minutes,
            "tags": tags or [], "dependencies": dependencies or [],
            "created_at": now}


async def list_tasks(status: str = None, priority: str = None,
                     due_before: str = None, tags: list[str] = None,
                     owner_id: str = None) -> list[dict]:
    db = await get_db()
    query = "SELECT * FROM tasks WHERE 1=1"
    params = []
    if owner_id:
        query += " AND owner_id = ?"
        params.append(owner_id)
    if status:
        query += " AND status = ?"
        params.append(status)
    if priority:
        query += " AND priority = ?"
        params.append(priority)
    if due_before:
        query += " AND due_date <= ?"
        params.append(due_before)
    query += " ORDER BY CASE priority WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, due_date"
    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["dependencies"] = json.loads(d.get("dependencies") or "[]")
        if tags:
            if not any(t in d["tags"] for t in tags):
                continue
        results.append(d)
    return results


async def get_task(task_id: str) -> dict | None:
    db = await get_db()
    cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = await cursor.fetchone()
    if row:
        d = dict(row)
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["dependencies"] = json.loads(d.get("dependencies") or "[]")
        return d
    return None


async def update_task(task_id: str, **fields) -> dict | None:
    db = await get_db()
    fields["updated_at"] = datetime.now().isoformat()
    if "tags" in fields and isinstance(fields["tags"], list):
        fields["tags"] = json.dumps(fields["tags"])
    if "dependencies" in fields and isinstance(fields["dependencies"], list):
        fields["dependencies"] = json.dumps(fields["dependencies"])
    fields = _validate_fields("tasks", fields)
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [task_id]
    await db.execute(f"UPDATE tasks SET {sets} WHERE id = ?", vals)
    await db.commit()
    return await get_task(task_id)


async def complete_task(task_id: str) -> dict | None:
    return await update_task(task_id, status="completed")


async def delete_task(task_id: str) -> bool:
    """Delete a task by ID. Returns True if a row was actually deleted."""
    conn = await get_db()
    cursor = await conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    await conn.commit()
    return cursor.rowcount > 0


async def get_overdue_tasks() -> list[dict]:
    now = datetime.now().isoformat()
    return await list_tasks(status="pending", due_before=now)


# ── Note Operations ────────────────────────────────────

async def create_note(title: str, content: str,
                      tags: list[str] = None, linked_task_ids: list[str] = None,
                      owner_id: str = "nexus_default_user") -> dict:
    db = await get_db()
    note_id = str(uuid4())
    now = datetime.now().isoformat()
    await db.execute(
        "INSERT INTO notes (id, title, content, tags, linked_task_ids, owner_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (note_id, title, content, json.dumps(tags or []),
         json.dumps(linked_task_ids or []), owner_id, now, now)
    )
    await db.commit()
    return {"id": note_id, "title": title, "content": content,
            "tags": tags or [], "linked_task_ids": linked_task_ids or [],
            "created_at": now}


async def list_notes(tags: list[str] = None, owner_id: str = None) -> list[dict]:
    db = await get_db()
    query = "SELECT * FROM notes"
    params = []
    if owner_id:
        query += " WHERE owner_id = ?"
        params.append(owner_id)
    query += " ORDER BY updated_at DESC"
    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["linked_task_ids"] = json.loads(d.get("linked_task_ids") or "[]")
        if tags:
            if not any(t in d["tags"] for t in tags):
                continue
        results.append(d)
    return results


async def get_note(note_id: str) -> dict | None:
    db = await get_db()
    cursor = await db.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
    row = await cursor.fetchone()
    if row:
        d = dict(row)
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["linked_task_ids"] = json.loads(d.get("linked_task_ids") or "[]")
        return d
    return None


async def update_note(note_id: str, **fields) -> dict | None:
    db = await get_db()
    fields["updated_at"] = datetime.now().isoformat()
    if "tags" in fields and isinstance(fields["tags"], list):
        fields["tags"] = json.dumps(fields["tags"])
    if "linked_task_ids" in fields and isinstance(fields["linked_task_ids"], list):
        fields["linked_task_ids"] = json.dumps(fields["linked_task_ids"])
    fields = _validate_fields("notes", fields)
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [note_id]
    await db.execute(f"UPDATE notes SET {sets} WHERE id = ?", vals)
    await db.commit()
    return await get_note(note_id)


async def search_notes(query: str, owner_id: str = None) -> list[dict]:
    db = await get_db()
    sql = "SELECT * FROM notes WHERE (title LIKE ? OR content LIKE ?)"
    params = [f"%{query}%", f"%{query}%"]
    if owner_id:
        sql += " AND owner_id = ?"
        params.append(owner_id)
    sql += " ORDER BY updated_at DESC"
    cursor = await db.execute(sql, params)
    rows = await cursor.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["linked_task_ids"] = json.loads(d.get("linked_task_ids") or "[]")
        results.append(d)
    return results


async def append_to_note(note_id: str, content: str) -> dict | None:
    note = await get_note(note_id)
    if not note:
        return None
    new_content = note["content"] + "\n\n" + content
    return await update_note(note_id, content=new_content)


# ── Agent State Operations ─────────────────────────────

async def create_agent_run(user_message: str, intent: str = None) -> dict:
    db = await get_db()
    run_id = str(uuid4())
    now = datetime.now().isoformat()
    await db.execute(
        "INSERT INTO agent_runs (id, user_message, intent, status, created_at) VALUES (?, ?, ?, 'running', ?)",
        (run_id, user_message, intent, now)
    )
    await db.commit()
    return {"id": run_id, "user_message": user_message, "intent": intent,
            "status": "running", "created_at": now}


async def update_agent_run(run_id: str, **fields) -> dict | None:
    db = await get_db()
    if "plan" in fields and isinstance(fields["plan"], dict):
        fields["plan"] = json.dumps(fields["plan"])
    fields = _validate_fields("agent_runs", fields)
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [run_id]
    await db.execute(f"UPDATE agent_runs SET {sets} WHERE id = ?", vals)
    await db.commit()
    cursor = await db.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def log_agent_step(run_id: str, agent_name: str, step_type: str, content: dict) -> dict:
    db = await get_db()
    step_id = str(uuid4())
    now = datetime.now().isoformat()
    await db.execute(
        "INSERT INTO agent_steps (id, run_id, agent_name, step_type, content, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (step_id, run_id, agent_name, step_type, json.dumps(content), now)
    )
    await db.commit()
    return {"id": step_id, "run_id": run_id, "agent_name": agent_name,
            "step_type": step_type, "content": content, "created_at": now}


# ── Learning Loop: User Corrections ──────────────────

async def log_correction(original_prompt: str, detected_error: str,
                         final_correction: str, correction_type: str = "general") -> dict:
    """Record a user correction for the learning loop.

    Called when the agent detects the user is correcting a previous action
    (e.g., "No, change the date to April 8th") and successfully applies
    the fix via an update tool.
    """
    db = await get_db()
    correction_id = str(uuid4())
    now = datetime.now().isoformat()
    await db.execute(
        "INSERT INTO user_corrections (id, original_prompt, detected_error, "
        "final_correction, correction_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (correction_id, original_prompt, detected_error, final_correction,
         correction_type, now)
    )
    await db.commit()
    return {"id": correction_id, "correction_type": correction_type, "created_at": now}


async def get_top_corrections(limit: int = 5) -> list[dict]:
    """Get the most frequent correction patterns for session injection.

    Groups by correction_type and returns the most common patterns
    so the system prompt can include personalized user behaviors.
    """
    db = await get_db()
    cursor = await db.execute("""
        SELECT correction_type,
               COUNT(*) as frequency,
               GROUP_CONCAT(detected_error, ' | ') as examples
        FROM user_corrections
        GROUP BY correction_type
        ORDER BY frequency DESC
        LIMIT ?
    """, (limit,))
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_recent_corrections(limit: int = 10) -> list[dict]:
    """Get the most recent corrections (for debugging / observability)."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM user_corrections ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_action_snapshot(limit: int = 3) -> list[dict]:
    """Build the Action Snapshot: the last N database mutations.

    Queries agent_steps for the most recent tool_result entries that
    created, updated, or completed items. Returns a lightweight JSON
    summary the LLM can use to resolve "that", "it", "the task", etc.
    """
    db = await get_db()
    cursor = await db.execute("""
        SELECT content, created_at FROM agent_steps
        WHERE step_type = 'tool_result'
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit * 3,))  # Over-fetch since not all results are mutations
    rows = await cursor.fetchall()

    snapshot = []
    for row in rows:
        if len(snapshot) >= limit:
            break
        try:
            data = json.loads(row["content"])
            tool = data.get("tool", "")
            result = data.get("result", {})

            # Only include mutation operations (create/update/complete)
            if not any(kw in tool for kw in ("create", "update", "complete", "delete")):
                continue

            entry = {
                "tool": tool,
                "timestamp": row["created_at"],
            }

            # Extract the most useful fields from the result
            if isinstance(result, dict):
                for field in ("id", "title", "due_date", "start_time", "end_time",
                              "status", "priority"):
                    if field in result:
                        entry[field] = result[field]

            if "id" in entry or "title" in entry:
                snapshot.append(entry)
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    return snapshot


# ── Email Inbox Simulation (Mental Peace) ─────────────

async def create_email(sender: str, subject: str, preview: str = None,
                       category: str = "paperwork", tldr: str = None,
                       linked_note_id: str = None) -> dict:
    """Simulate receiving an email — used by the Email Mental Peace feature."""
    db = await get_db()
    email_id = str(uuid4())
    now = datetime.now().isoformat()
    await db.execute(
        "INSERT INTO email_inbox (id, sender, subject, preview, category, tldr, "
        "linked_note_id, is_read, received_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
        (email_id, sender, subject, preview, category, tldr, linked_note_id, now)
    )
    await db.commit()
    return {"id": email_id, "sender": sender, "subject": subject,
            "preview": preview, "category": category, "tldr": tldr,
            "linked_note_id": linked_note_id, "is_read": False, "received_at": now}


async def list_emails(category: str = None, unread_only: bool = False) -> list[dict]:
    """List emails, optionally filtered by category or unread status."""
    db = await get_db()
    query = "SELECT * FROM email_inbox WHERE 1=1"
    params = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if unread_only:
        query += " AND is_read = 0"
    query += " ORDER BY received_at DESC"
    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def mark_email_read(email_id: str) -> bool:
    """Mark an email as read."""
    db = await get_db()
    cursor = await db.execute(
        "UPDATE email_inbox SET is_read = 1 WHERE id = ?", (email_id,))
    await db.commit()
    return cursor.rowcount > 0


async def get_email_counts() -> dict:
    """Get email counts by category for the inbox HUD."""
    db = await get_db()
    cursor = await db.execute("""
        SELECT category, COUNT(*) as total,
               SUM(CASE WHEN is_read = 0 THEN 1 ELSE 0 END) as unread
        FROM email_inbox GROUP BY category
    """)
    rows = await cursor.fetchall()
    result = {}
    for r in rows:
        d = dict(r)
        result[d["category"]] = {"total": d["total"], "unread": d["unread"]}
    return result


# ── User Operations ───────────────────────────────────

async def create_user(username: str, password_hash: str,
                      display_name: str = "") -> dict:
    """Create a new user account."""
    db = await get_db()
    user_id = str(uuid4())
    now = datetime.now().isoformat()
    await db.execute(
        "INSERT INTO users (id, username, password_hash, display_name, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, username, password_hash, display_name, now)
    )
    await db.commit()
    return {"id": user_id, "username": username, "display_name": display_name,
            "created_at": now}


async def get_user_by_username(username: str) -> dict | None:
    """Get a user by username (for login)."""
    db = await get_db()
    cursor = await db.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_user_by_id(user_id: str) -> dict | None:
    """Get a user by ID."""
    db = await get_db()
    cursor = await db.execute("SELECT id, username, display_name, created_at FROM users WHERE id = ?",
                              (user_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


# ── Integration Operations (Multi-Tenant) ────────────

VALID_PROVIDERS = {"outlook", "slack", "linkedin"}


async def upsert_integration(user_id: str, provider: str,
                              access_token: str, refresh_token: str = "",
                              scopes: str = "", token_expiry: str = "",
                              extra_data: dict = None) -> dict:
    """Create or update a user's integration for a provider.

    Uses INSERT-OR-REPLACE to enforce one active connection per provider per user.
    Tokens should already be encrypted before reaching this function.
    """
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"Invalid provider: {provider}. Must be one of: {VALID_PROVIDERS}")

    db = await get_db()
    now = datetime.now().isoformat()
    integration_id = str(uuid4())
    extra_json = json.dumps(extra_data or {})

    # Deactivate any existing integration for this user+provider
    await db.execute(
        "UPDATE integrations SET is_active = 0, updated_at = ? "
        "WHERE user_id = ? AND provider = ? AND is_active = 1",
        (now, user_id, provider)
    )

    # Insert new active integration
    await db.execute(
        "INSERT INTO integrations (id, user_id, provider, access_token, refresh_token, "
        "scopes, token_expiry, extra_data, is_active, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
        (integration_id, user_id, provider, access_token, refresh_token,
         scopes, token_expiry, extra_json, now, now)
    )
    await db.commit()

    return {
        "id": integration_id,
        "user_id": user_id,
        "provider": provider,
        "is_active": True,
        "scopes": scopes,
        "created_at": now,
    }


async def get_integration(user_id: str, provider: str) -> dict | None:
    """Get the active integration for a user + provider.

    Returns the full row including encrypted tokens.
    """
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM integrations "
        "WHERE user_id = ? AND provider = ? AND is_active = 1 "
        "ORDER BY updated_at DESC LIMIT 1",
        (user_id, provider)
    )
    row = await cursor.fetchone()
    if not row:
        return None
    result = dict(row)
    # Parse extra_data JSON
    try:
        result["extra_data"] = json.loads(result.get("extra_data", "{}"))
    except (json.JSONDecodeError, TypeError):
        result["extra_data"] = {}
    return result


async def get_user_integrations(user_id: str) -> list[dict]:
    """Get all active integrations for a user (metadata only, no tokens)."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, user_id, provider, scopes, is_active, created_at, updated_at "
        "FROM integrations WHERE user_id = ? AND is_active = 1 "
        "ORDER BY provider",
        (user_id,)
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def delete_integration(user_id: str, provider: str) -> bool:
    """Soft-delete (deactivate) a user's integration for a provider.

    Wipes tokens from the row for security.
    """
    db = await get_db()
    now = datetime.now().isoformat()
    cursor = await db.execute(
        "UPDATE integrations SET is_active = 0, access_token = '', "
        "refresh_token = '', updated_at = ? "
        "WHERE user_id = ? AND provider = ? AND is_active = 1",
        (now, user_id, provider)
    )
    await db.commit()
    return cursor.rowcount > 0


async def update_integration_tokens(user_id: str, provider: str,
                                     access_token: str, refresh_token: str = "",
                                     token_expiry: str = "") -> bool:
    """Update tokens for an existing active integration (e.g., after refresh)."""
    db = await get_db()
    now = datetime.now().isoformat()
    cursor = await db.execute(
        "UPDATE integrations SET access_token = ?, refresh_token = ?, "
        "token_expiry = ?, updated_at = ? "
        "WHERE user_id = ? AND provider = ? AND is_active = 1",
        (access_token, refresh_token, token_expiry, now, user_id, provider)
    )
    await db.commit()
    return cursor.rowcount > 0
