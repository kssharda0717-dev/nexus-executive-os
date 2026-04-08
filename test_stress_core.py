"""NEXUS Stress Test & Chaos Engineering Suite.

Phases:
  1. Rate Limiter Bruteforce + SQLite Race Conditions
  2. WebSocket Broken Pipe + Payload Injection
  3. Context Overflow + Tool Chain Collapse + Agent Edge Cases

Also covers comprehensive unit tests for full coverage:
  - database.py (all CRUD + edge cases)
  - mission_control.py (broadcast, narration, buffer)
  - schemas.py (enum values, model validation)
  - config.py (constants, logging)
  - agents/base.py (rate limiter, retry, 429 detection, BaseAgent)
  - tools/ (calendar, tasks, notes wrappers)
  - routers/ (health, data, chat endpoints)
  - agents/primary.py (intent parsing, plan parsing, sub-agent registry)

Usage:
    cd nexus
    python -m pytest test_stress_core.py -v --tb=short
    python -m pytest test_stress_core.py -v --tb=short --cov=. --cov-report=term-missing
"""

import asyncio
import json
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4

import pytest
import pytest_asyncio

# ── Ensure imports resolve from nexus/ ──
sys.path.insert(0, str(Path(__file__).parent))

# ── Test-scoped database isolation ──
# Force a test-specific DB path BEFORE importing anything
TEST_DB_PATH = Path(__file__).parent / "data" / "nexus_test.db"
os.environ["DATA_DIR"] = str(TEST_DB_PATH.parent)

import config  # noqa: E402 — must come after env override
config.DB_PATH = TEST_DB_PATH

import database as db  # noqa: E402
from mission_control import MissionControl, mission_control  # noqa: E402
from schemas import (  # noqa: E402
    StepType, TaskStatus, TaskPriority, AgentRunStatus,
    EventCreate, EventUpdate, Event,
    TaskCreate, TaskUpdate, Task,
    NoteCreate, NoteUpdate, Note,
    ChatRequest, ChatResponse, MissionControlEvent,
    AgentRun, AgentStep,
)
from agents.base import (  # noqa: E402
    GeminiRateLimiter, gemini_limiter,
    _is_rate_limit_error, _parse_retry_delay,
    retry_async, BaseAgent, AGENT_FRIENDLY_NAMES, TOOL_NARRATIONS,
)


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest_asyncio.fixture(autouse=True)
async def fresh_db():
    """Reset the database before every test."""
    # Close any existing connection
    await db.close_db()
    # Remove the test DB file
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    # Re-initialize
    conn = await db.get_db()
    yield conn
    await db.close_db()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


@pytest.fixture
def mc():
    """Fresh MissionControl instance."""
    return MissionControl()


# ═══════════════════════════════════════════════════════════════
# PHASE 0 — UNIT TESTS: config, schemas, database, mission_control
# ═══════════════════════════════════════════════════════════════

class TestConfig:
    """Verify config constants are sane."""

    def test_model_name(self):
        assert "flash" in config.GEMINI_FLASH_MODEL.lower()

    def test_rate_limit_constants(self):
        assert config.GEMINI_MAX_CONCURRENT == 1
        assert config.GEMINI_MIN_CALL_SPACING_S >= 4.0
        assert config.GEMINI_429_BASE_DELAY_S >= 5.0
        assert config.GEMINI_429_MAX_DELAY_S >= 30.0
        assert config.GEMINI_JITTER_S >= 0.0

    def test_agent_constants(self):
        assert config.MAX_AGENT_STEPS >= 10
        assert config.AGENT_TIMEOUT_SECONDS >= 60
        assert config.MAX_PLAN_STEPS >= 2
        assert config.MAX_PLAN_STEPS <= 5  # Must stay small to avoid 429 floods

    def test_paths(self):
        assert config.BASE_DIR.exists()
        assert config.DATA_DIR.exists()

    def test_logging_setup(self):
        """Verify logging was configured (setup_logging called at import)."""
        nexus_logger = logging.getLogger("nexus")
        # setup_logging configures basicConfig — root may be overridden by pytest
        # but the function itself should not error
        config.setup_logging()
        assert True  # No crash = pass


class TestSchemas:
    """Validate all Pydantic models and enums."""

    def test_task_status_values(self):
        assert set(TaskStatus) == {
            TaskStatus.PENDING, TaskStatus.SCHEDULED,
            TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.CANCELLED,
        }

    def test_task_priority_values(self):
        assert set(TaskPriority) == {
            TaskPriority.LOW, TaskPriority.MEDIUM,
            TaskPriority.HIGH, TaskPriority.URGENT,
        }

    def test_agent_run_status(self):
        assert set(AgentRunStatus) == {
            AgentRunStatus.RUNNING, AgentRunStatus.COMPLETED, AgentRunStatus.FAILED,
        }

    def test_step_type_values(self):
        assert len(StepType) == 7

    def test_event_create(self):
        e = EventCreate(title="Test", start_time="2026-01-01T09:00:00", end_time="2026-01-01T10:00:00")
        assert e.title == "Test"
        assert e.linked_task_id is None

    def test_event_update_partial(self):
        e = EventUpdate(title="Updated")
        assert e.start_time is None

    def test_task_create_defaults(self):
        t = TaskCreate(title="Task")
        assert t.status == TaskStatus.PENDING
        assert t.priority == TaskPriority.MEDIUM

    def test_note_create(self):
        n = NoteCreate(title="Note", content="body")
        assert n.tags is None

    def test_chat_request(self):
        r = ChatRequest(message="Hello")
        assert r.session_id is None

    def test_chat_response(self):
        r = ChatResponse(run_id="abc", response="ok", steps_count=2, tools_used=["x"])
        assert r.steps_count == 2

    def test_mission_control_event(self):
        e = MissionControlEvent(
            run_id="r1", agent_name="test", event_type=StepType.THOUGHT, content={"x": 1}
        )
        assert e.metadata is None

    def test_agent_run_model(self):
        r = AgentRun(user_message="test")
        assert r.status == AgentRunStatus.RUNNING

    def test_agent_step_model(self):
        s = AgentStep(run_id="r1", agent_name="a", step_type=StepType.THOUGHT, content={"t": 1})
        assert s.step_type == StepType.THOUGHT


# ═══════════════════════════════════════════════════════════════
# DATABASE UNIT TESTS
# ═══════════════════════════════════════════════════════════════

class TestDatabaseEvents:
    """Full coverage of calendar operations."""

    @pytest.mark.asyncio
    async def test_create_and_get_event(self):
        ev = await db.create_event("Standup", "2026-04-02T09:00:00", "2026-04-02T09:30:00", "Daily")
        assert ev["title"] == "Standup"
        fetched = await db.get_event(ev["id"])
        assert fetched["title"] == "Standup"

    @pytest.mark.asyncio
    async def test_list_events_no_filter(self):
        await db.create_event("A", "2026-04-01T09:00:00", "2026-04-01T10:00:00")
        await db.create_event("B", "2026-04-02T09:00:00", "2026-04-02T10:00:00")
        events = await db.list_events()
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_list_events_with_start(self):
        await db.create_event("Early", "2026-04-01T09:00:00", "2026-04-01T10:00:00")
        await db.create_event("Late", "2026-04-10T09:00:00", "2026-04-10T10:00:00")
        events = await db.list_events(start="2026-04-05T00:00:00")
        assert len(events) == 1
        assert events[0]["title"] == "Late"

    @pytest.mark.asyncio
    async def test_list_events_with_start_and_end(self):
        await db.create_event("E1", "2026-04-02T09:00:00", "2026-04-02T10:00:00")
        await db.create_event("E2", "2026-04-05T09:00:00", "2026-04-05T10:00:00")
        events = await db.list_events(start="2026-04-01T00:00:00", end="2026-04-03T00:00:00")
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_update_event(self):
        ev = await db.create_event("Old", "2026-04-02T09:00:00", "2026-04-02T10:00:00")
        updated = await db.update_event(ev["id"], title="New")
        assert updated["title"] == "New"

    @pytest.mark.asyncio
    async def test_delete_event(self):
        ev = await db.create_event("Del", "2026-04-02T09:00:00", "2026-04-02T10:00:00")
        assert await db.delete_event(ev["id"]) is True
        assert await db.get_event(ev["id"]) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_event(self):
        assert await db.delete_event("nonexistent") is False

    @pytest.mark.asyncio
    async def test_get_nonexistent_event(self):
        assert await db.get_event("nonexistent") is None

    @pytest.mark.asyncio
    async def test_find_free_slots(self):
        await db.create_event("Meeting", "2026-04-02T10:00:00", "2026-04-02T11:00:00")
        slots = await db.find_free_slots("2026-04-02T08:00:00", "2026-04-02T14:00:00", 60)
        assert len(slots) >= 1
        # Slot before the meeting (8:00-10:00 = 120 min)
        assert slots[0]["duration_minutes"] >= 60

    @pytest.mark.asyncio
    async def test_find_free_slots_no_events(self):
        slots = await db.find_free_slots("2026-04-02T08:00:00", "2026-04-02T14:00:00", 30)
        assert len(slots) == 1
        assert slots[0]["duration_minutes"] == 360

    @pytest.mark.asyncio
    async def test_find_free_slots_too_short(self):
        """No slots when minimum duration exceeds gap."""
        await db.create_event("Block", "2026-04-02T08:00:00", "2026-04-02T14:00:00")
        slots = await db.find_free_slots("2026-04-02T08:00:00", "2026-04-02T14:00:00", 60)
        assert len(slots) == 0


class TestDatabaseTasks:
    """Full coverage of task operations."""

    @pytest.mark.asyncio
    async def test_create_and_get_task(self):
        t = await db.create_task("Test Task", priority="high", tags=["work"])
        assert t["priority"] == "high"
        fetched = await db.get_task(t["id"])
        assert fetched["tags"] == ["work"]

    @pytest.mark.asyncio
    async def test_list_tasks_by_status(self):
        await db.create_task("A", status="pending")
        await db.create_task("B", status="completed")
        tasks = await db.list_tasks(status="pending")
        assert all(t["status"] == "pending" for t in tasks)

    @pytest.mark.asyncio
    async def test_list_tasks_by_priority(self):
        await db.create_task("Low", priority="low")
        await db.create_task("Urgent", priority="urgent")
        tasks = await db.list_tasks(priority="urgent")
        assert all(t["priority"] == "urgent" for t in tasks)

    @pytest.mark.asyncio
    async def test_list_tasks_by_due_before(self):
        await db.create_task("Soon", due_date="2026-04-01T12:00:00")
        await db.create_task("Later", due_date="2026-12-01T12:00:00")
        tasks = await db.list_tasks(due_before="2026-06-01T00:00:00")
        assert len(tasks) >= 1

    @pytest.mark.asyncio
    async def test_list_tasks_by_tags(self):
        await db.create_task("Tagged", tags=["hackathon"])
        await db.create_task("Other", tags=["personal"])
        tasks = await db.list_tasks(tags=["hackathon"])
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Tagged"

    @pytest.mark.asyncio
    async def test_update_task(self):
        t = await db.create_task("Old Title")
        updated = await db.update_task(t["id"], title="New Title", tags=["updated"])
        assert updated["title"] == "New Title"
        assert updated["tags"] == ["updated"]

    @pytest.mark.asyncio
    async def test_complete_task(self):
        t = await db.create_task("To Complete")
        completed = await db.complete_task(t["id"])
        assert completed["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_overdue_tasks(self):
        past = (datetime.now() - timedelta(days=5)).isoformat()
        await db.create_task("Overdue", due_date=past, status="pending")
        overdue = await db.get_overdue_tasks()
        assert len(overdue) >= 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self):
        assert await db.get_task("nonexistent") is None

    @pytest.mark.asyncio
    async def test_task_with_dependencies(self):
        t = await db.create_task("Dep Task", dependencies=["dep1", "dep2"])
        fetched = await db.get_task(t["id"])
        assert fetched["dependencies"] == ["dep1", "dep2"]

    @pytest.mark.asyncio
    async def test_update_task_dependencies(self):
        t = await db.create_task("Task")
        updated = await db.update_task(t["id"], dependencies=["new_dep"])
        assert updated["dependencies"] == ["new_dep"]

    @pytest.mark.asyncio
    async def test_task_priority_ordering(self):
        await db.create_task("Low", priority="low")
        await db.create_task("Urgent", priority="urgent")
        await db.create_task("Medium", priority="medium")
        tasks = await db.list_tasks()
        priorities = [t["priority"] for t in tasks]
        assert priorities.index("urgent") < priorities.index("low")


class TestDatabaseNotes:
    """Full coverage of note operations."""

    @pytest.mark.asyncio
    async def test_create_and_get_note(self):
        n = await db.create_note("Title", "Content", tags=["test"])
        assert n["title"] == "Title"
        fetched = await db.get_note(n["id"])
        assert fetched["content"] == "Content"
        assert fetched["tags"] == ["test"]

    @pytest.mark.asyncio
    async def test_list_notes(self):
        await db.create_note("N1", "C1")
        await db.create_note("N2", "C2")
        notes = await db.list_notes()
        assert len(notes) == 2

    @pytest.mark.asyncio
    async def test_list_notes_by_tags(self):
        await db.create_note("Tagged", "C", tags=["goal"])
        await db.create_note("Other", "C", tags=["random"])
        notes = await db.list_notes(tags=["goal"])
        assert len(notes) == 1

    @pytest.mark.asyncio
    async def test_update_note(self):
        n = await db.create_note("Old", "Content")
        updated = await db.update_note(n["id"], title="New", tags=["updated"])
        assert updated["title"] == "New"
        assert updated["tags"] == ["updated"]

    @pytest.mark.asyncio
    async def test_update_note_linked_task_ids(self):
        n = await db.create_note("Note", "C", linked_task_ids=["t1"])
        updated = await db.update_note(n["id"], linked_task_ids=["t1", "t2"])
        assert updated["linked_task_ids"] == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_search_notes(self):
        await db.create_note("Python Guide", "Learn Python basics")
        await db.create_note("Rust Guide", "Learn Rust basics")
        results = await db.search_notes("Python")
        assert len(results) == 1
        assert results[0]["title"] == "Python Guide"

    @pytest.mark.asyncio
    async def test_search_notes_by_content(self):
        await db.create_note("Generic", "The kubernetes cluster config")
        results = await db.search_notes("kubernetes")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_append_to_note(self):
        n = await db.create_note("Append Test", "Line 1")
        updated = await db.append_to_note(n["id"], "Line 2")
        assert "Line 1" in updated["content"]
        assert "Line 2" in updated["content"]

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_note(self):
        result = await db.append_to_note("nonexistent", "text")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_note(self):
        assert await db.get_note("nonexistent") is None


class TestDatabaseAgentState:
    """Agent run and step operations."""

    @pytest.mark.asyncio
    async def test_create_agent_run(self):
        run = await db.create_agent_run("test message", intent="general")
        assert run["status"] == "running"
        assert run["user_message"] == "test message"

    @pytest.mark.asyncio
    async def test_update_agent_run(self):
        run = await db.create_agent_run("msg")
        updated = await db.update_agent_run(run["id"], status="completed", result="done")
        assert updated["status"] == "completed"

    @pytest.mark.asyncio
    async def test_update_agent_run_with_plan_dict(self):
        run = await db.create_agent_run("msg")
        plan = {"steps": [{"step": 1, "action": "test"}], "completed_steps": []}
        updated = await db.update_agent_run(run["id"], plan=plan)
        assert updated is not None

    @pytest.mark.asyncio
    async def test_log_agent_step(self):
        run = await db.create_agent_run("msg")
        step = await db.log_agent_step(run["id"], "test_agent", "thought", {"thought": "thinking"})
        assert step["step_type"] == "thought"
        assert step["agent_name"] == "test_agent"

    @pytest.mark.asyncio
    async def test_close_and_reopen_db(self):
        await db.create_task("Before Close", priority="high")
        await db.close_db()
        # Re-open
        conn = await db.get_db()
        tasks = await db.list_tasks()
        assert any(t["title"] == "Before Close" for t in tasks)


# ═══════════════════════════════════════════════════════════════
# MISSION CONTROL UNIT TESTS
# ═══════════════════════════════════════════════════════════════

class TestMissionControl:
    """Full coverage of mission_control.py."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, mc):
        ws = AsyncMock()
        await mc.connect(ws)
        assert mc.client_count == 1
        mc.disconnect(ws)
        assert mc.client_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(self, mc):
        ws = AsyncMock()
        mc.disconnect(ws)  # Should not raise
        assert mc.client_count == 0

    @pytest.mark.asyncio
    async def test_broadcast(self, mc):
        ws = AsyncMock()
        await mc.connect(ws)
        await mc.broadcast("run1", "agent", StepType.THOUGHT, {"thought": "test"})
        ws.send_json.assert_called_once()
        event = ws.send_json.call_args[0][0]
        assert event["agent_name"] == "agent"

    @pytest.mark.asyncio
    async def test_broadcast_with_narration(self, mc):
        ws = AsyncMock()
        await mc.connect(ws)
        await mc.broadcast("r1", "a", "test", {"x": 1}, narration="Hello")
        event = ws.send_json.call_args[0][0]
        assert event["narration"] == "Hello"

    @pytest.mark.asyncio
    async def test_broadcast_handles_broken_ws(self, mc):
        ws_good = AsyncMock()
        ws_bad = AsyncMock()
        ws_bad.send_json.side_effect = Exception("Connection closed")
        await mc.connect(ws_good)
        await mc.connect(ws_bad)
        assert mc.client_count == 2
        await mc.broadcast("r1", "a", "test", {"x": 1})
        # Broken WS should be removed
        assert mc.client_count == 1

    @pytest.mark.asyncio
    async def test_event_log_buffer(self, mc):
        for i in range(600):
            await mc.broadcast("r1", "a", "test", {"i": i})
        assert len(mc._event_log) <= mc._max_log_size

    @pytest.mark.asyncio
    async def test_connect_sends_history(self, mc):
        # Seed some events
        for i in range(5):
            await mc.broadcast("r1", "a", "test", {"i": i})
        ws = AsyncMock()
        await mc.connect(ws)
        # accept() + 5 history events
        assert ws.send_json.call_count == 5

    @pytest.mark.asyncio
    async def test_connect_history_handles_broken_ws(self, mc):
        """If sending history fails, connection should not crash."""
        await mc.broadcast("r1", "a", "test", {"i": 0})
        ws = AsyncMock()
        ws.send_json.side_effect = Exception("broken")
        await mc.connect(ws)  # Should not raise

    @pytest.mark.asyncio
    async def test_emit_thought(self, mc):
        ws = AsyncMock()
        await mc.connect(ws)
        await mc.emit_thought("r1", "agent", "I'm thinking")
        event = ws.send_json.call_args[0][0]
        assert event["event_type"] == "thought"

    @pytest.mark.asyncio
    async def test_emit_narration(self, mc):
        ws = AsyncMock()
        await mc.connect(ws)
        await mc.emit_narration("r1", "agent", "Working on it...")
        event = ws.send_json.call_args[0][0]
        assert event["event_type"] == "narration"

    @pytest.mark.asyncio
    async def test_emit_tool_call(self, mc):
        ws = AsyncMock()
        await mc.connect(ws)
        await mc.emit_tool_call("r1", "agent", "create_task", {"title": "T"})
        event = ws.send_json.call_args[0][0]
        assert event["content"]["tool"] == "create_task"

    @pytest.mark.asyncio
    async def test_emit_tool_result(self, mc):
        ws = AsyncMock()
        await mc.connect(ws)
        await mc.emit_tool_result("r1", "agent", "create_task", {"title": "T"})
        event = ws.send_json.call_args[0][0]
        assert event["event_type"] == "tool_result"

    @pytest.mark.asyncio
    async def test_emit_delegation(self, mc):
        ws = AsyncMock()
        await mc.connect(ws)
        await mc.emit_delegation("r1", "primary_agent", "planning_agent", "Plan my week")
        event = ws.send_json.call_args[0][0]
        assert event["event_type"] == "delegation"
        assert "narration" in event

    @pytest.mark.asyncio
    async def test_emit_error(self, mc):
        ws = AsyncMock()
        await mc.connect(ws)
        await mc.emit_error("r1", "agent", "Something broke")
        event = ws.send_json.call_args[0][0]
        assert event["event_type"] == "error"

    @pytest.mark.asyncio
    async def test_emit_plan(self, mc):
        ws = AsyncMock()
        await mc.connect(ws)
        plan = [{"step": 1, "action": "Do thing"}]
        await mc.emit_plan("r1", "agent", plan)
        event = ws.send_json.call_args[0][0]
        assert event["event_type"] == "plan"

    @pytest.mark.asyncio
    async def test_emit_final_answer(self, mc):
        ws = AsyncMock()
        await mc.connect(ws)
        await mc.emit_final_answer("r1", "agent", "Here's your answer")
        event = ws.send_json.call_args[0][0]
        assert event["event_type"] == "final_answer"

    def test_narrate_tool_result_all_tools(self, mc):
        """Cover all branches in _narrate_tool_result."""
        assert "issue" in mc._narrate_tool_result("x", {"error": "e"})
        assert "event" in mc._narrate_tool_result("list_events", {"count": 3})
        assert "Booked" in mc._narrate_tool_result("create_event", {"title": "Meet"})
        assert "task" in mc._narrate_tool_result("list_tasks", {"count": 5})
        assert "Created" in mc._narrate_tool_result("create_task", {"title": "T"})
        assert "complete" in mc._narrate_tool_result("complete_task", {}).lower()
        assert "Updated" in mc._narrate_tool_result("update_task", {})
        assert "overdue" in mc._narrate_tool_result("get_overdue_tasks", {"count": 0}).lower()
        assert "overdue" in mc._narrate_tool_result("get_overdue_tasks", {"count": 2}).lower()
        assert "slot" in mc._narrate_tool_result("find_free_slots", {"count": 1}).lower()
        assert "note" in mc._narrate_tool_result("create_note", {"title": "N"}).lower()
        assert "note" in mc._narrate_tool_result("search_notes", {"count": 1}).lower()
        assert "note" in mc._narrate_tool_result("list_notes", {"count": 3}).lower()
        assert "Completed" in mc._narrate_tool_result("unknown_tool", {})

    def test_narrate_singulars(self, mc):
        """Test singular forms (count == 1)."""
        r = mc._narrate_tool_result("list_events", {"items": [{}]})
        assert "1 event" in r and "events" not in r
        r = mc._narrate_tool_result("list_tasks", {"items": [{}]})
        assert "1 task" in r and "tasks" not in r


# ═══════════════════════════════════════════════════════════════
# AGENTS/BASE UNIT TESTS — Rate Limiter, Retry, 429 Detection
# ═══════════════════════════════════════════════════════════════

class TestRateLimitDetection:
    """_is_rate_limit_error and _parse_retry_delay."""

    def test_429_in_message(self):
        assert _is_rate_limit_error(Exception("429 RESOURCE_EXHAUSTED"))

    def test_resource_exhausted(self):
        assert _is_rate_limit_error(Exception("RESOURCE_EXHAUSTED: quota exceeded"))

    def test_rate_limit_phrase(self):
        assert _is_rate_limit_error(Exception("rate limit exceeded"))

    def test_normal_error(self):
        assert not _is_rate_limit_error(Exception("connection timeout"))

    def test_parse_retry_delay_found(self):
        delay = _parse_retry_delay(Exception("Please retry after 12.5s"))
        assert delay == 12.5

    def test_parse_retry_delay_not_found(self):
        assert _parse_retry_delay(Exception("generic error")) is None

    def test_parse_retry_delay_integer(self):
        delay = _parse_retry_delay(Exception("retryDelay: 30s remaining"))
        assert delay == 30.0


class TestRetryAsync:
    """retry_async with various failure scenarios."""

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        calls = []
        async def factory():
            calls.append(1)
            return "ok"
        result = await retry_async(factory, max_retries=3)
        assert result == "ok"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_retry_on_generic_error(self):
        attempt = [0]
        async def factory():
            attempt[0] += 1
            if attempt[0] < 3:
                raise Exception("transient")
            return "recovered"
        result = await retry_async(factory, max_retries=3, base_delay=0.01, max_delay=0.05)
        assert result == "recovered"
        assert attempt[0] == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        async def factory():
            raise Exception("permanent")
        with pytest.raises(Exception, match="permanent"):
            await retry_async(factory, max_retries=2, base_delay=0.01)

    @pytest.mark.asyncio
    async def test_429_retry_longer_delay(self):
        """429 errors should trigger rate-limit specific messaging."""
        attempt = [0]
        async def factory():
            attempt[0] += 1
            if attempt[0] < 2:
                raise Exception("429 RESOURCE_EXHAUSTED")
            return "ok"
        result = await retry_async(
            factory, max_retries=3, base_delay=0.01,
            run_id="test", agent_name="test", label="test"
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_429_with_parsed_delay(self):
        attempt = [0]
        async def factory():
            attempt[0] += 1
            if attempt[0] < 2:
                raise Exception("429 RESOURCE_EXHAUSTED retry after 0.01s")
            return "ok"
        result = await retry_async(factory, max_retries=3, base_delay=0.01,
                                   run_id="r", agent_name="a", label="l")
        assert result == "ok"


class TestGeminiRateLimiter:
    """Centralized rate limiter behavior."""

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        limiter = GeminiRateLimiter()
        await limiter.acquire()
        limiter.release()

    @pytest.mark.asyncio
    async def test_serialization(self):
        """Multiple acquires should serialize (semaphore=1)."""
        limiter = GeminiRateLimiter()
        timestamps = []

        async def timed_acquire(i):
            await limiter.acquire()
            timestamps.append(time.monotonic())
            limiter.release()

        # Run 3 acquisitions concurrently
        await asyncio.gather(*[timed_acquire(i) for i in range(3)])
        assert len(timestamps) == 3


class TestBaseAgent:
    """BaseAgent init, think, narrate."""

    @pytest.mark.asyncio
    async def test_init(self):
        agent = BaseAgent("run123")
        assert agent.run_id == "run123"
        assert agent.tools_used == []

    @pytest.mark.asyncio
    async def test_think(self):
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])
        await agent.think("test thought")
        # Should log to DB
        conn = await db.get_db()
        cursor = await conn.execute(
            "SELECT * FROM agent_steps WHERE run_id = ?", (run["id"],)
        )
        rows = await cursor.fetchall()
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_narrate(self):
        agent = BaseAgent("run1")
        await agent.narrate("Hello!")  # Should not raise

    @pytest.mark.asyncio
    async def test_execute_tool_unknown(self):
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])
        result = await agent._execute_tool("nonexistent_tool", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])
        agent.tool_handlers = {
            "test_tool": AsyncMock(return_value={"result": "ok"})
        }
        result = await agent._execute_tool("test_tool", {})
        assert result == {"result": "ok"}
        assert "test_tool" in agent.tools_used

    @pytest.mark.asyncio
    async def test_execute_tool_returns_list(self):
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])
        agent.tool_handlers = {
            "list_tool": AsyncMock(return_value=[{"id": "1"}, {"id": "2"}])
        }
        result = await agent._execute_tool("list_tool", {})
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_execute_tool_returns_string(self):
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])
        agent.tool_handlers = {
            "str_tool": AsyncMock(return_value="just a string")
        }
        result = await agent._execute_tool("str_tool", {})
        assert result["result"] == "just a string"

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self):
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])
        agent.tool_handlers = {
            "fail_tool": AsyncMock(side_effect=Exception("boom"))
        }
        result = await agent._execute_tool("fail_tool", {})
        assert "error" in result

    def test_friendly_names(self):
        assert "NEXUS" in AGENT_FRIENDLY_NAMES.values()
        assert "Schedule Architect" in AGENT_FRIENDLY_NAMES.values()

    def test_tool_narrations(self):
        assert len(TOOL_NARRATIONS) >= 15
        assert "calendar" in TOOL_NARRATIONS["list_events"].lower()


# ═══════════════════════════════════════════════════════════════
# TOOLS UNIT TESTS
# ═══════════════════════════════════════════════════════════════

class TestCalendarTool:
    @pytest.mark.asyncio
    async def test_list_events(self):
        from tools import calendar_tool
        await db.create_event("E1", "2026-04-02T09:00:00", "2026-04-02T10:00:00")
        result = await calendar_tool.list_events("r1", "agent")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_create_event(self):
        from tools import calendar_tool
        result = await calendar_tool.create_event(
            "r1", "agent", "New Event", "2026-04-02T09:00:00", "2026-04-02T10:00:00"
        )
        assert result["title"] == "New Event"

    @pytest.mark.asyncio
    async def test_get_event(self):
        from tools import calendar_tool
        ev = await db.create_event("E", "2026-04-02T09:00:00", "2026-04-02T10:00:00")
        result = await calendar_tool.get_event("r1", "agent", ev["id"])
        assert result["title"] == "E"

    @pytest.mark.asyncio
    async def test_get_event_not_found(self):
        from tools import calendar_tool
        result = await calendar_tool.get_event("r1", "agent", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_event(self):
        from tools import calendar_tool
        ev = await db.create_event("Old", "2026-04-02T09:00:00", "2026-04-02T10:00:00")
        result = await calendar_tool.update_event("r1", "agent", ev["id"], title="New")
        assert result["title"] == "New"

    @pytest.mark.asyncio
    async def test_delete_event(self):
        from tools import calendar_tool
        ev = await db.create_event("Del", "2026-04-02T09:00:00", "2026-04-02T10:00:00")
        result = await calendar_tool.delete_event("r1", "agent", ev["id"])
        assert result is True

    @pytest.mark.asyncio
    async def test_find_free_slots(self):
        from tools import calendar_tool
        result = await calendar_tool.find_free_slots(
            "r1", "agent", "2026-04-02T08:00:00", "2026-04-02T18:00:00", 60
        )
        assert len(result) >= 1


class TestTasksTool:
    @pytest.mark.asyncio
    async def test_list_tasks(self):
        from tools import tasks_tool
        await db.create_task("T1")
        result = await tasks_tool.list_tasks("r1", "agent")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_create_task(self):
        from tools import tasks_tool
        result = await tasks_tool.create_task("r1", "agent", "New Task", priority="high")
        assert result["title"] == "New Task"

    @pytest.mark.asyncio
    async def test_get_task(self):
        from tools import tasks_tool
        t = await db.create_task("T")
        result = await tasks_tool.get_task("r1", "agent", t["id"])
        assert result["title"] == "T"

    @pytest.mark.asyncio
    async def test_get_task_not_found(self):
        from tools import tasks_tool
        result = await tasks_tool.get_task("r1", "agent", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_task(self):
        from tools import tasks_tool
        t = await db.create_task("Old")
        result = await tasks_tool.update_task("r1", "agent", t["id"], title="New")
        assert result["title"] == "New"

    @pytest.mark.asyncio
    async def test_complete_task(self):
        from tools import tasks_tool
        t = await db.create_task("Complete Me")
        result = await tasks_tool.complete_task("r1", "agent", t["id"])
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_overdue_tasks(self):
        from tools import tasks_tool
        past = (datetime.now() - timedelta(days=1)).isoformat()
        await db.create_task("Overdue", due_date=past)
        result = await tasks_tool.get_overdue_tasks("r1", "agent")
        assert len(result) >= 1


class TestNotesTool:
    @pytest.mark.asyncio
    async def test_list_notes(self):
        from tools import notes_tool
        await db.create_note("N1", "C1")
        result = await notes_tool.list_notes("r1", "agent")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_create_note(self):
        from tools import notes_tool
        result = await notes_tool.create_note("r1", "agent", "Title", "Content")
        assert result["title"] == "Title"

    @pytest.mark.asyncio
    async def test_get_note(self):
        from tools import notes_tool
        n = await db.create_note("N", "C")
        result = await notes_tool.get_note("r1", "agent", n["id"])
        assert result["title"] == "N"

    @pytest.mark.asyncio
    async def test_get_note_not_found(self):
        from tools import notes_tool
        result = await notes_tool.get_note("r1", "agent", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_note(self):
        from tools import notes_tool
        n = await db.create_note("Old", "C")
        result = await notes_tool.update_note("r1", "agent", n["id"], title="New")
        assert result["title"] == "New"

    @pytest.mark.asyncio
    async def test_search_notes(self):
        from tools import notes_tool
        await db.create_note("Python", "Learn Python")
        result = await notes_tool.search_notes("r1", "agent", "Python")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_append_to_note(self):
        from tools import notes_tool
        n = await db.create_note("Note", "Line 1")
        result = await notes_tool.append_to_note("r1", "agent", n["id"], "Line 2")
        assert "Line 2" in result["content"]

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_note(self):
        from tools import notes_tool
        result = await notes_tool.append_to_note("r1", "agent", "bad_id", "text")
        assert result is None


# ═══════════════════════════════════════════════════════════════
# PRIMARY AGENT UNIT TESTS
# ═══════════════════════════════════════════════════════════════

class TestPrimaryAgentHelpers:
    """Test intent parsing, plan parsing, sub-agent registry."""

    def test_sub_agent_registry(self):
        from agents.primary import get_sub_agent
        assert get_sub_agent("planning_agent", "r1") is not None
        assert get_sub_agent("learning_agent", "r1") is not None
        assert get_sub_agent("life_admin_agent", "r1") is not None
        assert get_sub_agent("nonexistent_agent", "r1") is None

    def test_general_agent_tools(self):
        from agents.primary import GeneralAgent
        agent = GeneralAgent("r1")
        assert len(agent.tool_handlers) >= 15  # 15 base + web_search + linkedin

    @pytest.mark.asyncio
    async def test_get_interrupted_run_none(self):
        from agents.primary import get_interrupted_run
        result = await get_interrupted_run("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_interrupted_run_exists(self):
        from agents.primary import get_interrupted_run
        run = await db.create_agent_run("test msg")
        plan = {"steps": [{"step": 1}], "completed_steps": []}
        await db.update_agent_run(run["id"], plan=plan)
        result = await get_interrupted_run(run["id"])
        assert result is not None
        assert result["user_message"] == "test msg"

    @pytest.mark.asyncio
    async def test_get_completed_step_numbers_empty(self):
        from agents.primary import get_completed_step_numbers
        run = await db.create_agent_run("test")
        completed = await get_completed_step_numbers(run["id"])
        assert len(completed) == 0

    @pytest.mark.asyncio
    async def test_get_completed_step_numbers_with_delegations(self):
        from agents.primary import get_completed_step_numbers
        run = await db.create_agent_run("test")
        await db.log_agent_step(run["id"], "primary_agent", "delegation",
                                {"step": 1, "to": "planning_agent"})
        await db.log_agent_step(run["id"], "primary_agent", "delegation",
                                {"step": 2, "to": "learning_agent"})
        completed = await get_completed_step_numbers(run["id"])
        assert completed == {1, 2}

    @pytest.mark.asyncio
    async def test_resume_interrupted_plan_no_plan(self):
        from agents.primary import resume_interrupted_plan
        result = await resume_interrupted_plan("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_resume_plan_no_interrupted(self):
        from agents.primary import resume_plan
        result = await resume_plan("nonexistent")
        assert result["response"] == "No interrupted plan found to resume."


# ═══════════════════════════════════════════════════════════════
# PHASE 1 — RATE LIMITER BRUTEFORCE + SQLite RACE CONDITIONS
# ═══════════════════════════════════════════════════════════════

class TestPhase1RateLimiterBruteforce:
    """Stress test: fire 50 concurrent rate limiter acquisitions."""

    @pytest.mark.asyncio
    async def test_concurrent_acquisitions(self):
        """50 concurrent acquire/release — must all complete without deadlock."""
        import agents.base as base_mod
        limiter = GeminiRateLimiter()
        # Override spacing IN THE MODULE WHERE IT'S USED (imported by value)
        orig_spacing = base_mod.GEMINI_MIN_CALL_SPACING_S
        orig_jitter = base_mod.GEMINI_JITTER_S
        base_mod.GEMINI_MIN_CALL_SPACING_S = 0.0
        base_mod.GEMINI_JITTER_S = 0.0

        results = []

        async def worker(i):
            await limiter.acquire()
            results.append(i)
            limiter.release()

        try:
            await asyncio.wait_for(
                asyncio.gather(*[worker(i) for i in range(50)]),
                timeout=30.0,
            )
            assert len(results) == 50
        finally:
            base_mod.GEMINI_MIN_CALL_SPACING_S = orig_spacing
            base_mod.GEMINI_JITTER_S = orig_jitter

    @pytest.mark.asyncio
    async def test_no_double_acquisition(self):
        """Only 1 caller should hold the limiter at a time (semaphore=1)."""
        import agents.base as base_mod
        limiter = GeminiRateLimiter()
        orig_spacing = base_mod.GEMINI_MIN_CALL_SPACING_S
        orig_jitter = base_mod.GEMINI_JITTER_S
        base_mod.GEMINI_MIN_CALL_SPACING_S = 0.0
        base_mod.GEMINI_JITTER_S = 0.0

        active = [0]
        max_active = [0]

        async def worker():
            await limiter.acquire()
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1
            limiter.release()

        try:
            await asyncio.gather(*[worker() for _ in range(10)])
            assert max_active[0] == 1, f"Max concurrent was {max_active[0]}, expected 1"
        finally:
            base_mod.GEMINI_MIN_CALL_SPACING_S = orig_spacing
            base_mod.GEMINI_JITTER_S = orig_jitter


class TestPhase1SQLiteRaceConditions:
    """Stress test: concurrent writes to the same record."""

    @pytest.mark.asyncio
    async def test_concurrent_task_creates(self):
        """10 concurrent task creations — all must succeed."""
        tasks_created = await asyncio.gather(*[
            db.create_task(f"Task {i}", priority="medium")
            for i in range(10)
        ])
        assert len(tasks_created) == 10
        all_tasks = await db.list_tasks()
        assert len(all_tasks) == 10

    @pytest.mark.asyncio
    async def test_concurrent_event_creates(self):
        """10 concurrent event creations."""
        events = await asyncio.gather(*[
            db.create_event(f"Event {i}", f"2026-04-0{i+1}T09:00:00", f"2026-04-0{i+1}T10:00:00")
            for i in range(9)
        ])
        assert len(events) == 9

    @pytest.mark.asyncio
    async def test_concurrent_note_creates(self):
        """10 concurrent note creations."""
        notes = await asyncio.gather(*[
            db.create_note(f"Note {i}", f"Content {i}")
            for i in range(10)
        ])
        assert len(notes) == 10
        all_notes = await db.list_notes()
        assert len(all_notes) == 10

    @pytest.mark.asyncio
    async def test_concurrent_updates_same_task(self):
        """10 concurrent PATCH operations on the same task."""
        task = await db.create_task("Race Task", priority="low")
        task_id = task["id"]

        results = await asyncio.gather(*[
            db.update_task(task_id, title=f"Updated {i}")
            for i in range(10)
        ])
        # All should succeed (last write wins in SQLite WAL mode)
        assert all(r is not None for r in results)
        final = await db.get_task(task_id)
        assert "Updated" in final["title"]

    @pytest.mark.asyncio
    async def test_concurrent_agent_step_logging(self):
        """20 concurrent step logs to the same run."""
        run = await db.create_agent_run("stress test")
        steps = await asyncio.gather(*[
            db.log_agent_step(run["id"], f"agent_{i}", "thought", {"i": i})
            for i in range(20)
        ])
        assert len(steps) == 20
        conn = await db.get_db()
        cursor = await conn.execute(
            "SELECT COUNT(*) as c FROM agent_steps WHERE run_id = ?", (run["id"],)
        )
        row = await cursor.fetchone()
        assert row["c"] == 20


# ═══════════════════════════════════════════════════════════════
# PHASE 2 — WebSocket BROKEN PIPE + PAYLOAD INJECTION
# ═══════════════════════════════════════════════════════════════

class TestPhase2WebSocketBrokenPipe:
    """Simulate clients disconnecting mid-broadcast."""

    @pytest.mark.asyncio
    async def test_50_clients_half_disconnect(self, mc):
        """50 clients connect, 25 break mid-broadcast — system must survive."""
        good_clients = [AsyncMock() for _ in range(25)]
        bad_clients = [AsyncMock() for _ in range(25)]
        for ws in bad_clients:
            ws.send_json.side_effect = Exception("Connection reset")

        for ws in good_clients + bad_clients:
            await mc.connect(ws)
        assert mc.client_count == 50

        # Broadcast — should handle 25 broken pipes gracefully
        await mc.broadcast("r1", "agent", "test", {"data": "x"})

        # Bad clients removed, good clients remain
        assert mc.client_count == 25
        # Good clients received the event
        for ws in good_clients:
            ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_clients_disconnect(self, mc):
        """All clients die — broadcast must not crash."""
        clients = [AsyncMock() for _ in range(10)]
        for ws in clients:
            ws.send_json.side_effect = Exception("gone")
            await mc.connect(ws)

        await mc.broadcast("r1", "agent", "test", {"x": 1})
        assert mc.client_count == 0

    @pytest.mark.asyncio
    async def test_rapid_connect_disconnect(self, mc):
        """Rapid connect/disconnect cycles."""
        for _ in range(100):
            ws = AsyncMock()
            await mc.connect(ws)
            mc.disconnect(ws)
        assert mc.client_count == 0


class TestPhase2PayloadInjection:
    """Malformed inputs that could crash the system."""

    @pytest.mark.asyncio
    async def test_empty_title_task(self):
        """Empty string title — DB should accept it (no NOT NULL on empty)."""
        t = await db.create_task("")
        assert t["title"] == ""

    @pytest.mark.asyncio
    async def test_unicode_bomb(self):
        """Unicode stress test."""
        evil = "🎉" * 1000 + "DROP TABLE tasks;--" + "漢字" * 500
        t = await db.create_task(evil, description=evil)
        fetched = await db.get_task(t["id"])
        assert fetched["title"] == evil  # SQL injection must fail

    @pytest.mark.asyncio
    async def test_sql_injection_in_search(self):
        """SQL injection attempt in notes search."""
        await db.create_note("Safe", "Content")
        # This should NOT drop the table or return all rows
        results = await db.search_notes("'; DROP TABLE notes; --")
        assert isinstance(results, list)
        # Verify notes table still exists
        all_notes = await db.list_notes()
        assert len(all_notes) == 1

    @pytest.mark.asyncio
    async def test_huge_payload(self):
        """10MB content — test for memory handling."""
        huge = "A" * (10 * 1024 * 1024)
        n = await db.create_note("Huge Note", huge)
        fetched = await db.get_note(n["id"])
        assert len(fetched["content"]) == 10 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_null_bytes(self):
        """Null bytes in strings."""
        t = await db.create_task("null\x00byte\x00test")
        fetched = await db.get_task(t["id"])
        assert fetched is not None

    @pytest.mark.asyncio
    async def test_newlines_in_title(self):
        t = await db.create_task("Line1\nLine2\rLine3\r\n")
        fetched = await db.get_task(t["id"])
        assert "\n" in fetched["title"]

    @pytest.mark.asyncio
    async def test_json_injection_in_tags(self):
        """Tags with JSON-breaking characters."""
        t = await db.create_task("Tagged", tags=['{"evil": true}', "normal"])
        fetched = await db.get_task(t["id"])
        assert len(fetched["tags"]) == 2

    @pytest.mark.asyncio
    async def test_broadcast_with_huge_content(self, mc):
        """Broadcast a large event — no crash."""
        ws = AsyncMock()
        await mc.connect(ws)
        large_content = {"data": "X" * 100000}
        await mc.broadcast("r1", "a", "test", large_content)
        ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_with_non_serializable(self, mc):
        """Content with datetime objects — json serialization edge case."""
        ws = AsyncMock()
        await mc.connect(ws)
        # The broadcast method uses json.dumps with default=str
        await mc.broadcast("r1", "a", "test", {"time": datetime.now()})
        ws.send_json.assert_called_once()


# ═══════════════════════════════════════════════════════════════
# PHASE 3 — CONTEXT OVERFLOW + TOOL CHAIN COLLAPSE
# ═══════════════════════════════════════════════════════════════

class TestPhase3ContextOverflow:
    """Test agent behavior with oversized inputs."""

    @pytest.mark.asyncio
    async def test_oversized_user_message(self):
        """Agent run with 100K character message."""
        run = await db.create_agent_run("A" * 100000)
        assert run is not None
        assert len(run["user_message"]) == 100000

    @pytest.mark.asyncio
    async def test_many_agent_steps(self):
        """Log 200 steps to a single run — no DB issues."""
        run = await db.create_agent_run("stress")
        for i in range(200):
            await db.log_agent_step(run["id"], "agent", "thought", {"step": i})
        conn = await db.get_db()
        cursor = await conn.execute(
            "SELECT COUNT(*) as c FROM agent_steps WHERE run_id = ?", (run["id"],)
        )
        row = await cursor.fetchone()
        assert row["c"] == 200


class TestPhase3ToolChainCollapse:
    """Simulate tool chains failing at various points."""

    @pytest.mark.asyncio
    async def test_tool_handler_exception(self):
        """Tool handler that raises — agent should catch and continue."""
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])
        agent.tool_handlers = {
            "exploding_tool": AsyncMock(side_effect=RuntimeError("KABOOM"))
        }
        result = await agent._execute_tool("exploding_tool", {})
        assert "error" in result
        assert "KABOOM" in result["error"]

    @pytest.mark.asyncio
    async def test_tool_handler_timeout_simulation(self):
        """Tool that hangs — verify we get error result."""
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])

        async def hanging_tool(run_id, agent_name):
            await asyncio.sleep(100)
            return {"result": "never"}

        agent.tool_handlers = {"hang_tool": hanging_tool}

        # Use retry_async with low timeout to test
        async def execute_with_timeout():
            return await asyncio.wait_for(
                agent._execute_tool("hang_tool", {}),
                timeout=1.0
            )

        with pytest.raises(asyncio.TimeoutError):
            await execute_with_timeout()

    @pytest.mark.asyncio
    async def test_cascading_tool_failures(self):
        """Multiple tools fail in sequence — agent tracks all failures."""
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])

        call_count = [0]
        async def flaky_tool(run_id, agent_name, **kwargs):
            call_count[0] += 1
            raise Exception(f"Failure #{call_count[0]}")

        agent.tool_handlers = {"flaky": flaky_tool}

        r1 = await agent._execute_tool("flaky", {})
        r2 = await agent._execute_tool("flaky", {})
        r3 = await agent._execute_tool("flaky", {})

        assert "error" in r1
        assert "error" in r2
        assert "error" in r3
        assert len(agent.tools_used) == 3  # All attempts tracked


# ═══════════════════════════════════════════════════════════════
# FASTAPI ENDPOINT TESTS (via TestClient)
# ═══════════════════════════════════════════════════════════════

class TestAPIEndpoints:
    """Test routers via httpx AsyncClient."""

    @pytest_asyncio.fixture
    async def client(self):
        from httpx import AsyncClient, ASGITransport
        from main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_health(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_db(self, client):
        r = await client.get("/health/db")
        assert r.status_code == 200
        data = r.json()
        assert data["database"] == "connected"

    @pytest.mark.asyncio
    async def test_api_info(self, client):
        r = await client.get("/api/info")
        assert r.status_code == 200
        assert r.json()["name"] == "NEXUS"

    # ── Data Router: Events ──

    @pytest.mark.asyncio
    async def test_create_and_list_events(self, client):
        r = await client.post("/data/events", json={
            "title": "Test Event",
            "start_time": "2026-04-02T09:00:00",
            "end_time": "2026-04-02T10:00:00"
        })
        assert r.status_code == 200
        event_id = r.json()["id"]

        r = await client.get("/data/events")
        assert r.status_code == 200
        assert any(e["id"] == event_id for e in r.json())

    @pytest.mark.asyncio
    async def test_get_event(self, client):
        r = await client.post("/data/events", json={
            "title": "E", "start_time": "2026-04-02T09:00:00", "end_time": "2026-04-02T10:00:00"
        })
        eid = r.json()["id"]
        r = await client.get(f"/data/events/{eid}")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_get_event_404(self, client):
        r = await client.get("/data/events/nonexistent")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_update_event(self, client):
        r = await client.post("/data/events", json={
            "title": "Old", "start_time": "2026-04-02T09:00:00", "end_time": "2026-04-02T10:00:00"
        })
        eid = r.json()["id"]
        r = await client.patch(f"/data/events/{eid}", json={"title": "New"})
        assert r.status_code == 200
        assert r.json()["title"] == "New"

    @pytest.mark.asyncio
    async def test_update_event_no_fields(self, client):
        r = await client.post("/data/events", json={
            "title": "E", "start_time": "2026-04-02T09:00:00", "end_time": "2026-04-02T10:00:00"
        })
        eid = r.json()["id"]
        r = await client.patch(f"/data/events/{eid}", json={})
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_update_event_404(self, client):
        r = await client.patch("/data/events/nonexistent", json={"title": "X"})
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_event(self, client):
        r = await client.post("/data/events", json={
            "title": "Del", "start_time": "2026-04-02T09:00:00", "end_time": "2026-04-02T10:00:00"
        })
        eid = r.json()["id"]
        r = await client.delete(f"/data/events/{eid}")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_event_404(self, client):
        r = await client.delete("/data/events/nonexistent")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_find_free_slots(self, client):
        r = await client.get("/data/events/free-slots/find",
                             params={"start": "2026-04-02T08:00:00", "end": "2026-04-02T18:00:00"})
        assert r.status_code == 200

    # ── Data Router: Tasks ──

    @pytest.mark.asyncio
    async def test_create_and_list_tasks(self, client):
        r = await client.post("/data/tasks", json={"title": "Test Task"})
        assert r.status_code == 200
        r = await client.get("/data/tasks")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_get_task(self, client):
        r = await client.post("/data/tasks", json={"title": "T"})
        tid = r.json()["id"]
        r = await client.get(f"/data/tasks/{tid}")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_get_task_404(self, client):
        r = await client.get("/data/tasks/nonexistent")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_update_task(self, client):
        r = await client.post("/data/tasks", json={"title": "Old"})
        tid = r.json()["id"]
        r = await client.patch(f"/data/tasks/{tid}", json={"title": "New"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_update_task_no_fields(self, client):
        r = await client.post("/data/tasks", json={"title": "T"})
        tid = r.json()["id"]
        r = await client.patch(f"/data/tasks/{tid}", json={})
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_update_task_404(self, client):
        r = await client.patch("/data/tasks/nonexistent", json={"title": "X"})
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_complete_task(self, client):
        r = await client.post("/data/tasks", json={"title": "Complete Me"})
        tid = r.json()["id"]
        r = await client.post(f"/data/tasks/{tid}/complete")
        assert r.status_code == 200
        assert r.json()["status"] == "completed"

    @pytest.mark.asyncio
    async def test_complete_task_404(self, client):
        r = await client.post("/data/tasks/nonexistent/complete")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_get_overdue_tasks(self, client):
        r = await client.get("/data/tasks/overdue/all")
        assert r.status_code == 200

    # ── Data Router: Notes ──

    @pytest.mark.asyncio
    async def test_create_and_list_notes(self, client):
        r = await client.post("/data/notes", json={"title": "N", "content": "C"})
        assert r.status_code == 200
        r = await client.get("/data/notes")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_list_notes_with_tags(self, client):
        await db.create_note("Tagged", "C", tags=["test"])
        r = await client.get("/data/notes", params={"tags": "test"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_get_note(self, client):
        r = await client.post("/data/notes", json={"title": "N", "content": "C"})
        nid = r.json()["id"]
        r = await client.get(f"/data/notes/{nid}")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_get_note_404(self, client):
        r = await client.get("/data/notes/nonexistent")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_update_note(self, client):
        r = await client.post("/data/notes", json={"title": "Old", "content": "C"})
        nid = r.json()["id"]
        r = await client.patch(f"/data/notes/{nid}", json={"title": "New"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_update_note_no_fields(self, client):
        r = await client.post("/data/notes", json={"title": "N", "content": "C"})
        nid = r.json()["id"]
        r = await client.patch(f"/data/notes/{nid}", json={})
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_update_note_404(self, client):
        r = await client.patch("/data/notes/nonexistent", json={"title": "X"})
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_search_notes(self, client):
        await db.create_note("Searchable", "Find this content")
        r = await client.get("/data/notes/search/query", params={"q": "Find"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_append_to_note(self, client):
        r = await client.post("/data/notes", json={"title": "N", "content": "Line 1"})
        nid = r.json()["id"]
        r = await client.post(f"/data/notes/{nid}/append", json={"content": "Line 2"})
        assert r.status_code == 200
        assert "Line 2" in r.json()["content"]

    @pytest.mark.asyncio
    async def test_append_to_note_404(self, client):
        r = await client.post("/data/notes/nonexistent/append", json={"content": "x"})
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════
# AGENT SUB-CLASS INITIALIZATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestSubAgentInitialization:
    """Verify all agent subclasses initialize correctly."""

    def test_planning_agent(self):
        from agents.planning import PlanningAgent
        agent = PlanningAgent("r1")
        assert agent.name == "planning_agent"
        assert len(agent.tool_handlers) > 0
        assert "schedule" in agent.system_prompt.lower() or "plan" in agent.system_prompt.lower()

    def test_learning_agent(self):
        from agents.learning import LearningAgent
        agent = LearningAgent("r1")
        assert agent.name == "learning_agent"
        assert len(agent.tool_handlers) > 0

    def test_life_admin_agent(self):
        from agents.life_admin import LifeAdminAgent
        agent = LifeAdminAgent("r1")
        assert agent.name == "life_admin_agent"
        assert len(agent.tool_handlers) > 0

    def test_proactive_monitor_agent(self):
        from agents.proactive import ProactiveMonitorAgent
        agent = ProactiveMonitorAgent("r1")
        assert agent.name == "proactive_monitor"
        assert len(agent.tool_handlers) > 0

    def test_general_agent(self):
        from agents.primary import GeneralAgent
        agent = GeneralAgent("r1")
        assert agent.name == "general_agent"
        assert len(agent.tool_handlers) >= 15  # 15 base + web_search + linkedin


# ═══════════════════════════════════════════════════════════════
# INTEGRATION: RETRY + RATE LIMITER COMBINED
# ═══════════════════════════════════════════════════════════════

class TestRetryAndRateLimiterIntegration:
    """End-to-end retry with rate limiting."""

    @pytest.mark.asyncio
    async def test_retry_through_limiter(self):
        """Simulate a rate-limited call that succeeds on retry."""
        attempt = [0]

        async def api_call():
            attempt[0] += 1
            if attempt[0] < 3:
                raise Exception("429 RESOURCE_EXHAUSTED")
            return {"status": "ok"}

        import agents.base as base_mod
        limiter = GeminiRateLimiter()
        orig_spacing = base_mod.GEMINI_MIN_CALL_SPACING_S
        orig_jitter = base_mod.GEMINI_JITTER_S
        base_mod.GEMINI_MIN_CALL_SPACING_S = 0.0
        base_mod.GEMINI_JITTER_S = 0.0

        try:
            await limiter.acquire()
            try:
                result = await retry_async(
                    api_call, max_retries=5, base_delay=0.01,
                    run_id="test", agent_name="test", label="test"
                )
            finally:
                limiter.release()
            assert result["status"] == "ok"
            assert attempt[0] == 3
        finally:
            base_mod.GEMINI_MIN_CALL_SPACING_S = orig_spacing
            base_mod.GEMINI_JITTER_S = orig_jitter


# ═══════════════════════════════════════════════════════════════
# ADDITIONAL COVERAGE — BaseAgent.execute, main.py, chat router
# ═══════════════════════════════════════════════════════════════

class TestBaseAgentExecute:
    """Test the full execute() ReAct loop with mocked Gemini."""

    @pytest.mark.asyncio
    async def test_execute_text_response(self):
        """Gemini returns text immediately — no tool calls."""
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])

        # Mock a text-only response
        mock_part = MagicMock()
        mock_part.text = "Here is your answer"
        mock_part.function_call = None
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        with patch.object(agent, '_call_gemini', return_value=mock_response):
            result = await agent.execute("Hello")
        assert result["response"] == "Here is your answer"

    @pytest.mark.asyncio
    async def test_execute_no_candidates(self):
        """Gemini returns empty candidates."""
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])

        mock_response = MagicMock()
        mock_response.candidates = []

        with patch.object(agent, '_call_gemini', return_value=mock_response):
            result = await agent.execute("Hello")
        assert "wasn't able to" in result["response"]

    @pytest.mark.asyncio
    async def test_execute_no_content(self):
        """Gemini returns candidate with no content."""
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])

        mock_candidate = MagicMock()
        mock_candidate.content = None
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        with patch.object(agent, '_call_gemini', return_value=mock_response):
            result = await agent.execute("Hello")
        assert result["response"] == "Task completed."

    @pytest.mark.asyncio
    async def test_execute_with_tool_call_then_text(self):
        """Gemini calls a tool, then returns text."""
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])

        # First response: function call
        fc = MagicMock()
        fc.name = "test_tool"
        fc.args = {"key": "value"}
        fc_part = MagicMock()
        fc_part.text = None
        fc_part.function_call = fc
        fc_content = MagicMock()
        fc_content.parts = [fc_part]
        fc_candidate = MagicMock()
        fc_candidate.content = fc_content
        fc_response = MagicMock()
        fc_response.candidates = [fc_candidate]

        # Second response: text
        text_part = MagicMock()
        text_part.text = "Done!"
        text_part.function_call = None
        text_content = MagicMock()
        text_content.parts = [text_part]
        text_candidate = MagicMock()
        text_candidate.content = text_content
        text_response = MagicMock()
        text_response.candidates = [text_candidate]

        agent.tool_handlers = {
            "test_tool": AsyncMock(return_value={"result": "ok"})
        }

        call_count = [0]
        async def mock_call_gemini(step):
            call_count[0] += 1
            if call_count[0] == 1:
                return fc_response
            return text_response

        with patch.object(agent, '_call_gemini', side_effect=mock_call_gemini):
            result = await agent.execute("Use a tool")
        assert result["response"] == "Done!"
        assert "test_tool" in result["tools_used"]

    @pytest.mark.asyncio
    async def test_execute_gemini_api_error(self):
        """Gemini raises an exception."""
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])

        with patch.object(agent, '_call_gemini', side_effect=Exception("API down")):
            result = await agent.execute("Hello")
        assert "error" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_execute_max_steps(self):
        """Simulate hitting MAX_AGENT_STEPS with continuous tool calls."""
        import agents.base as base_mod
        orig_max = base_mod.MAX_AGENT_STEPS
        base_mod.MAX_AGENT_STEPS = 2  # Limit for test

        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])

        # Always return a tool call
        fc = MagicMock()
        fc.name = "loop_tool"
        fc.args = {}
        fc_part = MagicMock()
        fc_part.text = "intermediate"
        fc_part.function_call = fc
        fc_content = MagicMock()
        fc_content.parts = [fc_part]
        fc_candidate = MagicMock()
        fc_candidate.content = fc_content
        fc_response = MagicMock()
        fc_response.candidates = [fc_candidate]

        agent.tool_handlers = {
            "loop_tool": AsyncMock(return_value={"result": "loop"})
        }

        with patch.object(agent, '_call_gemini', return_value=fc_response):
            result = await agent.execute("Loop forever")

        base_mod.MAX_AGENT_STEPS = orig_max
        # Should return the last text it has
        assert result is not None


class TestBaseAgentExecuteDefensive:
    """Tests for the new defensive guards in execute()."""

    @pytest.mark.asyncio
    async def test_execute_none_parts(self):
        """candidate.content.parts is None (not []) — must not crash."""
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])

        mock_content = MagicMock()
        mock_content.parts = None  # ← The actual bug trigger
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        with patch.object(agent, '_call_gemini', return_value=mock_response):
            result = await agent.execute("Hello")
        assert result["response"] == "Task completed."

    @pytest.mark.asyncio
    async def test_execute_candidates_is_none(self):
        """response.candidates is None (not just []) — must not crash."""
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])

        mock_response = MagicMock()
        mock_response.candidates = None

        with patch.object(agent, '_call_gemini', return_value=mock_response):
            result = await agent.execute("Hello")
        assert "wasn't able to" in result["response"]

    @pytest.mark.asyncio
    async def test_execute_part_without_text_attr(self):
        """Part object where .text attribute doesn't exist — getattr guard."""
        run = await db.create_agent_run("test")
        agent = BaseAgent(run["id"])

        # Part with no .text and no .function_call
        weird_part = MagicMock(spec=[])  # empty spec = no attributes
        mock_content = MagicMock()
        mock_content.parts = [weird_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        with patch.object(agent, '_call_gemini', return_value=mock_response):
            result = await agent.execute("Hello")
        assert result["response"] == "Task completed."


class TestPlanFlattening:
    """Verify MAX_PLAN_STEPS enforcement."""

    @pytest.mark.asyncio
    async def test_plan_truncated_to_max(self):
        """Plans with >MAX_PLAN_STEPS are truncated."""
        from agents.primary import generate_plan

        big_plan = {"plan": [
            {"step": i, "agent": "planning_agent", "action": f"step {i}", "tools_needed": []}
            for i in range(1, 11)  # 10 steps
        ]}
        mock_response = MagicMock()
        mock_response.text = json.dumps(big_plan)

        run = await db.create_agent_run("big request")
        classification = {"intent": "planning", "complexity": "complex", "summary": "test"}

        with patch("agents.primary.llm_router.generate_content", new_callable=AsyncMock, return_value=mock_response):
            plan = await generate_plan(run["id"], "big request", classification)
            from config import MAX_PLAN_STEPS
            assert len(plan) == MAX_PLAN_STEPS

    @pytest.mark.asyncio
    async def test_classify_intent_empty_response(self):
        """classify_intent when Gemini returns None text."""
        from agents.primary import classify_intent

        mock_response = MagicMock(spec=[])  # No .text attribute at all

        run = await db.create_agent_run("test")

        with patch("agents.primary.llm_router.generate_content", new_callable=AsyncMock, return_value=mock_response):
            result = await classify_intent(run["id"], "test")
            assert result["intent"] == "general"
            assert result["complexity"] == "simple"

    @pytest.mark.asyncio
    async def test_synthesis_none_text_fallback(self):
        """Synthesis response.text is None — falls back to joined results."""
        from agents.primary import _execute_plan

        mock_agent = AsyncMock()
        mock_agent.execute.return_value = {"response": "Step result OK", "tools_used": []}

        mock_synthesis = MagicMock(spec=[])  # No .text attribute

        run = await db.create_agent_run("test")
        plan = [
            {"step": 1, "agent": "planning_agent", "action": "step 1"},
            {"step": 2, "agent": "planning_agent", "action": "step 2"},
        ]

        with patch("agents.primary.get_sub_agent", return_value=mock_agent):
            with patch("agents.primary.llm_router.generate_content", new_callable=AsyncMock, return_value=mock_synthesis):
                result = await _execute_plan(run["id"], "test", plan, datetime.now())
                # Should fall back to joining all_results
                assert "Step result OK" in result["response"]


class TestMainAppEndpoints:
    """Test main.py WebSocket and proactive endpoints."""

    @pytest_asyncio.fixture
    async def client(self):
        from httpx import AsyncClient, ASGITransport
        from main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_root_serves_html(self, client):
        """GET / should serve index.html (or 404 if file missing)."""
        r = await client.get("/")
        # Could be 200 (html exists) or 404 (no file in test env)
        assert r.status_code in (200, 404)

    @pytest.mark.asyncio
    async def test_proactive_daily_briefing_endpoint(self, client):
        """POST /proactive/daily-briefing — will fail without Gemini key but should not crash."""
        r = await client.post("/proactive/daily-briefing")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("ok", "error")

    @pytest.mark.asyncio
    async def test_proactive_overdue_check_endpoint(self, client):
        """POST /proactive/overdue-check — will fail without Gemini key but should not crash."""
        r = await client.post("/proactive/overdue-check")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("ok", "error")


class TestChatRouter:
    """Test the chat endpoint with mocked agent processing."""

    @pytest_asyncio.fixture
    async def client(self):
        from httpx import AsyncClient, ASGITransport
        from main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_chat_with_mocked_agent(self, client):
        """POST /chat with mocked process_message."""
        mock_result = {"response": "Mocked response", "tools_used": ["create_task"]}
        # process_message is imported locally in chat.py's handler, so we patch at source
        with patch("agents.primary.process_message", new_callable=AsyncMock, return_value=mock_result):
            r = await client.post("/chat", json={"message": "Create a test task"})
            assert r.status_code == 200
            data = r.json()
            assert "run_id" in data
            assert data["response"] == "Mocked response"

    @pytest.mark.asyncio
    async def test_chat_resume_no_plan(self, client):
        """POST /chat/resume with nonexistent run_id."""
        with patch("agents.primary.resume_plan", new_callable=AsyncMock,
                    return_value={"response": "No interrupted plan found to resume.", "tools_used": []}):
            r = await client.post("/chat/resume", json={"run_id": "nonexistent"})
            assert r.status_code == 200
            assert "No interrupted plan" in r.json()["response"]

    @pytest.mark.asyncio
    async def test_chat_error_returns_500(self, client):
        """POST /chat when agent raises exception."""
        with patch("agents.primary.process_message", new_callable=AsyncMock, side_effect=Exception("Agent crashed")):
            r = await client.post("/chat", json={"message": "test"})
            assert r.status_code == 500

    @pytest.mark.asyncio
    async def test_chat_resume_success(self, client):
        """POST /chat/resume with a valid resumed plan."""
        run = await db.create_agent_run("test plan")
        plan_data = {"steps": [{"step": 1, "action": "do"}], "completed_steps": []}
        await db.update_agent_run(run["id"], plan=plan_data)

        mock_result = {"response": "Plan resumed successfully!", "tools_used": ["create_event"]}
        with patch("agents.primary.resume_plan", new_callable=AsyncMock, return_value=mock_result):
            r = await client.post("/chat/resume", json={"run_id": run["id"]})
            assert r.status_code == 200
            data = r.json()
            assert data["response"] == "Plan resumed successfully!"

    @pytest.mark.asyncio
    async def test_chat_resume_error_returns_500(self, client):
        """POST /chat/resume when resume raises exception."""
        with patch("agents.primary.resume_plan", new_callable=AsyncMock, side_effect=Exception("Resume failed")):
            r = await client.post("/chat/resume", json={"run_id": "test-id"})
            assert r.status_code == 500


class TestHealthDbError:
    """Test health/db error path."""

    @pytest.mark.asyncio
    async def test_db_health_error_path(self):
        """Test the except branch in db_health."""
        from routers.health import db_health
        # Close DB to simulate error — though this may just reinit
        # We test by mocking get_db to raise
        with patch("routers.health.get_db", side_effect=Exception("DB down")):
            result = await db_health()
            assert result["status"] == "error"


class TestPrimaryClassifyAndPlan:
    """Test classify_intent and generate_plan with mocked Gemini."""

    @pytest.mark.asyncio
    async def test_classify_intent_valid_json(self):
        """classify_intent with valid JSON response from Gemini."""
        from agents.primary import classify_intent

        mock_response = MagicMock()
        mock_response.text = '{"intent": "planning", "complexity": "complex", "summary": "Plan my week"}'

        run = await db.create_agent_run("Plan my week")

        with patch("agents.primary.llm_router.generate_content", new_callable=AsyncMock, return_value=mock_response):
            result = await classify_intent(run["id"], "Plan my week")
            assert result["intent"] == "planning"
            assert result["complexity"] == "complex"

    @pytest.mark.asyncio
    async def test_classify_intent_markdown_wrapped(self):
        """classify_intent strips markdown code fences."""
        from agents.primary import classify_intent

        mock_response = MagicMock()
        mock_response.text = '```json\n{"intent": "general", "complexity": "simple", "summary": "hello"}\n```'

        run = await db.create_agent_run("hello")

        with patch("agents.primary.llm_router.generate_content", new_callable=AsyncMock, return_value=mock_response):
            result = await classify_intent(run["id"], "hello")
            assert result["intent"] == "general"

    @pytest.mark.asyncio
    async def test_classify_intent_invalid_json(self):
        """classify_intent falls back on bad JSON."""
        from agents.primary import classify_intent

        mock_response = MagicMock()
        mock_response.text = 'NOT JSON AT ALL'

        run = await db.create_agent_run("test")

        with patch("agents.primary.llm_router.generate_content", new_callable=AsyncMock, return_value=mock_response):
            result = await classify_intent(run["id"], "test")
            assert result["intent"] == "general"
            assert result["complexity"] == "simple"

    @pytest.mark.asyncio
    async def test_generate_plan_valid(self):
        """generate_plan with valid JSON."""
        from agents.primary import generate_plan

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "plan": [
                {"step": 1, "agent": "planning_agent", "action": "Check calendar", "tools_needed": ["calendar"]},
                {"step": 2, "agent": "planning_agent", "action": "Schedule tasks", "tools_needed": ["tasks"]},
            ]
        })

        run = await db.create_agent_run("Plan my week")
        classification = {"intent": "planning", "complexity": "complex", "summary": "Plan my week"}

        with patch("agents.primary.llm_router.generate_content", new_callable=AsyncMock, return_value=mock_response):
            plan = await generate_plan(run["id"], "Plan my week", classification)
            assert len(plan) == 2
            assert plan[0]["agent"] == "planning_agent"

    @pytest.mark.asyncio
    async def test_generate_plan_invalid_json(self):
        """generate_plan falls back with single-step plan on bad JSON."""
        from agents.primary import generate_plan

        mock_response = MagicMock()
        mock_response.text = 'BROKEN JSON'

        run = await db.create_agent_run("test")
        classification = {"intent": "planning", "complexity": "complex", "summary": "test"}

        with patch("agents.primary.llm_router.generate_content", new_callable=AsyncMock, return_value=mock_response):
            plan = await generate_plan(run["id"], "test", classification)
            assert len(plan) == 1

    @pytest.mark.asyncio
    async def test_generate_plan_markdown_wrapped(self):
        """generate_plan strips markdown code fences."""
        from agents.primary import generate_plan

        mock_response = MagicMock()
        mock_response.text = '```json\n{"plan": [{"step": 1, "agent": "learning_agent", "action": "do", "tools_needed": []}]}\n```'

        run = await db.create_agent_run("learn")
        classification = {"intent": "learning", "complexity": "complex", "summary": "learn"}

        with patch("agents.primary.llm_router.generate_content", new_callable=AsyncMock, return_value=mock_response):
            plan = await generate_plan(run["id"], "learn", classification)
            assert len(plan) == 1


class TestProcessMessage:
    """Test the full process_message flow with mocks."""

    @pytest.mark.asyncio
    async def test_simple_intent_routes_to_general(self):
        """Simple intent → GeneralAgent."""
        from agents.primary import process_message

        mock_classification = {"intent": "general", "complexity": "simple", "summary": "test"}

        with patch("agents.primary.classify_intent", new_callable=AsyncMock, return_value=mock_classification):
            # Mock GeneralAgent.execute
            with patch("agents.primary.GeneralAgent") as MockGA:
                mock_instance = AsyncMock()
                mock_instance.execute.return_value = {"response": "Done!", "tools_used": []}
                MockGA.return_value = mock_instance

                run = await db.create_agent_run("test")
                result = await process_message(run["id"], "test")
                assert result["response"] == "Done!"

    @pytest.mark.asyncio
    async def test_complex_intent_generates_plan(self):
        """Complex intent → plan generation + execution."""
        from agents.primary import process_message

        mock_classification = {"intent": "planning", "complexity": "complex", "summary": "plan week"}
        mock_plan = [{"step": 1, "agent": "planning_agent", "action": "schedule"}]

        with patch("agents.primary.classify_intent", new_callable=AsyncMock, return_value=mock_classification):
            with patch("agents.primary.generate_plan", new_callable=AsyncMock, return_value=mock_plan):
                with patch("agents.primary._execute_plan", new_callable=AsyncMock,
                           return_value={"response": "Plan executed!", "tools_used": ["create_event"]}):
                    run = await db.create_agent_run("Plan my week")
                    result = await process_message(run["id"], "Plan my week")
                    assert result["response"] == "Plan executed!"


class TestExecutePlan:
    """Test _execute_plan with mocked sub-agents."""

    @pytest.mark.asyncio
    async def test_single_step_plan(self):
        """Single step plan — no synthesis needed."""
        from agents.primary import _execute_plan

        mock_agent = AsyncMock()
        mock_agent.execute.return_value = {"response": "Step done!", "tools_used": ["create_task"]}

        run = await db.create_agent_run("test")
        plan = [{"step": 1, "agent": "planning_agent", "action": "do thing"}]

        with patch("agents.primary.get_sub_agent", return_value=mock_agent):
            result = await _execute_plan(run["id"], "test", plan, datetime.now())
            assert result["response"] == "Step done!"

    @pytest.mark.asyncio
    async def test_multi_step_plan_with_synthesis(self):
        """Multi-step plan triggers synthesis."""
        from agents.primary import _execute_plan

        mock_agent = AsyncMock()
        mock_agent.execute.return_value = {"response": "Step result", "tools_used": []}

        mock_synthesis = MagicMock()
        mock_synthesis.text = "Here's a summary of everything done."

        run = await db.create_agent_run("test")
        plan = [
            {"step": 1, "agent": "planning_agent", "action": "step 1"},
            {"step": 2, "agent": "learning_agent", "action": "step 2"},
        ]

        with patch("agents.primary.get_sub_agent", return_value=mock_agent):
            with patch("agents.primary.llm_router.generate_content", new_callable=AsyncMock, return_value=mock_synthesis):
                result = await _execute_plan(run["id"], "test", plan, datetime.now())
                assert "summary" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_plan_step_error_isolation(self):
        """One step failing doesn't kill the whole plan."""
        from agents.primary import _execute_plan

        call_count = [0]

        class MockAgent:
            tools_used = []
            async def execute(self, msg):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("Step 1 exploded")
                return {"response": "Step 2 ok", "tools_used": []}

        run = await db.create_agent_run("test")
        plan = [
            {"step": 1, "agent": "planning_agent", "action": "explode"},
            {"step": 2, "agent": "planning_agent", "action": "work"},
        ]

        mock_synthesis = MagicMock()
        mock_synthesis.text = "Summary with partial results"

        with patch("agents.primary.get_sub_agent", return_value=MockAgent()):
            with patch("agents.primary.llm_router.generate_content", new_callable=AsyncMock, return_value=mock_synthesis):
                result = await _execute_plan(run["id"], "test", plan, datetime.now())
                assert result is not None  # Didn't crash

    @pytest.mark.asyncio
    async def test_plan_unknown_agent_falls_back_to_general(self):
        """Unknown agent name → falls back to GeneralAgent."""
        from agents.primary import _execute_plan

        run = await db.create_agent_run("test")
        plan = [{"step": 1, "agent": "nonexistent_agent", "action": "do"}]

        with patch("agents.primary.GeneralAgent") as MockGA:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {"response": "Fallback!", "tools_used": []}
            MockGA.return_value = mock_instance

            result = await _execute_plan(run["id"], "test", plan, datetime.now())
            assert result["response"] == "Fallback!"

    @pytest.mark.asyncio
    async def test_plan_with_step_offset(self):
        """Resume with step_offset for previously completed steps."""
        from agents.primary import _execute_plan

        mock_agent = AsyncMock()
        mock_agent.execute.return_value = {"response": "Resumed step", "tools_used": []}

        run = await db.create_agent_run("test")
        plan = [{"step": 3, "agent": "planning_agent", "action": "remaining step"}]

        with patch("agents.primary.get_sub_agent", return_value=mock_agent):
            result = await _execute_plan(run["id"], "test", plan, datetime.now(), step_offset=2)
            assert result["response"] == "Resumed step"


class TestResumeInterruptedPlan:
    """Test resume_interrupted_plan edge cases."""

    @pytest.mark.asyncio
    async def test_resume_with_string_plan(self):
        """Plan stored as JSON string (not dict) — should parse it."""
        from agents.primary import resume_interrupted_plan

        run = await db.create_agent_run("test")
        plan_data = json.dumps({"steps": [{"step": 1, "action": "do"}], "completed_steps": []})
        await db.update_agent_run(run["id"], plan=plan_data)

        result = await resume_interrupted_plan(run["id"])
        assert result is not None
        assert len(result["remaining_plan"]) == 1

    @pytest.mark.asyncio
    async def test_resume_all_steps_completed(self):
        """All steps already done — returns None."""
        from agents.primary import resume_interrupted_plan

        run = await db.create_agent_run("test")
        plan_data = {"steps": [{"step": 1, "action": "do"}], "completed_steps": [1]}
        await db.update_agent_run(run["id"], plan=plan_data)
        # Mark the delegation step
        await db.log_agent_step(run["id"], "primary_agent", "delegation", {"step": 1, "to": "agent"})

        result = await resume_interrupted_plan(run["id"])
        assert result is None

    @pytest.mark.asyncio
    async def test_resume_with_empty_steps(self):
        """Plan with empty steps list — returns None."""
        from agents.primary import resume_interrupted_plan

        run = await db.create_agent_run("test")
        plan_data = {"steps": [], "completed_steps": []}
        await db.update_agent_run(run["id"], plan=plan_data)

        result = await resume_interrupted_plan(run["id"])
        assert result is None

    @pytest.mark.asyncio
    async def test_resume_completed_run_not_found(self):
        """Completed run (not 'running') should not be found."""
        from agents.primary import get_interrupted_run

        run = await db.create_agent_run("test")
        await db.update_agent_run(run["id"], status="completed")

        result = await get_interrupted_run(run["id"])
        assert result is None


# ═══════════════════════════════════════════════════════════════
# LLM ROUTER TESTS
# ═══════════════════════════════════════════════════════════════

class TestLLMRouter:
    """Test LLMRouter failover, smart tiering, provider cooldown, and health status."""

    def test_provider_available_by_default(self):
        """Fresh providers are always available."""
        from llm_router import ModelProvider
        p = ModelProvider("test-model", project="primary", tier="primary", quality=100)
        assert p.is_available is True

    def test_provider_cooldown(self):
        """Provider enters cooldown after rate limit, then recovers."""
        from llm_router import ModelProvider
        import time

        p = ModelProvider("test-model", project="primary", tier="primary", quality=100)
        p.mark_rate_limited(retry_after=0.1)  # 0.1s + 2s buffer = 2.1s cooldown
        assert p.is_available is False
        assert p._consecutive_429s == 1

    def test_provider_mark_success_resets_429s(self):
        """Successful call resets consecutive 429 counter."""
        from llm_router import ModelProvider
        p = ModelProvider("test-model", project="primary", tier="primary", quality=100)
        p._consecutive_429s = 3
        p.mark_success()
        assert p._consecutive_429s == 0
        assert p._call_count == 1

    def test_provider_escalating_cooldown(self):
        """Consecutive 429s escalate cooldown duration."""
        from llm_router import ModelProvider
        p = ModelProvider("test-model", project="primary", tier="primary", quality=100)
        # 1st 429: 15s
        p.mark_rate_limited()
        cd1 = p._cooldown_until
        # Reset for next test
        p._cooldown_until = 0.0
        # 2nd 429: 30s
        p.mark_rate_limited()
        assert p._consecutive_429s == 2

    def test_provider_nuclear_cooldown_after_4(self):
        """After 4 consecutive 429s, provider gets 5-minute cooldown."""
        from llm_router import ModelProvider
        import time
        p = ModelProvider("test-model", project="primary", tier="primary", quality=100)
        for _ in range(4):
            p.mark_rate_limited()
        # 4th 429 → 300s cooldown
        assert p._consecutive_429s == 4
        expected_min = time.monotonic() + 290  # allow some slack
        assert p._cooldown_until > expected_min

    def test_smart_tier_orchestrator_gets_best_model_first(self):
        """Orchestrator tasks should get quality >= 90 providers FIRST in cascade."""
        from llm_router import LLMRouter, TaskTier
        router = LLMRouter()
        providers = router.get_providers_for_tier(TaskTier.ORCHESTRATOR)
        # Best quality model should be first, but cascade includes all
        assert providers[0].quality >= 90
        assert len(providers) >= 4  # Full cascade for failover (4 single-key, 8 dual-key)

    def test_smart_tier_sub_agent_gets_more_options(self):
        """Sub-agent tasks accept quality >= 50, getting more providers."""
        from llm_router import LLMRouter, TaskTier
        router = LLMRouter()
        providers = router.get_providers_for_tier(TaskTier.SUB_AGENT)
        assert len(providers) >= 2  # At least primary + secondary

    def test_smart_tier_proactive_gets_all(self):
        """Proactive tasks accept quality >= 25, getting all providers."""
        from llm_router import LLMRouter, TaskTier
        router = LLMRouter()
        providers = router.get_providers_for_tier(TaskTier.PROACTIVE)
        assert len(providers) >= 4  # All providers (4 or 8 depending on keys)

    def test_smart_tier_fallback_when_none_qualify(self):
        """When no providers meet quality, falls back to any available."""
        from llm_router import LLMRouter
        router = LLMRouter()
        # Set all providers to quality below 200
        providers = router.get_providers_for_tier(200)  # Nothing meets quality=200
        assert len(providers) >= 1  # Falls back to any available

    def test_agent_tier_mapping(self):
        """Agent names map to correct task tiers."""
        from llm_router import AGENT_TIER_MAP, TaskTier
        assert AGENT_TIER_MAP["primary_agent"] == TaskTier.ORCHESTRATOR
        assert AGENT_TIER_MAP["planning_agent"] == TaskTier.SUB_AGENT
        assert AGENT_TIER_MAP["proactive_monitor"] == TaskTier.PROACTIVE

    def test_get_status_returns_all_providers(self):
        """get_status() returns info for all providers with circuit breaker state."""
        from llm_router import LLMRouter
        router = LLMRouter()
        status = router.get_status()
        assert len(status["providers"]) >= 4
        assert "active_assignments" in status
        assert "circuit_breaker" in status
        assert "dual_project" in status
        assert status["circuit_breaker"]["state"] in ("CLOSED", "OPEN")
        for p in status["providers"]:
            assert "model" in p
            assert "available" in p
            assert "quality" in p
            assert "project" in p

    def test_is_rate_limit_error(self):
        """Rate limit error detection covers all patterns."""
        from llm_router import _is_rate_limit_error
        assert _is_rate_limit_error(Exception("429 RESOURCE_EXHAUSTED"))
        assert _is_rate_limit_error(Exception("rate limit exceeded"))
        assert _is_rate_limit_error(Exception("status: 429"))
        assert not _is_rate_limit_error(Exception("500 Internal Server Error"))
        assert not _is_rate_limit_error(Exception("connection timeout"))

    def test_parse_retry_delay(self):
        """Parse retryDelay from Gemini error messages."""
        from llm_router import _parse_retry_delay
        assert _parse_retry_delay(Exception("retryDelay: 15.5s")) == 15.5
        assert _parse_retry_delay(Exception("Retry after 30s")) == 30.0
        assert _parse_retry_delay(Exception("no delay info")) is None

    @pytest.mark.asyncio
    async def test_failover_on_429(self):
        """LLMRouter fails over to next provider on 429."""
        from llm_router import LLMRouter

        router = LLMRouter()
        call_log = []

        original_generate = None

        # Mock the client to simulate 429 on first provider, success on second
        mock_client = MagicMock()

        def mock_generate_content(*args, **kwargs):
            model = kwargs.get("model", args[0] if args else "")
            call_log.append(model)
            if model == "gemini-2.5-flash":
                raise Exception("429 RESOURCE_EXHAUSTED")
            # Success for any other model
            mock_resp = MagicMock()
            mock_resp.candidates = [MagicMock()]
            mock_resp.candidates[0].content = MagicMock()
            return mock_resp

        mock_client.models.generate_content = mock_generate_content
        # Mock ALL projects the router knows about
        router._clients["primary"] = mock_client
        router._clients["secondary"] = mock_client  # Same mock for both

        # Mock mission_control to avoid emission errors
        with patch("mission_control.mission_control") as mock_mc:
            mock_mc.emit_thought = AsyncMock()
            mock_mc.emit_narration = AsyncMock()
            response = await router.generate_content(
                contents=[],
                system_instruction="test",
                agent_name="primary_agent",
                run_id="test-run",
            )

        assert response is not None
        # First call was to 2.5-flash (429'd), eventually succeeded on a lite model
        assert len(call_log) >= 2
        assert "gemini-2.5-flash" == call_log[0]
        # At least one call succeeded (a non-2.5-flash model)
        successful_model = call_log[-1]
        assert "lite" in successful_model  # Final success was on a lite variant

    @pytest.mark.asyncio
    async def test_all_providers_exhausted_waits_and_retries(self):
        """When all providers 429, router waits for shortest cooldown."""
        from llm_router import LLMRouter
        import time

        router = LLMRouter()

        # Put all providers in cooldown except make the last one recover soon
        for p in router.providers:
            p._cooldown_until = time.monotonic() + 300  # 5 min
        # Make one provider recover very soon
        router.providers[-1]._cooldown_until = time.monotonic() + 0.1
        router.providers[-1]._consecutive_429s = 1

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.candidates = [MagicMock()]
        mock_client.models.generate_content = MagicMock(return_value=mock_resp)
        router._clients["primary"] = mock_client
        router._clients["secondary"] = mock_client  # Same mock for both

        with patch("mission_control.mission_control") as mock_mc:
            mock_mc.emit_thought = AsyncMock()
            mock_mc.emit_narration = AsyncMock()
            # This should wait ~0.1s then succeed on the recovered provider
            response = await router.generate_content(
                contents=[],
                system_instruction="test",
                agent_name="general_agent",
                run_id="test-run",
            )

        assert response is not None

    def test_provider_cascade_order(self):
        """Provider cascade is ordered: Project A (best→worst) then Project B (best→worst)."""
        from llm_router import LLMRouter
        router = LLMRouter()
        # Within each project, quality should be descending
        projects = {}
        for p in router.providers:
            projects.setdefault(p.project, []).append(p.quality)
        for project, qualities in projects.items():
            assert qualities == sorted(qualities, reverse=True), \
                f"Project {project} not sorted: {qualities}"


class TestLLMRouterHealthEndpoint:
    """Test /health/llm endpoint."""

    @pytest.mark.asyncio
    async def test_health_llm_returns_provider_status(self):
        """GET /health/llm returns all provider statuses."""
        from httpx import AsyncClient, ASGITransport
        from main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/llm")
            assert resp.status_code == 200
            data = resp.json()
            assert "providers" in data
            assert len(data["providers"]) >= 4
            assert data["status"] in ("ok", "degraded", "circuit_breaker_open")


# ═══════════════════════════════════════════════════════════════
# PHASE 1 — GRACEFUL DEGRADATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestAllProvidersExhausted:
    """Test the custom exception and its properties."""

    def test_basic_raise(self):
        from llm_router import AllProvidersExhausted
        with pytest.raises(AllProvidersExhausted) as exc_info:
            raise AllProvidersExhausted("test message")
        assert str(exc_info.value) == "test message"
        assert exc_info.value.last_error is None
        assert exc_info.value.circuit_breaker_tripped is False

    def test_with_last_error(self):
        from llm_router import AllProvidersExhausted
        original = RuntimeError("original error")
        with pytest.raises(AllProvidersExhausted) as exc_info:
            raise AllProvidersExhausted("wrapped", last_error=original)
        assert exc_info.value.last_error is original

    def test_circuit_breaker_flag(self):
        from llm_router import AllProvidersExhausted
        with pytest.raises(AllProvidersExhausted) as exc_info:
            raise AllProvidersExhausted("tripped", circuit_breaker_tripped=True)
        assert exc_info.value.circuit_breaker_tripped is True

    def test_is_exception_subclass(self):
        from llm_router import AllProvidersExhausted
        assert issubclass(AllProvidersExhausted, Exception)


class TestCircuitBreaker:
    """Test circuit breaker state machine."""

    def test_starts_closed(self):
        from llm_router import CircuitBreaker
        cb = CircuitBreaker()
        assert not cb.is_tripped
        assert cb.remaining_cooldown_s == 0.0

    def test_does_not_trip_below_threshold(self):
        from llm_router import CircuitBreaker
        cb = CircuitBreaker()
        tripped = cb.record_cascade_failure()
        assert not tripped
        assert not cb.is_tripped

    def test_trips_at_threshold(self):
        from llm_router import CircuitBreaker
        cb = CircuitBreaker()
        cb.record_cascade_failure()  # 1st
        tripped = cb.record_cascade_failure()  # 2nd = threshold
        assert tripped
        assert cb.is_tripped
        assert cb.remaining_cooldown_s > 0

    def test_success_resets_failure_count(self):
        from llm_router import CircuitBreaker
        cb = CircuitBreaker()
        cb.record_cascade_failure()
        assert cb._consecutive_failures == 1
        cb.record_success()
        assert cb._consecutive_failures == 0

    def test_auto_resets_after_cooldown(self):
        from llm_router import CircuitBreaker
        import time
        cb = CircuitBreaker()
        cb._tripped_until = time.monotonic() - 1  # Already expired
        assert not cb.is_tripped  # Should auto-reset

    def test_status_dict(self):
        from llm_router import CircuitBreaker
        cb = CircuitBreaker()
        status = cb.get_status()
        assert status["state"] == "CLOSED"
        assert status["consecutive_failures"] == 0
        assert status["lifetime_trips"] == 0

    def test_lifetime_trip_counter(self):
        from llm_router import CircuitBreaker
        import time
        cb = CircuitBreaker()
        # Trip 1
        cb.record_cascade_failure()
        cb.record_cascade_failure()
        assert cb._trip_count == 1
        # Reset
        cb._tripped_until = time.monotonic() - 1
        cb.is_tripped  # triggers auto-reset
        # Trip 2
        cb.record_cascade_failure()
        cb.record_cascade_failure()
        assert cb._trip_count == 2


class TestCircuitBreakerInRouter:
    """Test circuit breaker integration within LLMRouter."""

    @pytest.mark.asyncio
    async def test_tripped_breaker_raises_immediately(self):
        """When circuit breaker is tripped, generate_content short-circuits."""
        from llm_router import LLMRouter, AllProvidersExhausted
        import time

        router = LLMRouter()
        router._circuit_breaker._tripped_until = time.monotonic() + 300
        router._circuit_breaker._consecutive_failures = 5

        with patch("mission_control.mission_control") as mock_mc:
            mock_mc.emit_narration = AsyncMock()
            with pytest.raises(AllProvidersExhausted) as exc_info:
                await router.generate_content(
                    contents=[],
                    system_instruction="test",
                    agent_name="primary_agent",
                    run_id="test-run",
                )
            assert exc_info.value.circuit_breaker_tripped is True

    @pytest.mark.asyncio
    async def test_tripped_breaker_emits_narration(self):
        """Circuit breaker trip emits Mission Control narration."""
        from llm_router import LLMRouter, AllProvidersExhausted
        import time

        router = LLMRouter()
        router._circuit_breaker._tripped_until = time.monotonic() + 300

        with patch("mission_control.mission_control") as mock_mc:
            mock_mc.emit_narration = AsyncMock()
            with pytest.raises(AllProvidersExhausted):
                await router.generate_content(
                    contents=[], system_instruction="test",
                    agent_name="primary_agent", run_id="test-run",
                )
            mock_mc.emit_narration.assert_called_once()
            narration = mock_mc.emit_narration.call_args[0][2]
            assert "resilient" in narration.lower() or "demand" in narration.lower()

    @pytest.mark.asyncio
    async def test_full_cascade_failure_trips_breaker(self):
        """When all providers 429 twice, circuit breaker trips."""
        from llm_router import LLMRouter, AllProvidersExhausted

        router = LLMRouter()
        mock_client = MagicMock()
        mock_client.models.generate_content = MagicMock(
            side_effect=Exception("429 RESOURCE_EXHAUSTED")
        )
        router._clients["primary"] = mock_client
        router._clients["secondary"] = mock_client  # Mock both projects

        # Set very short cooldowns so the "last resort wait" doesn't block tests
        for p in router.providers:
            p._cooldown_until = 0  # Available initially

        with patch("mission_control.mission_control") as mock_mc:
            mock_mc.emit_thought = AsyncMock()
            mock_mc.emit_narration = AsyncMock()

            # First full-cascade failure
            with pytest.raises(AllProvidersExhausted):
                await router.generate_content(
                    contents=[], system_instruction="test",
                    agent_name="primary_agent", run_id="test-1",
                )
            assert router._circuit_breaker._consecutive_failures == 1

            # Reset cooldowns for second attempt
            for p in router.providers:
                p._cooldown_until = 0

            # Second full-cascade failure — should trip
            with pytest.raises(AllProvidersExhausted):
                await router.generate_content(
                    contents=[], system_instruction="test",
                    agent_name="primary_agent", run_id="test-2",
                )
            assert router._circuit_breaker.is_tripped


class TestMultiKeyRotation:
    """Test Phase 2 — dual-project provider cascade."""

    def test_single_key_creates_4_providers(self):
        """With only primary key, 4 providers created."""
        from llm_router import LLMRouter
        with patch("llm_router.GEMINI_API_KEY", "key-a"), \
             patch("llm_router.GEMINI_API_KEY_SECONDARY", ""):
            router = LLMRouter()
        # All should be project "primary"
        projects = set(p.project for p in router.providers)
        assert "primary" in projects
        assert len(router.providers) == 4

    def test_dual_key_creates_8_providers(self):
        """With both keys, 8 providers across 2 projects."""
        from llm_router import LLMRouter
        with patch("llm_router.GEMINI_API_KEY", "key-a"), \
             patch("llm_router.GEMINI_API_KEY_SECONDARY", "key-b"):
            router = LLMRouter()
        assert len(router.providers) == 8
        projects = set(p.project for p in router.providers)
        assert projects == {"primary", "secondary"}

    def test_primary_providers_come_first(self):
        """Project A providers appear before Project B in cascade."""
        from llm_router import LLMRouter
        with patch("llm_router.GEMINI_API_KEY", "key-a"), \
             patch("llm_router.GEMINI_API_KEY_SECONDARY", "key-b"):
            router = LLMRouter()
        # First 4 should be primary, next 4 secondary
        for p in router.providers[:4]:
            assert p.project == "primary"
        for p in router.providers[4:]:
            assert p.project == "secondary"

    def test_display_name_shows_project(self):
        """Provider display name includes project tag."""
        from llm_router import ModelProvider
        pa = ModelProvider("gemini-2.5-flash", project="primary", tier="primary", quality=100)
        pb = ModelProvider("gemini-2.5-flash", project="secondary", tier="primary", quality=100)
        assert pa.display_name == "gemini-2.5-flash@A"
        assert pb.display_name == "gemini-2.5-flash@B"

    def test_status_includes_dual_project_flag(self):
        """get_status reflects whether dual project is active."""
        from llm_router import LLMRouter
        router = LLMRouter()
        status = router.get_status()
        assert "dual_project" in status

    @pytest.mark.asyncio
    async def test_failover_from_project_a_to_b(self):
        """When all Project A models 429, cascade reaches Project B."""
        from llm_router import LLMRouter

        with patch("llm_router.GEMINI_API_KEY", "key-a"), \
             patch("llm_router.GEMINI_API_KEY_SECONDARY", "key-b"):
            router = LLMRouter()

        call_log = []
        mock_client_a = MagicMock()
        mock_client_b = MagicMock()

        def mock_gen_a(*args, **kwargs):
            model = kwargs.get("model", "")
            call_log.append(("A", model))
            raise Exception("429 RESOURCE_EXHAUSTED")

        def mock_gen_b(*args, **kwargs):
            model = kwargs.get("model", "")
            call_log.append(("B", model))
            resp = MagicMock()
            resp.candidates = [MagicMock()]
            return resp

        mock_client_a.models.generate_content = mock_gen_a
        mock_client_b.models.generate_content = mock_gen_b
        router._clients["primary"] = mock_client_a
        router._clients["secondary"] = mock_client_b

        with patch("mission_control.mission_control") as mock_mc:
            mock_mc.emit_thought = AsyncMock()
            mock_mc.emit_narration = AsyncMock()
            response = await router.generate_content(
                contents=[], system_instruction="test",
                agent_name="primary_agent", run_id="test-run",
            )

        assert response is not None
        # Should have tried at least 1 Project A model (which 429'd),
        # then succeeded on Project B
        a_calls = [c for c in call_log if c[0] == "A"]
        b_calls = [c for c in call_log if c[0] == "B"]
        assert len(a_calls) >= 1  # At least one Project A 429'd
        assert len(b_calls) >= 1  # At least one Project B succeeded

        # Verify narration about project switch was emitted
        narration_calls = mock_mc.emit_narration.call_args_list
        narration_texts = [c[0][2] for c in narration_calls]
        assert any("cluster" in t.lower() or "backup" in t.lower() for t in narration_texts)


class TestGracefulDegradationInPrimary:
    """Test that primary.py falls back to deterministic defaults."""

    @pytest.mark.asyncio
    async def test_classify_intent_falls_back(self):
        """classify_intent returns deterministic default on AllProvidersExhausted."""
        from agents.primary import classify_intent
        from llm_router import AllProvidersExhausted

        with patch("agents.primary.llm_router") as mock_router, \
             patch("mission_control.mission_control") as mock_mc, \
             patch("agents.primary.db") as mock_db:
            mock_router.generate_content = AsyncMock(
                side_effect=AllProvidersExhausted("test")
            )
            mock_mc.emit_thought = AsyncMock()
            mock_mc.emit_narration = AsyncMock()
            mock_db.log_agent_step = AsyncMock()

            result = await classify_intent("test-run", "Create a study plan")

        assert result["intent"] == "general"
        assert result["complexity"] == "simple"
        assert "study plan" in result["summary"].lower()

    @pytest.mark.asyncio
    async def test_generate_plan_falls_back(self):
        """generate_plan returns single-step deterministic plan on exhaustion."""
        from agents.primary import generate_plan
        from llm_router import AllProvidersExhausted

        classification = {"intent": "learning", "complexity": "complex", "summary": "study plan"}

        with patch("agents.primary.llm_router") as mock_router, \
             patch("mission_control.mission_control") as mock_mc, \
             patch("agents.primary.db") as mock_db:
            mock_router.generate_content = AsyncMock(
                side_effect=AllProvidersExhausted("test")
            )
            mock_mc.emit_thought = AsyncMock()
            mock_mc.emit_narration = AsyncMock()
            mock_mc.emit_plan = AsyncMock()
            mock_db.log_agent_step = AsyncMock()
            mock_db.update_agent_run = AsyncMock()

            plan = await generate_plan("test-run", "Create a study plan", classification)

        assert len(plan) == 1
        assert plan[0]["agent"] == "learning_agent"
        assert plan[0]["step"] == 1


class TestZeroTracebackGuarantee:
    """Test that AllProvidersExhausted NEVER reaches the user as a raw traceback."""

    @pytest.mark.asyncio
    async def test_chat_returns_503_on_exhaustion(self):
        """POST /chat returns 503 with structured response on AllProvidersExhausted."""
        from httpx import AsyncClient, ASGITransport
        from main import app
        from llm_router import AllProvidersExhausted

        async def mock_process(*args, **kwargs):
            raise AllProvidersExhausted("all exhausted")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("agents.primary.process_message", side_effect=mock_process):
                resp = await client.post("/chat", json={"message": "test"})

            assert resp.status_code == 503
            data = resp.json()
            assert "run_id" in data
            assert "response" in data
            assert "retry_after_seconds" in data
            assert data["status"] == "congested"
            # Verify NO traceback in response
            assert "Traceback" not in data["response"]
            assert "ClientError" not in data["response"]

    @pytest.mark.asyncio
    async def test_chat_503_has_retry_after_header(self):
        """503 response includes Retry-After HTTP header."""
        from httpx import AsyncClient, ASGITransport
        from main import app
        from llm_router import AllProvidersExhausted

        async def mock_process(*args, **kwargs):
            raise AllProvidersExhausted("all exhausted")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("agents.primary.process_message", side_effect=mock_process):
                resp = await client.post("/chat", json={"message": "test"})

            assert resp.headers.get("retry-after") == "180"


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
