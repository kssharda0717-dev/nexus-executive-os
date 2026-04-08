"""Proactive Monitor Agent — Background watchdog that acts without user prompting.

This agent runs on a schedule (via APScheduler) and:
- Detects overdue tasks and reschedules them
- Generates daily briefing notes
- Checks for calendar conflicts
- Sends alerts for upcoming deadlines
- Creates daily summary notes

It can also be triggered manually via the /chat endpoint.
"""

from datetime import datetime, timedelta
from google.genai import types
from agents.base import BaseAgent
from tools import calendar_tool, tasks_tool, notes_tool
from mission_control import mission_control
import database as db


class ProactiveMonitorAgent(BaseAgent):
    name = "proactive_monitor"
    description = "Background monitor: overdue detection, daily briefings, rescheduling"
    system_prompt = """You are the Proactive Monitor of NEXUS, an autonomous life operating system.

You run in the background and proactively manage the user's system state.

## Your responsibilities:
1. **Overdue Detection**: Find tasks past their due date and flag or reschedule them
2. **Daily Briefing**: Generate a morning briefing with today's events, due tasks, and priorities
3. **Conflict Detection**: Check for overlapping calendar events
4. **Deadline Alerts**: Flag tasks due in the next 24 hours
5. **System Health**: Ensure data consistency across tools

## When generating a daily briefing:
- List today's calendar events chronologically
- List tasks due today or overdue
- Highlight urgent items
- Suggest priority order for the day
- Create a note titled "Daily Briefing — [date]"

## When handling overdue tasks:
- List all overdue tasks
- For each: suggest rescheduling to the next available slot
- Update task status and due dates
- Log changes in a note

## Important:
- Always use ISO 8601 format
- Be concise — briefings should be scannable
- Prioritize actionability over completeness

Today's date context will be provided in the user message."""

    def __init__(self, run_id: str):
        super().__init__(run_id)
        self.tools = [self._build_tools()]
        self.tool_handlers = {
            "list_events": calendar_tool.list_events,
            "create_event": calendar_tool.create_event,
            "find_free_slots": calendar_tool.find_free_slots,
            "delete_event": calendar_tool.delete_event,
            "list_tasks": tasks_tool.list_tasks,
            "create_task": tasks_tool.create_task,
            "update_task": tasks_tool.update_task,
            "get_overdue_tasks": tasks_tool.get_overdue_tasks,
            "complete_task": tasks_tool.complete_task,
            "list_notes": notes_tool.list_notes,
            "create_note": notes_tool.create_note,
            "search_notes": notes_tool.search_notes,
        }

    def _build_tools(self):
        return types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="list_events",
                description="List calendar events for a given time range",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "start": types.Schema(type="STRING"),
                        "end": types.Schema(type="STRING"),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="create_event",
                description="Create a rescheduled event",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "title": types.Schema(type="STRING"),
                        "start_time": types.Schema(type="STRING"),
                        "end_time": types.Schema(type="STRING"),
                        "description": types.Schema(type="STRING"),
                        "linked_task_id": types.Schema(type="STRING"),
                    },
                    required=["title", "start_time", "end_time"],
                ),
            ),
            types.FunctionDeclaration(
                name="find_free_slots",
                description="Find available slots for rescheduling",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "start": types.Schema(type="STRING"),
                        "end": types.Schema(type="STRING"),
                        "duration_minutes": types.Schema(type="INTEGER"),
                    },
                    required=["start", "end"],
                ),
            ),
            types.FunctionDeclaration(
                name="delete_event",
                description="Remove an outdated event",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"event_id": types.Schema(type="STRING")},
                    required=["event_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="list_tasks",
                description="List tasks with optional filters",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "status": types.Schema(type="STRING"),
                        "priority": types.Schema(type="STRING"),
                        "due_before": types.Schema(type="STRING"),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="create_task",
                description="Create a follow-up task",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "title": types.Schema(type="STRING"),
                        "description": types.Schema(type="STRING"),
                        "priority": types.Schema(type="STRING"),
                        "due_date": types.Schema(type="STRING"),
                        "estimated_minutes": types.Schema(type="INTEGER"),
                        "tags": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    },
                    required=["title"],
                ),
            ),
            types.FunctionDeclaration(
                name="update_task",
                description="Reschedule or update a task",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "task_id": types.Schema(type="STRING"),
                        "status": types.Schema(type="STRING"),
                        "priority": types.Schema(type="STRING"),
                        "due_date": types.Schema(type="STRING"),
                    },
                    required=["task_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_overdue_tasks",
                description="Get all overdue tasks",
                parameters=types.Schema(type="OBJECT", properties={}),
            ),
            types.FunctionDeclaration(
                name="complete_task",
                description="Mark a task as completed",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"task_id": types.Schema(type="STRING")},
                    required=["task_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="list_notes",
                description="List existing notes",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "tags": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="create_note",
                description="Create a briefing or status note",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "title": types.Schema(type="STRING"),
                        "content": types.Schema(type="STRING"),
                        "tags": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    },
                    required=["title", "content"],
                ),
            ),
            types.FunctionDeclaration(
                name="search_notes",
                description="Search for related notes",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"query": types.Schema(type="STRING")},
                    required=["query"],
                ),
            ),
        ])


# ── Proactive Trigger Functions (called by APScheduler) ──

async def run_daily_briefing():
    """Generate a daily briefing — called by the scheduler each morning."""
    run = await db.create_agent_run("PROACTIVE: Daily briefing", intent="proactive")
    run_id = run["id"]

    await mission_control.emit_thought(run_id, "proactive_monitor",
                                       "Triggered: Daily briefing generation")

    now = datetime.now()
    agent = ProactiveMonitorAgent(run_id)
    message = (
        f"[Current date/time: {now.isoformat()}]\n\n"
        f"Generate a daily briefing for today ({now.strftime('%A, %B %d, %Y')}). "
        f"Check today's calendar events, tasks due today and overdue tasks, "
        f"and create a briefing note with priorities for the day."
    )

    result = await agent.execute(message)
    await db.update_agent_run(run_id, status="completed", result=result.get("response", ""),
                              completed_at=datetime.now().isoformat())
    return result


async def run_overdue_check():
    """Check for overdue tasks and reschedule — called by scheduler."""
    run = await db.create_agent_run("PROACTIVE: Overdue check", intent="proactive")
    run_id = run["id"]

    await mission_control.emit_thought(run_id, "proactive_monitor",
                                       "Triggered: Overdue task detection and rescheduling")

    now = datetime.now()
    agent = ProactiveMonitorAgent(run_id)
    message = (
        f"[Current date/time: {now.isoformat()}]\n\n"
        f"Check for overdue tasks. For any found, reschedule them to the next available "
        f"time slot and update their due dates. Create a note logging all changes made."
    )

    result = await agent.execute(message)
    await db.update_agent_run(run_id, status="completed", result=result.get("response", ""),
                              completed_at=datetime.now().isoformat())
    return result
