"""Calendar tool functions for agents — wraps DB ops with Mission Control streaming.

Includes smart default duration logic: when an LLM omits end_time,
creates a 30-minute event from start_time instead of failing.
"""

from datetime import datetime, timedelta

import database as db
from mission_control import mission_control

# Default event duration when the LLM omits end_time
DEFAULT_EVENT_DURATION_MINUTES = 30


def _ensure_end_time(start_time: str, end_time: str | None) -> str:
    """If end_time is missing or empty, compute it as start_time + 30 minutes."""
    if end_time and end_time.strip():
        return end_time
    try:
        start_dt = datetime.fromisoformat(start_time)
        return (start_dt + timedelta(minutes=DEFAULT_EVENT_DURATION_MINUTES)).isoformat()
    except (ValueError, TypeError):
        # If start_time is also malformed, pass through and let DB layer handle it
        return start_time


async def list_events(run_id: str, agent_name: str,
                      start: str = None, end: str = None) -> list[dict]:
    """List calendar events with optional time range filter."""
    await mission_control.emit_tool_call(run_id, agent_name, "calendar.list_events",
                                         {"start": start, "end": end})
    result = await db.list_events(start, end)
    await mission_control.emit_tool_result(run_id, agent_name, "calendar.list_events",
                                           {"count": len(result), "events": result})
    return result


async def create_event(run_id: str, agent_name: str,
                       title: str, start_time: str, end_time: str = None,
                       description: str = None, linked_task_id: str = None) -> dict:
    """Create a new calendar event.

    If end_time is missing, defaults to start_time + 30 minutes.
    """
    end_time = _ensure_end_time(start_time, end_time)
    await mission_control.emit_tool_call(run_id, agent_name, "calendar.create_event",
                                         {"title": title, "start_time": start_time,
                                          "end_time": end_time, "description": description})
    result = await db.create_event(title, start_time, end_time, description, linked_task_id)
    await mission_control.emit_tool_result(run_id, agent_name, "calendar.create_event", result)
    return result


async def get_event(run_id: str, agent_name: str, event_id: str) -> dict | None:
    """Get a single event by ID."""
    await mission_control.emit_tool_call(run_id, agent_name, "calendar.get_event",
                                         {"event_id": event_id})
    result = await db.get_event(event_id)
    await mission_control.emit_tool_result(run_id, agent_name, "calendar.get_event",
                                           result or {"error": "not found"})
    return result


async def update_event(run_id: str, agent_name: str, event_id: str, **fields) -> dict | None:
    """Update a calendar event."""
    await mission_control.emit_tool_call(run_id, agent_name, "calendar.update_event",
                                         {"event_id": event_id, **fields})
    result = await db.update_event(event_id, **fields)
    await mission_control.emit_tool_result(run_id, agent_name, "calendar.update_event",
                                           result or {"error": "not found"})
    return result


async def delete_event(run_id: str, agent_name: str, event_id: str) -> bool:
    """Delete a calendar event."""
    await mission_control.emit_tool_call(run_id, agent_name, "calendar.delete_event",
                                         {"event_id": event_id})
    result = await db.delete_event(event_id)
    await mission_control.emit_tool_result(run_id, agent_name, "calendar.delete_event",
                                           {"deleted": result})
    return result


async def find_free_slots(run_id: str, agent_name: str,
                          start: str, end: str, duration_minutes: int = 60) -> list[dict]:
    """Find free time slots in the calendar."""
    await mission_control.emit_tool_call(run_id, agent_name, "calendar.find_free_slots",
                                         {"start": start, "end": end,
                                          "duration_minutes": duration_minutes})
    result = await db.find_free_slots(start, end, duration_minutes)
    await mission_control.emit_tool_result(run_id, agent_name, "calendar.find_free_slots",
                                           {"count": len(result), "slots": result})
    return result
