"""Calendar MCP Server — SQLite-backed calendar with full CRUD + free-slot finder."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
import database as db

mcp = FastMCP("nexus-calendar", instructions="Calendar management for NEXUS. All times in ISO 8601.")


@mcp.tool()
async def list_events(start: str = None, end: str = None) -> list[dict]:
    """List calendar events, optionally filtered by time range.

    Args:
        start: ISO 8601 start datetime filter (inclusive)
        end: ISO 8601 end datetime filter (inclusive)
    """
    return await db.list_events(start, end)


@mcp.tool()
async def create_event(title: str, start_time: str, end_time: str,
                       description: str = None, linked_task_id: str = None) -> dict:
    """Create a new calendar event.

    Args:
        title: Event title
        start_time: ISO 8601 start datetime
        end_time: ISO 8601 end datetime
        description: Optional event description
        linked_task_id: Optional task ID to link this event to
    """
    return await db.create_event(title, start_time, end_time, description, linked_task_id)


@mcp.tool()
async def get_event(event_id: str) -> dict:
    """Get a single calendar event by ID.

    Args:
        event_id: The event's unique identifier
    """
    result = await db.get_event(event_id)
    if not result:
        return {"error": f"Event {event_id} not found"}
    return result


@mcp.tool()
async def update_event(event_id: str, title: str = None, description: str = None,
                       start_time: str = None, end_time: str = None,
                       linked_task_id: str = None) -> dict:
    """Update an existing calendar event.

    Args:
        event_id: The event to update
        title: New title (optional)
        description: New description (optional)
        start_time: New start time (optional)
        end_time: New end time (optional)
        linked_task_id: New linked task ID (optional)
    """
    fields = {k: v for k, v in {
        "title": title, "description": description,
        "start_time": start_time, "end_time": end_time,
        "linked_task_id": linked_task_id
    }.items() if v is not None}
    if not fields:
        return {"error": "No fields to update"}
    result = await db.update_event(event_id, **fields)
    if not result:
        return {"error": f"Event {event_id} not found"}
    return result


@mcp.tool()
async def delete_event(event_id: str) -> dict:
    """Delete a calendar event.

    Args:
        event_id: The event to delete
    """
    success = await db.delete_event(event_id)
    return {"deleted": success, "event_id": event_id}


@mcp.tool()
async def find_free_slots(start: str, end: str, duration_minutes: int = 60) -> list[dict]:
    """Find available time slots in the calendar.

    Args:
        start: ISO 8601 start of search window
        end: ISO 8601 end of search window
        duration_minutes: Minimum slot duration in minutes (default 60)
    """
    return await db.find_free_slots(start, end, duration_minutes)


if __name__ == "__main__":
    mcp.run(transport="stdio")
