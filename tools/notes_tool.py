"""Notes tool functions for agents — wraps DB ops with Mission Control streaming."""

import database as db
from mission_control import mission_control


async def list_notes(run_id: str, agent_name: str,
                     tags: list[str] = None) -> list[dict]:
    """List notes with optional tag filter."""
    await mission_control.emit_tool_call(run_id, agent_name, "notes.list_notes",
                                         {"tags": tags})
    result = await db.list_notes(tags=tags)
    await mission_control.emit_tool_result(run_id, agent_name, "notes.list_notes",
                                           {"count": len(result), "notes": result})
    return result


async def create_note(run_id: str, agent_name: str,
                      title: str, content: str,
                      tags: list[str] = None,
                      linked_task_ids: list[str] = None) -> dict:
    """Create a new note."""
    await mission_control.emit_tool_call(run_id, agent_name, "notes.create_note",
                                         {"title": title, "tags": tags})
    result = await db.create_note(title=title, content=content,
                                   tags=tags, linked_task_ids=linked_task_ids)
    await mission_control.emit_tool_result(run_id, agent_name, "notes.create_note", result)
    return result


async def get_note(run_id: str, agent_name: str, note_id: str) -> dict | None:
    """Get a single note by ID."""
    await mission_control.emit_tool_call(run_id, agent_name, "notes.get_note",
                                         {"note_id": note_id})
    result = await db.get_note(note_id)
    await mission_control.emit_tool_result(run_id, agent_name, "notes.get_note",
                                           result or {"error": "not found"})
    return result


async def update_note(run_id: str, agent_name: str, note_id: str, **fields) -> dict | None:
    """Update a note."""
    await mission_control.emit_tool_call(run_id, agent_name, "notes.update_note",
                                         {"note_id": note_id, **fields})
    result = await db.update_note(note_id, **fields)
    await mission_control.emit_tool_result(run_id, agent_name, "notes.update_note",
                                           result or {"error": "not found"})
    return result


async def search_notes(run_id: str, agent_name: str, query: str) -> list[dict]:
    """Search notes by title or content."""
    await mission_control.emit_tool_call(run_id, agent_name, "notes.search_notes",
                                         {"query": query})
    result = await db.search_notes(query)
    await mission_control.emit_tool_result(run_id, agent_name, "notes.search_notes",
                                           {"count": len(result), "notes": result})
    return result


async def append_to_note(run_id: str, agent_name: str,
                         note_id: str, content: str) -> dict | None:
    """Append content to an existing note."""
    await mission_control.emit_tool_call(run_id, agent_name, "notes.append_to_note",
                                         {"note_id": note_id, "content_preview": content[:100]})
    result = await db.append_to_note(note_id, content)
    await mission_control.emit_tool_result(run_id, agent_name, "notes.append_to_note",
                                           result or {"error": "not found"})
    return result
