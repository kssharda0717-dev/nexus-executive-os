"""Notes MCP Server — SQLite-backed notes with search, tags, and task linking."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
import database as db

mcp = FastMCP("nexus-notes", instructions="Notes management for NEXUS. Supports rich content, tags, search, and task linking.")


@mcp.tool()
async def list_notes(tags: str = None) -> list[dict]:
    """List all notes, optionally filtered by tags.

    Args:
        tags: Comma-separated tags to filter by
    """
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    return await db.list_notes(tags=tag_list)


@mcp.tool()
async def create_note(title: str, content: str, tags: str = None,
                      linked_task_ids: str = None) -> dict:
    """Create a new note.

    Args:
        title: Note title
        content: Note content (supports markdown)
        tags: Comma-separated tags
        linked_task_ids: Comma-separated task IDs to link to this note
    """
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    task_list = [t.strip() for t in linked_task_ids.split(",")] if linked_task_ids else None
    return await db.create_note(title=title, content=content,
                                tags=tag_list, linked_task_ids=task_list)


@mcp.tool()
async def get_note(note_id: str) -> dict:
    """Get a single note by ID.

    Args:
        note_id: The note's unique identifier
    """
    result = await db.get_note(note_id)
    if not result:
        return {"error": f"Note {note_id} not found"}
    return result


@mcp.tool()
async def update_note(note_id: str, title: str = None, content: str = None,
                      tags: str = None, linked_task_ids: str = None) -> dict:
    """Update an existing note.

    Args:
        note_id: The note to update
        title: New title
        content: New content
        tags: New comma-separated tags
        linked_task_ids: New comma-separated linked task IDs
    """
    fields = {}
    if title is not None: fields["title"] = title
    if content is not None: fields["content"] = content
    if tags is not None: fields["tags"] = [t.strip() for t in tags.split(",")]
    if linked_task_ids is not None: fields["linked_task_ids"] = [t.strip() for t in linked_task_ids.split(",")]
    if not fields:
        return {"error": "No fields to update"}
    result = await db.update_note(note_id, **fields)
    if not result:
        return {"error": f"Note {note_id} not found"}
    return result


@mcp.tool()
async def search_notes(query: str) -> list[dict]:
    """Search notes by title or content.

    Args:
        query: Search query string
    """
    return await db.search_notes(query)


@mcp.tool()
async def append_to_note(note_id: str, content: str) -> dict:
    """Append content to an existing note.

    Args:
        note_id: The note to append to
        content: Content to append
    """
    result = await db.append_to_note(note_id, content)
    if not result:
        return {"error": f"Note {note_id} not found"}
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
