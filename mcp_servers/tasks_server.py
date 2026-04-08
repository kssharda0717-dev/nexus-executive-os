"""Tasks MCP Server — SQLite-backed task manager with priority + dependency tracking."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
import database as db

mcp = FastMCP("nexus-tasks", instructions="Task management for NEXUS. Supports priorities, tags, dependencies, and status tracking.")


@mcp.tool()
async def list_tasks(status: str = None, priority: str = None,
                     due_before: str = None, tags: str = None) -> list[dict]:
    """List tasks with optional filters.

    Args:
        status: Filter by status (pending, scheduled, in_progress, completed, cancelled)
        priority: Filter by priority (low, medium, high, urgent)
        due_before: ISO 8601 datetime — return tasks due before this date
        tags: Comma-separated tags to filter by
    """
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    return await db.list_tasks(status=status, priority=priority,
                               due_before=due_before, tags=tag_list)


@mcp.tool()
async def create_task(title: str, description: str = None,
                      priority: str = "medium", due_date: str = None,
                      estimated_minutes: int = None, tags: str = None,
                      dependencies: str = None) -> dict:
    """Create a new task.

    Args:
        title: Task title
        description: Task description
        priority: Priority level (low, medium, high, urgent)
        due_date: ISO 8601 due date
        estimated_minutes: Estimated time to complete in minutes
        tags: Comma-separated tags
        dependencies: Comma-separated task IDs this task depends on
    """
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    dep_list = [d.strip() for d in dependencies.split(",")] if dependencies else None
    return await db.create_task(
        title=title, description=description, priority=priority,
        due_date=due_date, estimated_minutes=estimated_minutes,
        tags=tag_list, dependencies=dep_list
    )


@mcp.tool()
async def get_task(task_id: str) -> dict:
    """Get a single task by ID.

    Args:
        task_id: The task's unique identifier
    """
    result = await db.get_task(task_id)
    if not result:
        return {"error": f"Task {task_id} not found"}
    return result


@mcp.tool()
async def update_task(task_id: str, title: str = None, description: str = None,
                      status: str = None, priority: str = None,
                      due_date: str = None, estimated_minutes: int = None,
                      tags: str = None, dependencies: str = None) -> dict:
    """Update an existing task.

    Args:
        task_id: The task to update
        title: New title
        description: New description
        status: New status (pending, scheduled, in_progress, completed, cancelled)
        priority: New priority (low, medium, high, urgent)
        due_date: New due date (ISO 8601)
        estimated_minutes: New time estimate
        tags: New comma-separated tags
        dependencies: New comma-separated dependency task IDs
    """
    fields = {}
    if title is not None: fields["title"] = title
    if description is not None: fields["description"] = description
    if status is not None: fields["status"] = status
    if priority is not None: fields["priority"] = priority
    if due_date is not None: fields["due_date"] = due_date
    if estimated_minutes is not None: fields["estimated_minutes"] = estimated_minutes
    if tags is not None: fields["tags"] = [t.strip() for t in tags.split(",")]
    if dependencies is not None: fields["dependencies"] = [d.strip() for d in dependencies.split(",")]
    if not fields:
        return {"error": "No fields to update"}
    result = await db.update_task(task_id, **fields)
    if not result:
        return {"error": f"Task {task_id} not found"}
    return result


@mcp.tool()
async def complete_task(task_id: str) -> dict:
    """Mark a task as completed.

    Args:
        task_id: The task to complete
    """
    result = await db.complete_task(task_id)
    if not result:
        return {"error": f"Task {task_id} not found"}
    return result


@mcp.tool()
async def get_overdue_tasks() -> list[dict]:
    """Get all tasks that are past their due date and not completed."""
    return await db.get_overdue_tasks()


if __name__ == "__main__":
    mcp.run(transport="stdio")
