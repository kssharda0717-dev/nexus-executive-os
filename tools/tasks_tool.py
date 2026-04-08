"""Tasks tool functions for agents — wraps DB ops with Mission Control streaming."""

import database as db
from mission_control import mission_control


async def list_tasks(run_id: str, agent_name: str,
                     status: str = None, priority: str = None,
                     due_before: str = None, tags: list[str] = None) -> list[dict]:
    """List tasks with optional filters."""
    await mission_control.emit_tool_call(run_id, agent_name, "tasks.list_tasks",
                                         {"status": status, "priority": priority,
                                          "due_before": due_before, "tags": tags})
    result = await db.list_tasks(status=status, priority=priority,
                                  due_before=due_before, tags=tags)
    await mission_control.emit_tool_result(run_id, agent_name, "tasks.list_tasks",
                                           {"count": len(result), "tasks": result})
    return result


async def create_task(run_id: str, agent_name: str,
                      title: str, description: str = None,
                      priority: str = "medium", due_date: str = None,
                      estimated_minutes: int = None, tags: list[str] = None,
                      dependencies: list[str] = None) -> dict:
    """Create a new task."""
    await mission_control.emit_tool_call(run_id, agent_name, "tasks.create_task",
                                         {"title": title, "priority": priority,
                                          "due_date": due_date, "tags": tags})
    result = await db.create_task(
        title=title, description=description, priority=priority,
        due_date=due_date, estimated_minutes=estimated_minutes,
        tags=tags, dependencies=dependencies
    )
    await mission_control.emit_tool_result(run_id, agent_name, "tasks.create_task", result)
    return result


async def get_task(run_id: str, agent_name: str, task_id: str) -> dict | None:
    """Get a single task by ID."""
    await mission_control.emit_tool_call(run_id, agent_name, "tasks.get_task",
                                         {"task_id": task_id})
    result = await db.get_task(task_id)
    await mission_control.emit_tool_result(run_id, agent_name, "tasks.get_task",
                                           result or {"error": "not found"})
    return result


async def update_task(run_id: str, agent_name: str, task_id: str, **fields) -> dict | None:
    """Update a task."""
    await mission_control.emit_tool_call(run_id, agent_name, "tasks.update_task",
                                         {"task_id": task_id, **fields})
    result = await db.update_task(task_id, **fields)
    await mission_control.emit_tool_result(run_id, agent_name, "tasks.update_task",
                                           result or {"error": "not found"})
    return result


async def complete_task(run_id: str, agent_name: str, task_id: str) -> dict | None:
    """Mark a task as completed."""
    await mission_control.emit_tool_call(run_id, agent_name, "tasks.complete_task",
                                         {"task_id": task_id})
    result = await db.complete_task(task_id)
    await mission_control.emit_tool_result(run_id, agent_name, "tasks.complete_task",
                                           result or {"error": "not found"})
    return result


async def get_overdue_tasks(run_id: str, agent_name: str) -> list[dict]:
    """Get all overdue tasks."""
    await mission_control.emit_tool_call(run_id, agent_name, "tasks.get_overdue", {})
    result = await db.get_overdue_tasks()
    await mission_control.emit_tool_result(run_id, agent_name, "tasks.get_overdue",
                                           {"count": len(result), "tasks": result})
    return result
