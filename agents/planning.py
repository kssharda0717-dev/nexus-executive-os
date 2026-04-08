"""Planning Sub-Agent — Week orchestration, schedule optimization, task prioritization.

This agent handles the "Monday Morning Miracle" use case:
- Reads all open tasks, notes tagged with goals, and calendar events
- Performs constraint-satisfaction scheduling
- Creates calendar blocks for each task
- Writes a structured week plan note
"""

from google.genai import types
from agents.base import BaseAgent
from tools import calendar_tool, tasks_tool, notes_tool


class PlanningAgent(BaseAgent):
    name = "planning_agent"
    description = "Handles scheduling, week planning, time management, and task prioritization"
    system_prompt = """You are the Planning Agent of NEXUS, an autonomous life operating system.

Your specialty is intelligent scheduling and time management. You think like a world-class executive assistant.

## Your capabilities:
- Read all open tasks, their priorities, deadlines, and time estimates
- Read all calendar events to understand existing commitments
- Read notes tagged with goals/projects for context
- Find free time slots in the calendar
- Create calendar events for scheduled task blocks
- Update task statuses to 'scheduled' after booking them
- Create comprehensive plan notes

## Your scheduling philosophy:
- Deep work (high-focus tasks) in the morning (9 AM - 12 PM)
- Meetings and collaborative work midday (12 PM - 2 PM)
- Administrative and low-energy tasks in the afternoon (2 PM - 5 PM)
- Always leave 15-minute buffer between events
- Respect existing calendar events — never double-book
- Prioritize urgent and high-priority tasks first
- Group similar tasks together when possible

## Important rules:
- Always use ISO 8601 datetime format
- When creating calendar events for tasks, link them with linked_task_id
- After scheduling a task, update its status to 'scheduled'
- Create a summary note with the full plan
- Be specific with times, not vague

Today's date context will be provided in the user message."""

    def __init__(self, run_id: str):
        super().__init__(run_id)
        self.tools = [self._build_tools()]
        self.tool_handlers = {
            "list_events": calendar_tool.list_events,
            "create_event": calendar_tool.create_event,
            "find_free_slots": calendar_tool.find_free_slots,
            "update_event": calendar_tool.update_event,
            "delete_event": calendar_tool.delete_event,
            "list_tasks": tasks_tool.list_tasks,
            "create_task": tasks_tool.create_task,
            "update_task": tasks_tool.update_task,
            "complete_task": tasks_tool.complete_task,
            "get_overdue_tasks": tasks_tool.get_overdue_tasks,
            "list_notes": notes_tool.list_notes,
            "create_note": notes_tool.create_note,
            "search_notes": notes_tool.search_notes,
        }

    def _build_tools(self):
        return types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="list_events",
                description="List calendar events in a date range to see existing commitments",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "start": types.Schema(type="STRING", description="ISO 8601 start"),
                        "end": types.Schema(type="STRING", description="ISO 8601 end"),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="create_event",
                description="Create a calendar event (use for scheduling task blocks)",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "title": types.Schema(type="STRING"),
                        "start_time": types.Schema(type="STRING", description="ISO 8601"),
                        "end_time": types.Schema(type="STRING", description="ISO 8601"),
                        "description": types.Schema(type="STRING"),
                        "linked_task_id": types.Schema(type="STRING", description="Task ID this event is for"),
                    },
                    required=["title", "start_time", "end_time"],
                ),
            ),
            types.FunctionDeclaration(
                name="find_free_slots",
                description="Find available time slots in the calendar",
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
                name="update_event",
                description="Update an existing calendar event",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "event_id": types.Schema(type="STRING"),
                        "title": types.Schema(type="STRING"),
                        "start_time": types.Schema(type="STRING"),
                        "end_time": types.Schema(type="STRING"),
                        "description": types.Schema(type="STRING"),
                    },
                    required=["event_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="delete_event",
                description="Delete a calendar event",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"event_id": types.Schema(type="STRING")},
                    required=["event_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="list_tasks",
                description="List tasks, optionally filtered",
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
                description="Create a new task",
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
                description="Update a task (e.g., set status to 'scheduled' after booking calendar time)",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "task_id": types.Schema(type="STRING"),
                        "title": types.Schema(type="STRING"),
                        "status": types.Schema(type="STRING"),
                        "priority": types.Schema(type="STRING"),
                        "due_date": types.Schema(type="STRING"),
                    },
                    required=["task_id"],
                ),
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
                name="get_overdue_tasks",
                description="Get all overdue tasks",
                parameters=types.Schema(type="OBJECT", properties={}),
            ),
            types.FunctionDeclaration(
                name="list_notes",
                description="List notes, optionally filtered by tags (e.g., goals, projects)",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "tags": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="create_note",
                description="Create a note (use for week plan summaries)",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "title": types.Schema(type="STRING"),
                        "content": types.Schema(type="STRING"),
                        "tags": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                        "linked_task_ids": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    },
                    required=["title", "content"],
                ),
            ),
            types.FunctionDeclaration(
                name="search_notes",
                description="Search notes by title or content",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"query": types.Schema(type="STRING")},
                    required=["query"],
                ),
            ),
        ])
