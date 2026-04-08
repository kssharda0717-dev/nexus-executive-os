"""Life Admin Sub-Agent — Life events, checklists, errand orchestration.

This agent handles the "Life Admin Assassin" use case:
- Detects life events (moving, travel, health appointments, etc.)
- Generates comprehensive checklists with smart deadlines
- Creates tasks for every item with relative-to-event scheduling
- Blocks calendar time for critical activities
- Creates structured plan notes
- Cross-references existing calendar for conflicts
"""

from google.genai import types
from agents.base import BaseAgent
from tools import calendar_tool, tasks_tool, notes_tool


class LifeAdminAgent(BaseAgent):
    name = "life_admin_agent"
    description = "Handles life events, errands, checklists, and day-to-day logistics"
    system_prompt = """You are the Life Admin Agent of NEXUS, an autonomous life operating system.

Your specialty is breaking down overwhelming life events into manageable, scheduled action items.

## Your capabilities:
- Detect the type of life event and generate comprehensive checklists
- Create tasks with smart deadlines relative to the event date
- Schedule critical activities on the calendar
- Cross-reference existing calendar for conflicts
- Create structured plan notes with phase breakdowns
- Prioritize tasks by urgency and importance

## Life event handling:
When a user mentions a life event (moving, travel, wedding, job change, etc.):
1. Generate a COMPREHENSIVE checklist (aim for 15-30+ items across categories)
2. Organize items into phases (e.g., "2 weeks before", "1 week before", "day of", "after")
3. Assign smart deadlines relative to the event date
4. Set appropriate priorities (urgent for time-sensitive items)
5. Create calendar blocks for critical activities
6. Flag any conflicts with existing calendar events
7. Create a master plan note with the full breakdown

## Task creation rules:
- Use descriptive titles that include the context (e.g., "Moving: Cancel internet service")
- Set due dates relative to the event (e.g., event is April 15 → "Cancel internet" due April 8)
- Tag all tasks with the event type (e.g., "moving", "travel")
- Set estimated_minutes for each task
- Use priority levels wisely: urgent (must happen on specific date), high (this week), medium (flexible)

## Important:
- Always use ISO 8601 datetime format
- Be thorough — it's better to over-prepare than under-prepare
- Group related tasks together
- Think about dependencies between tasks

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
            "list_notes": notes_tool.list_notes,
            "create_note": notes_tool.create_note,
            "search_notes": notes_tool.search_notes,
            "append_to_note": notes_tool.append_to_note,
        }

    def _build_tools(self):
        return types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="list_events",
                description="Check existing calendar for conflicts and commitments",
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
                description="Block calendar time for critical activities",
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
                description="Find available time for scheduling activities",
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
                description="Remove a conflicting event",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"event_id": types.Schema(type="STRING")},
                    required=["event_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="list_tasks",
                description="Check existing tasks",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "status": types.Schema(type="STRING"),
                        "priority": types.Schema(type="STRING"),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="create_task",
                description="Create a checklist task with deadline and priority",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "title": types.Schema(type="STRING"),
                        "description": types.Schema(type="STRING"),
                        "priority": types.Schema(type="STRING"),
                        "due_date": types.Schema(type="STRING"),
                        "estimated_minutes": types.Schema(type="INTEGER"),
                        "tags": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                        "dependencies": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    },
                    required=["title"],
                ),
            ),
            types.FunctionDeclaration(
                name="update_task",
                description="Update a task's status or details",
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
                description="Check for overdue tasks that might conflict",
                parameters=types.Schema(type="OBJECT", properties={}),
            ),
            types.FunctionDeclaration(
                name="list_notes",
                description="Check existing notes for related plans",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "tags": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="create_note",
                description="Create a comprehensive plan note for the life event",
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
                description="Search for related notes",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"query": types.Schema(type="STRING")},
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="append_to_note",
                description="Add information to an existing plan note",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "note_id": types.Schema(type="STRING"),
                        "content": types.Schema(type="STRING"),
                    },
                    required=["note_id", "content"],
                ),
            ),
        ])
