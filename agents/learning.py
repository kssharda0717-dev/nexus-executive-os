"""Learning Sub-Agent — Curriculum generation, study scheduling, spaced repetition.

This agent handles the "Research Reactor" use case:
- Generates structured learning curricula from a learning goal
- Creates tasks for each module with time estimates
- Schedules study sessions on the calendar
- Creates comprehensive curriculum notes
- Supports spaced repetition review scheduling
"""

from google.genai import types
from agents.base import BaseAgent
from tools import calendar_tool, tasks_tool, notes_tool


class LearningAgent(BaseAgent):
    name = "learning_agent"
    description = "Handles learning goals, skill development, study plans, and knowledge tracking"
    system_prompt = """You are the Learning Agent of NEXUS, an autonomous life operating system.

Your specialty is creating personalized, structured learning paths and ensuring users stay on track.

## Your capabilities:
- Generate detailed learning curricula for any topic
- Break curricula into modules with estimated study times
- Create tasks for each learning module
- Find free slots and schedule study sessions on the calendar
- Create comprehensive curriculum notes with resources
- Schedule spaced-repetition review sessions

## Your learning design philosophy:
- Start with fundamentals before advancing
- 45-60 minute focused study sessions (Pomodoro-friendly)
- Mix theory and practice (at least 40% hands-on)
- Build in review sessions using spaced repetition (1 day, 3 days, 7 days, 14 days after)
- Include clear milestones and checkpoints
- Link resources and exercises in notes

## Study scheduling rules:
- Prefer morning slots (9 AM - 12 PM) for learning new concepts
- Afternoon slots (2 PM - 5 PM) for practice and exercises
- Maximum 2 study sessions per day
- At least 1 rest day per week
- Use ISO 8601 datetime format always

## When creating a curriculum:
1. First create a master note with the full curriculum outline
2. Create tasks for each module with proper sequencing
3. Find free calendar slots and schedule study sessions
4. Link calendar events to their corresponding tasks

Today's date context will be provided in the user message."""

    def __init__(self, run_id: str):
        super().__init__(run_id)
        self.tools = [self._build_tools()]
        self.tool_handlers = {
            "list_events": calendar_tool.list_events,
            "create_event": calendar_tool.create_event,
            "find_free_slots": calendar_tool.find_free_slots,
            "list_tasks": tasks_tool.list_tasks,
            "create_task": tasks_tool.create_task,
            "update_task": tasks_tool.update_task,
            "list_notes": notes_tool.list_notes,
            "create_note": notes_tool.create_note,
            "search_notes": notes_tool.search_notes,
            "append_to_note": notes_tool.append_to_note,
        }

    def _build_tools(self):
        return types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="list_events",
                description="List calendar events to check existing schedule",
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
                description="Schedule a study session on the calendar",
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
                description="Find available study slots in the calendar",
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
                name="list_tasks",
                description="List existing tasks",
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
                description="Create a learning module task",
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
                description="Update a task (e.g., mark as scheduled)",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "task_id": types.Schema(type="STRING"),
                        "status": types.Schema(type="STRING"),
                        "due_date": types.Schema(type="STRING"),
                    },
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
                description="Create a curriculum note or study material",
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
                description="Search for existing study notes",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"query": types.Schema(type="STRING")},
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="append_to_note",
                description="Add content to an existing note (e.g., add review notes)",
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
