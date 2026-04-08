"""Pydantic schemas for NEXUS API and internal data models."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from uuid import uuid4


# ── Enums ──────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskUrgency(str, Enum):
    """Urgency DNA — controls reminder escalation cadence."""
    ASYNC = "async"         # Green: No pings, user checks at leisure
    STANDARD = "standard"   # Amber: T-2h gentle nudge
    CRITICAL = "critical"   # Red: T-4h, T-1h, T-15m escalating alerts


class AgentRunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StepType(str, Enum):
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    DELEGATION = "delegation"
    ERROR = "error"
    PLAN = "plan"
    FINAL_ANSWER = "final_answer"


# ── Calendar ───────────────────────────────────────────

class EventCreate(BaseModel):
    title: str
    description: Optional[str] = None
    start_time: str  # ISO 8601
    end_time: str
    linked_task_id: Optional[str] = None


class EventUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    linked_task_id: Optional[str] = None


class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: Optional[str] = None
    start_time: str
    end_time: str
    linked_task_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ── Tasks ──────────────────────────────────────────────

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    urgency: TaskUrgency = TaskUrgency.STANDARD
    due_date: Optional[str] = None
    estimated_minutes: Optional[int] = None
    tags: Optional[list[str]] = None
    dependencies: Optional[list[str]] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    urgency: Optional[TaskUrgency] = None
    due_date: Optional[str] = None
    estimated_minutes: Optional[int] = None
    tags: Optional[list[str]] = None
    dependencies: Optional[list[str]] = None


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    urgency: TaskUrgency = TaskUrgency.STANDARD
    due_date: Optional[str] = None
    estimated_minutes: Optional[int] = None
    tags: Optional[list[str]] = None
    dependencies: Optional[list[str]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ── Notes ──────────────────────────────────────────────

class NoteCreate(BaseModel):
    title: str
    content: str
    tags: Optional[list[str]] = None
    linked_task_ids: Optional[list[str]] = None


class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[list[str]] = None
    linked_task_ids: Optional[list[str]] = None


class Note(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    content: str
    tags: Optional[list[str]] = None
    linked_task_ids: Optional[list[str]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ── Agent State ────────────────────────────────────────

class AgentRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_message: str
    intent: Optional[str] = None
    status: AgentRunStatus = AgentRunStatus.RUNNING
    plan: Optional[dict] = None
    result: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class AgentStep(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    agent_name: str
    step_type: StepType
    content: dict
    created_at: Optional[str] = None


# ── API Request/Response ───────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    run_id: str
    response: str
    steps_count: int
    tools_used: list[str]


# ── Mission Control Events ─────────────────────────────

class MissionControlEvent(BaseModel):
    run_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    agent_name: str
    event_type: StepType
    content: dict
    metadata: Optional[dict] = None
