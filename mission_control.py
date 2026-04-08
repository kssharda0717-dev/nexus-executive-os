"""Mission Control — Real-time WebSocket broadcast of agent internals.

Every agent thought, tool call, and result is streamed to all connected
Mission Control clients. This gives the frontend (and judges) full
visibility into the agent mesh's internal reasoning.

Events have two layers:
- Technical: raw tool calls, args, results (for developers)
- Narration: personable, human-friendly status messages (for demos/judges)
"""

import asyncio
import json
import logging
from datetime import datetime
from fastapi import WebSocket
from schemas import StepType

logger = logging.getLogger("nexus.mission_control")


class MissionControl:
    """Manages WebSocket connections and broadcasts agent events."""

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._event_log: list[dict] = []
        self._max_log_size = 500

    async def connect(self, websocket: WebSocket):
        """Accept a new Mission Control client."""
        await websocket.accept()
        self._connections.append(websocket)
        # Send recent event history so new clients get context
        for event in self._event_log[-50:]:
            try:
                await websocket.send_json(event)
            except Exception:
                break

    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected client."""
        if websocket in self._connections:
            self._connections.remove(websocket)

    @property
    def client_count(self) -> int:
        return len(self._connections)

    async def broadcast(self, run_id: str, agent_name: str,
                        event_type: StepType | str, content: dict,
                        metadata: dict = None, narration: str = None):
        """Broadcast an event to all connected Mission Control clients."""
        event = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "event_type": event_type if isinstance(event_type, str) else event_type.value,
            "content": content,
            "metadata": metadata or {},
        }
        # Personable narration layer — always present for frontend consumption
        if narration:
            event["narration"] = narration

        # Structured logging for production observability
        logger.info(f"[{agent_name}] {event['event_type']}: "
                    f"{narration or json.dumps(content, default=str)[:200]}")

        # Buffer for reconnecting clients
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size:]

        # Broadcast to all connected clients
        disconnected = []
        for ws in self._connections:
            try:
                await ws.send_json(event)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

    # ── Convenience methods with auto-narration ────────

    async def emit_thought(self, run_id: str, agent_name: str, thought: str):
        await self.broadcast(run_id, agent_name, StepType.THOUGHT,
                             {"thought": thought},
                             narration=thought)

    async def emit_narration(self, run_id: str, agent_name: str, message: str):
        """Emit a pure narration event — human-friendly status for the frontend."""
        await self.broadcast(run_id, agent_name, "narration",
                             {"message": message},
                             narration=message)

    async def emit_tool_call(self, run_id: str, agent_name: str,
                             tool_name: str, arguments: dict):
        await self.broadcast(run_id, agent_name, StepType.TOOL_CALL,
                             {"tool": tool_name, "arguments": arguments})

    async def emit_tool_result(self, run_id: str, agent_name: str,
                               tool_name: str, result: dict):
        # Generate a human-readable summary of the result
        narration = self._narrate_tool_result(tool_name, result)
        await self.broadcast(run_id, agent_name, StepType.TOOL_RESULT,
                             {"tool": tool_name, "result": result},
                             narration=narration)

    async def emit_delegation(self, run_id: str, from_agent: str,
                              to_agent: str, task: str):
        from agents.base import AGENT_FRIENDLY_NAMES
        friendly_from = AGENT_FRIENDLY_NAMES.get(from_agent, from_agent)
        friendly_to = AGENT_FRIENDLY_NAMES.get(to_agent, to_agent)
        narration = f"{friendly_from} is handing this off to {friendly_to}..."
        await self.broadcast(run_id, from_agent, StepType.DELEGATION,
                             {"delegated_to": to_agent, "task": task},
                             narration=narration)

    async def emit_error(self, run_id: str, agent_name: str, error: str):
        await self.broadcast(run_id, agent_name, StepType.ERROR,
                             {"error": error},
                             narration=f"Hit a snag: {error}")

    async def emit_plan(self, run_id: str, agent_name: str, plan: list[dict]):
        step_summary = ", ".join(s.get("action", "?") for s in plan[:5])
        narration = f"I've mapped out a {len(plan)}-step game plan: {step_summary}"
        await self.broadcast(run_id, agent_name, StepType.PLAN,
                             {"plan": plan},
                             narration=narration)

    async def emit_final_answer(self, run_id: str, agent_name: str, answer: str):
        await self.broadcast(run_id, agent_name, StepType.FINAL_ANSWER,
                             {"answer": answer},
                             narration="Here's what I've put together for you.")

    # ── Private helpers ────────────────────────────────

    @staticmethod
    def _narrate_tool_result(tool_name: str, result: dict) -> str:
        """Generate a human-friendly summary of a tool result."""
        if "error" in result:
            return f"Hmm, {tool_name} ran into an issue."

        if tool_name == "list_events":
            count = result.get("count", len(result.get("items", [])))
            return f"Found {count} event{'s' if count != 1 else ''} on the calendar."
        elif tool_name == "create_event":
            title = result.get("title", "an event")
            return f"Booked '{title}' on your calendar."
        elif tool_name == "list_tasks":
            count = result.get("count", len(result.get("items", [])))
            return f"Found {count} task{'s' if count != 1 else ''}."
        elif tool_name == "create_task":
            title = result.get("title", "a task")
            return f"Created task: '{title}'."
        elif tool_name == "complete_task":
            return "Marked a task as complete!"
        elif tool_name == "update_task":
            return "Updated the task."
        elif tool_name == "get_overdue_tasks":
            count = result.get("count", len(result.get("items", [])))
            if count == 0:
                return "No overdue tasks. You're all caught up!"
            return f"Found {count} overdue task{'s' if count != 1 else ''}."
        elif tool_name == "find_free_slots":
            count = result.get("count", len(result.get("slots", [])))
            return f"Found {count} available time slot{'s' if count != 1 else ''}."
        elif tool_name == "create_note":
            title = result.get("title", "a note")
            return f"Created note: '{title}'."
        elif tool_name == "search_notes":
            count = result.get("count", len(result.get("items", [])))
            return f"Found {count} matching note{'s' if count != 1 else ''}."
        elif tool_name == "list_notes":
            count = result.get("count", len(result.get("items", [])))
            return f"Found {count} note{'s' if count != 1 else ''}."
        else:
            return f"Completed {tool_name}."


# Singleton instance
mission_control = MissionControl()
