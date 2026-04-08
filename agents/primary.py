"""Primary Agent — The NEXUS Orchestrator.

Responsibilities:
1. Classify user intent into domains (planning, learning, life_admin, general)
2. For complex requests: generate a multi-step execution plan
3. Delegate to the appropriate sub-agent with step-level checkpointing
4. Resume interrupted plans from the last successful step
5. Synthesize results from sub-agents into a cohesive response
"""

import json
import logging
from datetime import datetime
from google.genai import types

from config import MAX_PLAN_STEPS
from mission_control import mission_control
from agents.base import AGENT_FRIENDLY_NAMES
from llm_router import llm_router, AllProvidersExhausted
import database as db

logger = logging.getLogger("nexus.primary")


# ── Intent Classification ──────────────────────────────

INTENT_SYSTEM_PROMPT = """You are the intent classifier for NEXUS, an autonomous life operating system.

Classify the user's message into exactly ONE of these intents:
- "planning": Scheduling, week planning, time management, calendar optimization, task prioritization
- "learning": Learning goals, skill development, study plans, curriculum building, knowledge tracking
- "life_admin": Life events (moving, travel, health), errands, checklists, administrative tasks, day-to-day logistics
- "general": Simple queries, task/note/event CRUD, status checks, questions about existing data, web searches, current events, LinkedIn/networking requests, real-time information lookups, sending Slack messages, checking email/Outlook inbox, accepting/declining meeting invites, Outlook calendar queries

Also assess complexity:
- "simple": Can be handled with 1-2 tool calls (e.g., "create a task", "what's on my calendar tomorrow")
- "complex": Requires multi-step planning, cross-tool synthesis, or proactive scheduling (e.g., "plan my week", "I'm moving apartments")

Respond with ONLY valid JSON:
{"intent": "<intent>", "complexity": "<simple|complex>", "summary": "<one-line summary of what the user wants>"}"""


async def classify_intent(run_id: str, user_message: str) -> dict:
    """Classify user intent using Gemini Flash with retry."""
    await mission_control.emit_thought(run_id, "primary_agent",
                                       f"Classifying intent for: {user_message[:100]}")

    # Truncate to save TPM — intent classification only needs the gist
    trimmed_message = user_message[:500]

    try:
        response = await llm_router.generate_content(
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=trimmed_message)])],
            system_instruction=INTENT_SYSTEM_PROMPT,
            temperature=0.0,
            agent_name="primary_agent",
            run_id=run_id,
        )
    except AllProvidersExhausted:
        # ── Deterministic fallback: default to simple/general ──
        logger.warning(f"[classify_intent] All providers exhausted — deterministic fallback")
        result = {"intent": "general", "complexity": "simple", "summary": user_message[:100]}
        await mission_control.emit_narration(
            run_id, "primary_agent",
            "Using built-in intelligence to classify your request..."
        )
        await db.log_agent_step(run_id, "primary_agent", "thought",
                                {"classification": result, "mode": "deterministic"})
        return result

    # ── Defensive: response.text may be None if response was blocked/empty ──
    raw_text = getattr(response, "text", None) or ""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text) if text else {}
        # Validate required keys exist
        if "intent" not in result or "complexity" not in result:
            raise ValueError("Missing required keys")
    except (json.JSONDecodeError, ValueError):
        result = {"intent": "general", "complexity": "simple", "summary": user_message[:100]}

    await mission_control.emit_narration(
        run_id, "primary_agent",
        f"I understand — this is a {result['complexity']} {result['intent']} request: {result['summary']}"
    )
    await db.log_agent_step(run_id, "primary_agent", "thought", {"classification": result})
    return result


# ── Plan Generation ────────────────────────────────────

PLAN_SYSTEM_PROMPT = """You are the strategic planner for NEXUS, an autonomous life operating system.

Given the user's request and its classified intent, generate a step-by-step execution plan.
Each step should specify:
- "step": Step number
- "agent": Which sub-agent handles it ("planning_agent", "learning_agent", "life_admin_agent")
- "action": What the agent should do (be specific and actionable)
- "tools_needed": Which tools will likely be used (calendar, tasks, notes)

CRITICAL CONSTRAINT: Generate AT MOST 3 steps. Combine related operations into a single
step where possible. Each step can use multiple tools internally — you do NOT need a
separate step per tool call. Fewer steps = faster execution.

Respond with ONLY valid JSON:
{"plan": [{"step": 1, "agent": "...", "action": "...", "tools_needed": ["..."]}, ...]}"""


async def generate_plan(run_id: str, user_message: str, classification: dict) -> list[dict]:
    """Generate a multi-step execution plan with retry."""
    await mission_control.emit_narration(
        run_id, "primary_agent",
        "Let me think about the best approach for this..."
    )

    prompt = f"""User request: {user_message[:500]}
Classification: {json.dumps(classification)}

Generate a detailed step-by-step plan."""

    try:
        response = await llm_router.generate_content(
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            system_instruction=PLAN_SYSTEM_PROMPT,
            temperature=0.1,
            agent_name="primary_agent",
            run_id=run_id,
        )
    except AllProvidersExhausted:
        # ── Deterministic fallback: single-step plan using the classified agent ──
        logger.warning(f"[generate_plan] All providers exhausted — deterministic single-step plan")
        plan = [{"step": 1, "agent": f"{classification['intent']}_agent",
                 "action": user_message, "tools_needed": ["tasks", "calendar", "notes"]}]
        await mission_control.emit_narration(
            run_id, "primary_agent",
            "Constructing a streamlined plan using built-in strategy..."
        )
        await mission_control.emit_plan(run_id, "primary_agent", plan)
        await db.log_agent_step(run_id, "primary_agent", "plan",
                                {"plan": plan, "mode": "deterministic"})
        await db.update_agent_run(run_id, plan={"steps": plan, "completed_steps": []},
                                  intent=classification["intent"])
        return plan

    # ── Defensive: response.text may be None if response was blocked/empty ──
    raw_text = getattr(response, "text", None) or ""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text) if text else {}
        plan = result.get("plan", [])
    except json.JSONDecodeError:
        plan = [{"step": 1, "agent": f"{classification['intent']}_agent",
                 "action": user_message, "tools_needed": ["tasks", "calendar", "notes"]}]

    # ── Plan Flattening: hard-cap steps to reduce Gemini call count ──
    if len(plan) > MAX_PLAN_STEPS:
        logger.warning(f"Plan had {len(plan)} steps — truncating to {MAX_PLAN_STEPS}")
        plan = plan[:MAX_PLAN_STEPS]

    await mission_control.emit_plan(run_id, "primary_agent", plan)
    await db.log_agent_step(run_id, "primary_agent", "plan", {"plan": plan})
    await db.update_agent_run(run_id, plan={"steps": plan, "completed_steps": []},
                              intent=classification["intent"])

    return plan


# ── Sub-Agent Registry ─────────────────────────────────

def get_sub_agent(agent_name: str, run_id: str):
    """Get a sub-agent instance by name."""
    from agents.planning import PlanningAgent
    from agents.learning import LearningAgent
    from agents.life_admin import LifeAdminAgent

    registry = {
        "planning_agent": PlanningAgent,
        "learning_agent": LearningAgent,
        "life_admin_agent": LifeAdminAgent,
    }

    agent_class = registry.get(agent_name)
    if agent_class:
        return agent_class(run_id)
    return None


# ── General Agent ──────────────────────────────────────

from agents.base import BaseAgent
from tools import calendar_tool, tasks_tool, notes_tool
from tools import web_search as web_search_tool
from tools import linkedin_agent as linkedin_tool
from tools import document_generator as doc_gen_tool
from mcp_executive import mcp_bridge


class GeneralAgent(BaseAgent):
    """Handles simple, direct queries that don't need specialized sub-agents."""

    name = "general_agent"
    description = "Handles simple queries: CRUD on tasks, events, notes, status checks"
    system_prompt = """You are NEXUS, an autonomous life operating system. You help users manage their tasks, calendar, notes, and connect them to global intelligence.

For the current request, use the available tools to fulfill it directly. Be concise and helpful.
When creating items, confirm what was created. When listing, format results clearly.

═══ MANDATORY CHAIN-OF-THOUGHT (CoT) — EXECUTE BEFORE EVERY TOOL CALL ═══
Before calling ANY tool, you MUST reason through these 4 gates IN ORDER:

Gate 1 — CURRENT STATE:
  "What is the [SYSTEM_TIME]? What does the [ACTION SNAPSHOT] show?"

Gate 2 — INTENT AUDIT:
  "Is this a NEW command or a CORRECTION of something I just did?"
  "Does the date/time make sense relative to [SYSTEM_TIME]?"

Gate 3 — SANITY CHECK:
  "Am I about to create a DUPLICATE of something in the snapshot?"
  "Is the date I'm using in the PAST? If so, I must ask the user."
  "Is the end_time BEFORE the start_time? If so, I must stop."

Gate 4 — FINAL DECISION:
  "Execute [tool] because it passes all logic gates."
  OR "BLOCK — ask the user for clarification because [reason]."

═══ TEMPORAL GROUNDING (CRITICAL) ═══
The [SYSTEM_TIME] header tells you the EXACT current date and time.
- If a user provides a date that is MATHEMATICALLY IN THE PAST compared to SYSTEM_TIME:
  → You MUST interrupt and ask: "I noticed [date] has already passed. Did you mean [date] of next year, or was that a mistake?"
  → NEVER silently create past-dated items.
- If end_time is before start_time → BLOCK and ask for correction.

═══ REFLECTION MANDATE ═══
Before executing ANY tool, you MUST check:
1. Read the [ACTION SNAPSHOT]. It shows the last 3 things you created/updated.
2. Ask: "Is the user REFINING a previous thought, or starting NEW?"
3. If refining → FORBIDDEN from creating a duplicate → find the ID → use update.
4. If new → proceed with create.

═══ CORRECTION PATTERNS (FEW-SHOT) ═══

Pattern A — The Date Correction:
  User: "No, April 8th" / "Actually make it Tuesday" / "Change the date"
  → CoT: "Gate 1: SYSTEM_TIME is Monday April 6. Snapshot shows create_task ID=abc Title='Meeting prep'."
  → CoT: "Gate 2: This is a CORRECTION — user said 'No'. Gate 3: Not a duplicate. Gate 4: Execute update."
  → Action: update_task(task_id="abc", due_date="2026-04-08")

Pattern B — The Ambiguity:
  User: "Change that to high priority"
  → CoT: "Gate 1: Snapshot has 2 items. Gate 2: User said 'that' — ambiguous reference."
  → If 1 recent item: update directly. If multiple: ask which one.

Pattern C — Past-Date Block:
  User: "Schedule a meeting for March 15th"
  → CoT: "Gate 1: SYSTEM_TIME is April 6, 2026. Gate 2: New command. Gate 3: March 15 is in the PAST."
  → Response: "I noticed March 15th has already passed. Did you mean March 15th 2027, or a different date?"

═══ TOOL SELECTION ═══
- If the user asks about current events, news, stock prices, weather, or anything NOT in the local Vault, use the `web_search` tool.
- If the user asks to find people, professionals, recruiters, or LinkedIn profiles, use the `linkedin_search` tool.
- For LinkedIn results, present them in a clean Markdown table with Name, Role, Company, and Profile link columns, followed by personalized conversation starters.
- For web search results, synthesize the findings into a clear, concise answer.

═══ DOCUMENT GENERATION ═══
- If the user asks to export, download, or generate a file (PDF, Word doc, CSV, Excel), use `generate_document`.
- Supported formats: "pdf", "docx", "csv", "xlsx".
- For tabular data (lists, search results), CSV or XLSX works best. For reports, use PDF or DOCX.

═══ MCP EXECUTIVE BRIDGE (Slack & Outlook) ═══
- If the user asks to send a Slack message, post to a channel, or notify someone on Slack → use `send_slack_message`.
- **File Uploads:** If the user says "send as PDF", "share the report as a Word doc", or requests a specific format to Slack, set `file_format` (pdf/docx/csv/xlsx) and `file_title`. The file will be auto-generated and uploaded as an attachment instead of a text wall.
- If the user asks to check email, see unread messages, or get an inbox summary → use `fetch_outlook_emails`.
- If the user asks to accept or decline a meeting invite → use `respond_to_outlook_invite`.
- If the user asks about their Outlook/work calendar or external meetings → use `fetch_outlook_calendar`.
- These tools connect to real Slack and Outlook APIs via the MCP Executive Bridge.
- If a tool returns {"status": "disabled"}, tell the user: "Please connect your [Service] account via the Neural Link section first."
- If a tool returns {"status": "unauthenticated"}, tell the user: "Your [Service] account needs to be re-authenticated. Please click 'Connect' in the Neural Link panel."

═══ MULTI-STEP REASONING (ReAct LOOP) ═══
You are NOT limited to a single tool call. When a request requires multiple steps, you MUST continue reasoning and calling tools until the task is FULLY complete.

Examples of multi-step chains:
- "Check my email and send a summary to Slack" → fetch_outlook_emails → send_slack_message
- "Create a task for the meeting invite and accept it" → create_task → respond_to_outlook_invite
- "Find free slots and create an event" → find_free_slots → create_event

After each tool call, evaluate: "Is the user's request FULLY satisfied?" If not, make the next tool call.
Never stop after one tool call if the user's intent clearly requires more steps.

═══ PERSONALIZED PATTERNS ═══
If [PERSONALIZED PATTERNS] appear at the top of the message, these are LEARNED from the user's correction history. Follow them strictly.

Today's date context will be provided in the user message. Always use ISO 8601 format for dates."""

    def __init__(self, run_id: str):
        super().__init__(run_id)
        self.tools = [self._build_tools()]
        self.tool_handlers = {
            "list_events": calendar_tool.list_events,
            "create_event": calendar_tool.create_event,
            "update_event": calendar_tool.update_event,
            "delete_event": calendar_tool.delete_event,
            "find_free_slots": calendar_tool.find_free_slots,
            "list_tasks": tasks_tool.list_tasks,
            "create_task": tasks_tool.create_task,
            "update_task": tasks_tool.update_task,
            "complete_task": tasks_tool.complete_task,
            "get_overdue_tasks": tasks_tool.get_overdue_tasks,
            "list_notes": notes_tool.list_notes,
            "create_note": notes_tool.create_note,
            "update_note": notes_tool.update_note,
            "search_notes": notes_tool.search_notes,
            "append_to_note": notes_tool.append_to_note,
            "web_search": web_search_tool.web_search,
            "linkedin_search": linkedin_tool.linkedin_search,
            "generate_document": doc_gen_tool.generate_document,
            # ── MCP Executive Bridge tools ──
            "send_slack_message": self._mcp_send_slack,
            "fetch_outlook_emails": self._mcp_fetch_emails,
            "respond_to_outlook_invite": self._mcp_respond_invite,
            "fetch_outlook_calendar": self._mcp_fetch_outlook_calendar,
        }

    # ── MCP Bridge wrappers ───────────────────────────────
    # Each wrapper:
    #   1. Accepts (self, run_id, agent_name, **tool_kwargs) to match BaseAgent._execute_tool dispatch
    #   2. Checks bridge availability before calling, returning a helpful user-facing message
    #   3. Delegates only the tool-specific kwargs to the bridge method

    async def _mcp_send_slack(self, run_id, agent_name, text: str, channel: str = None,
                              file_format: str = None, file_title: str = None) -> dict:
        """Send a Slack message via the MCP Executive Bridge.

        If file_format is specified, generates the document first then uploads
        it as a Slack file attachment instead of sending a long text wall.
        """
        if not mcp_bridge or not mcp_bridge.slack.enabled:
            return {"status": "disabled",
                    "message": "Please connect your Slack account via the Neural Link section first."}

        # ── File generation + upload flow ──
        if file_format:
            doc_result = await doc_gen_tool.generate_document(
                run_id, agent_name,
                content=text,
                format=file_format,
                title=file_title or "NEXUS Report",
            )
            if doc_result.get("status") == "ok":
                return await mcp_bridge.slack.send_message(
                    text=f"📎 Here is your {file_format.upper()} file: *{file_title or 'NEXUS Report'}*",
                    channel=channel,
                    file_path=doc_result["file_path"],
                    file_title=file_title or doc_result.get("filename", "report"),
                )
            return doc_result

        # ── Standard message flow (auto Block Kit formatting) ──
        return await mcp_bridge.slack.send_message(text=text, channel=channel)

    async def _mcp_fetch_emails(self, run_id, agent_name, top: int = 10) -> dict:
        """Fetch unread Outlook emails via the MCP Executive Bridge."""
        if not mcp_bridge or not mcp_bridge.outlook.enabled:
            return {"status": "disabled",
                    "message": "Please connect your Microsoft Outlook account via the Neural Link section first."}
        result = await mcp_bridge.outlook.fetch_unread_emails(top=top)
        if result.get("status") == "unauthenticated":
            result["message"] = "Your Outlook session has expired. Please reconnect via the Neural Link section."
        return result

    async def _mcp_respond_invite(self, run_id, agent_name, event_id: str, action: str, comment: str = "") -> dict:
        """Respond to an Outlook calendar invite via the MCP Executive Bridge."""
        if not mcp_bridge or not mcp_bridge.outlook.enabled:
            return {"status": "disabled",
                    "message": "Please connect your Microsoft Outlook account via the Neural Link section first."}
        result = await mcp_bridge.outlook.respond_to_invite(event_id=event_id, action=action, comment=comment)
        if result.get("status") == "unauthenticated":
            result["message"] = "Your Outlook session has expired. Please reconnect via the Neural Link section."
        return result

    async def _mcp_fetch_outlook_calendar(self, run_id, agent_name, start: str = None, end: str = None) -> dict:
        """Fetch Outlook calendar events via the MCP Executive Bridge."""
        if not mcp_bridge or not mcp_bridge.outlook.enabled:
            return {"status": "disabled",
                    "message": "Please connect your Microsoft Outlook account via the Neural Link section first."}
        result = await mcp_bridge.outlook.fetch_calendar_events(start=start, end=end)
        if result.get("status") == "unauthenticated":
            result["message"] = "Your Outlook session has expired. Please reconnect via the Neural Link section."
        return result

    def _build_tools(self):
        return types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="list_events",
                description="List calendar events, optionally filtered by date range",
                parameters=types.Schema(type="OBJECT", properties={
                    "start": types.Schema(type="STRING", description="ISO 8601 start filter"),
                    "end": types.Schema(type="STRING", description="ISO 8601 end filter"),
                }),
            ),
            types.FunctionDeclaration(
                name="create_event",
                description="Create a new calendar event",
                parameters=types.Schema(type="OBJECT", properties={
                    "title": types.Schema(type="STRING", description="Event title"),
                    "start_time": types.Schema(type="STRING", description="ISO 8601 start"),
                    "end_time": types.Schema(type="STRING", description="ISO 8601 end"),
                    "description": types.Schema(type="STRING", description="Event description"),
                    "linked_task_id": types.Schema(type="STRING", description="Link to a task ID"),
                }, required=["title", "start_time", "end_time"]),
            ),
            types.FunctionDeclaration(
                name="update_event",
                description="Update an existing calendar event",
                parameters=types.Schema(type="OBJECT", properties={
                    "event_id": types.Schema(type="STRING", description="Event ID to update"),
                    "title": types.Schema(type="STRING"),
                    "start_time": types.Schema(type="STRING"),
                    "end_time": types.Schema(type="STRING"),
                    "description": types.Schema(type="STRING"),
                }, required=["event_id"]),
            ),
            types.FunctionDeclaration(
                name="delete_event",
                description="Delete a calendar event",
                parameters=types.Schema(type="OBJECT", properties={
                    "event_id": types.Schema(type="STRING", description="Event ID to delete"),
                }, required=["event_id"]),
            ),
            types.FunctionDeclaration(
                name="find_free_slots",
                description="Find available time slots in the calendar",
                parameters=types.Schema(type="OBJECT", properties={
                    "start": types.Schema(type="STRING", description="ISO 8601 search window start"),
                    "end": types.Schema(type="STRING", description="ISO 8601 search window end"),
                    "duration_minutes": types.Schema(type="INTEGER", description="Minimum slot duration"),
                }, required=["start", "end"]),
            ),
            types.FunctionDeclaration(
                name="list_tasks",
                description="List tasks, optionally filtered by status/priority/due date",
                parameters=types.Schema(type="OBJECT", properties={
                    "status": types.Schema(type="STRING", description="pending/scheduled/in_progress/completed/cancelled"),
                    "priority": types.Schema(type="STRING", description="low/medium/high/urgent"),
                    "due_before": types.Schema(type="STRING", description="ISO 8601 date"),
                }),
            ),
            types.FunctionDeclaration(
                name="create_task",
                description="Create a new task",
                parameters=types.Schema(type="OBJECT", properties={
                    "title": types.Schema(type="STRING", description="Task title"),
                    "description": types.Schema(type="STRING", description="Task description"),
                    "priority": types.Schema(type="STRING", description="low/medium/high/urgent"),
                    "due_date": types.Schema(type="STRING", description="ISO 8601 due date"),
                    "estimated_minutes": types.Schema(type="INTEGER", description="Estimated time"),
                    "tags": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="Tags"),
                }, required=["title"]),
            ),
            types.FunctionDeclaration(
                name="update_task",
                description="Update an existing task",
                parameters=types.Schema(type="OBJECT", properties={
                    "task_id": types.Schema(type="STRING", description="Task ID to update"),
                    "title": types.Schema(type="STRING"),
                    "status": types.Schema(type="STRING"),
                    "priority": types.Schema(type="STRING"),
                    "due_date": types.Schema(type="STRING"),
                }, required=["task_id"]),
            ),
            types.FunctionDeclaration(
                name="complete_task",
                description="Mark a task as completed",
                parameters=types.Schema(type="OBJECT", properties={
                    "task_id": types.Schema(type="STRING", description="Task ID to complete"),
                }, required=["task_id"]),
            ),
            types.FunctionDeclaration(
                name="get_overdue_tasks",
                description="Get all tasks past their due date",
                parameters=types.Schema(type="OBJECT", properties={}),
            ),
            types.FunctionDeclaration(
                name="list_notes",
                description="List all notes, optionally filtered by tags",
                parameters=types.Schema(type="OBJECT", properties={
                    "tags": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="Filter by tags"),
                }),
            ),
            types.FunctionDeclaration(
                name="create_note",
                description="Create a new note",
                parameters=types.Schema(type="OBJECT", properties={
                    "title": types.Schema(type="STRING", description="Note title"),
                    "content": types.Schema(type="STRING", description="Note content (markdown)"),
                    "tags": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="Tags"),
                    "linked_task_ids": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="Linked task IDs"),
                }, required=["title", "content"]),
            ),
            types.FunctionDeclaration(
                name="update_note",
                description="Update an existing note",
                parameters=types.Schema(type="OBJECT", properties={
                    "note_id": types.Schema(type="STRING", description="Note ID to update"),
                    "title": types.Schema(type="STRING"),
                    "content": types.Schema(type="STRING"),
                    "tags": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                }, required=["note_id"]),
            ),
            types.FunctionDeclaration(
                name="search_notes",
                description="Search notes by title or content",
                parameters=types.Schema(type="OBJECT", properties={
                    "query": types.Schema(type="STRING", description="Search query"),
                }, required=["query"]),
            ),
            types.FunctionDeclaration(
                name="append_to_note",
                description="Append content to an existing note",
                parameters=types.Schema(type="OBJECT", properties={
                    "note_id": types.Schema(type="STRING", description="Note ID"),
                    "content": types.Schema(type="STRING", description="Content to append"),
                }, required=["note_id", "content"]),
            ),
            types.FunctionDeclaration(
                name="web_search",
                description="Search the live web for current events, news, stock prices, or any data not in the local Vault. Use this when the user asks about recent events, real-time data, or information you don't have internally.",
                parameters=types.Schema(type="OBJECT", properties={
                    "query": types.Schema(type="STRING", description="Search query"),
                    "max_results": types.Schema(type="INTEGER", description="Number of results (1-10, default 5)"),
                }, required=["query"]),
            ),
            types.FunctionDeclaration(
                name="linkedin_search",
                description="Search for LinkedIn profiles and generate personalized outreach messages. Use when the user wants to find professionals, recruiters, or network contacts.",
                parameters=types.Schema(type="OBJECT", properties={
                    "query": types.Schema(type="STRING", description="Search criteria (e.g. 'HCM recruiters in London')"),
                    "user_context": types.Schema(type="STRING", description="User's background for personalized outreach drafting"),
                    "max_results": types.Schema(type="INTEGER", description="Number of profiles (1-10, default 5)"),
                }, required=["query"]),
            ),
            types.FunctionDeclaration(
                name="generate_document",
                description="Convert Markdown content into a downloadable document file (PDF, DOCX, CSV, or Excel). Use when the user wants to export data, create a report file, or generate a document for sharing.",
                parameters=types.Schema(type="OBJECT", properties={
                    "content": types.Schema(type="STRING", description="Markdown text content to convert"),
                    "format": types.Schema(type="STRING", description="Output format: 'pdf', 'docx', 'csv', or 'xlsx'"),
                    "title": types.Schema(type="STRING", description="Document title"),
                }, required=["content", "format"]),
            ),
            # ── MCP Executive Bridge tools ──
            types.FunctionDeclaration(
                name="send_slack_message",
                description="Send a message to a Slack channel via the MCP Executive Bridge. Use when the user asks to send a Slack message, notify a channel, or post an update. If the user asks for a specific file format (PDF, DOCX, CSV, XLSX), set file_format to auto-generate and upload the file instead of sending a text wall.",
                parameters=types.Schema(type="OBJECT", properties={
                    "text": types.Schema(type="STRING", description="The message text to send (or content to convert if file_format is set)"),
                    "channel": types.Schema(type="STRING", description="Slack channel name or ID (e.g. '#general'). Uses default channel if omitted."),
                    "file_format": types.Schema(type="STRING", description="If set, generates a file in this format (pdf/docx/csv/xlsx) and uploads it instead of sending text. Use when user requests a document."),
                    "file_title": types.Schema(type="STRING", description="Title for the generated file (used with file_format)."),
                }, required=["text"]),
            ),
            types.FunctionDeclaration(
                name="fetch_outlook_emails",
                description="Fetch unread emails from the user's Outlook inbox via the MCP Executive Bridge. Use when the user asks to check email, see unread messages, or wants an inbox summary.",
                parameters=types.Schema(type="OBJECT", properties={
                    "top": types.Schema(type="INTEGER", description="Number of emails to fetch (default 10)"),
                }),
            ),
            types.FunctionDeclaration(
                name="respond_to_outlook_invite",
                description="Accept or decline an Outlook calendar invite via the MCP Executive Bridge. Use when the user asks to accept or decline a meeting invitation.",
                parameters=types.Schema(type="OBJECT", properties={
                    "event_id": types.Schema(type="STRING", description="The Outlook event/invite ID"),
                    "action": types.Schema(type="STRING", description="'accept' or 'decline'"),
                    "comment": types.Schema(type="STRING", description="Optional response comment"),
                }, required=["event_id", "action"]),
            ),
            types.FunctionDeclaration(
                name="fetch_outlook_calendar",
                description="Fetch calendar events from Outlook via the MCP Executive Bridge. Use when the user asks about their Outlook calendar, external meetings, or wants to see scheduled events from their work calendar.",
                parameters=types.Schema(type="OBJECT", properties={
                    "start": types.Schema(type="STRING", description="ISO 8601 start date filter"),
                    "end": types.Schema(type="STRING", description="ISO 8601 end date filter"),
                }),
            ),
        ])


# ── State Recovery ─────────────────────────────────────

async def get_interrupted_run(run_id: str) -> dict | None:
    """Check if a run was interrupted and return its saved state."""
    db_conn = await db.get_db()
    cursor = await db_conn.execute(
        "SELECT * FROM agent_runs WHERE id = ? AND status = 'running'", (run_id,)
    )
    row = await cursor.fetchone()
    if not row:
        return None

    run_data = dict(row)
    if run_data.get("plan"):
        try:
            run_data["plan"] = json.loads(run_data["plan"])
        except (json.JSONDecodeError, TypeError):
            pass
    return run_data


async def get_completed_step_numbers(run_id: str) -> set[int]:
    """Get step numbers that have already been completed for a run."""
    db_conn = await db.get_db()
    cursor = await db_conn.execute(
        "SELECT content FROM agent_steps WHERE run_id = ? AND step_type = 'delegation' ORDER BY created_at",
        (run_id,)
    )
    rows = await cursor.fetchall()
    completed = set()
    for row in rows:
        try:
            data = json.loads(row["content"])
            step_num = data.get("step")
            if step_num is not None:
                completed.add(int(step_num))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Check which delegated steps actually finished (have a subsequent final_answer or tool_result)
    # For simplicity, we count any step that was delegated as "attempted"
    return completed


async def resume_interrupted_plan(run_id: str) -> dict | None:
    """Attempt to resume an interrupted plan execution.

    Returns the result if resumed, or None if no interrupted plan exists.
    """
    run_data = await get_interrupted_run(run_id)
    if not run_data or not run_data.get("plan"):
        return None

    plan_data = run_data["plan"]
    if isinstance(plan_data, str):
        try:
            plan_data = json.loads(plan_data)
        except json.JSONDecodeError:
            return None

    plan = plan_data.get("steps", [])
    if not plan:
        return None

    completed_steps = await get_completed_step_numbers(run_id)
    remaining = [s for s in plan if s.get("step") not in completed_steps]

    if not remaining:
        return None

    await mission_control.emit_narration(
        run_id, "primary_agent",
        f"Resuming interrupted plan — {len(completed_steps)} steps done, "
        f"{len(remaining)} remaining. Picking up where I left off..."
    )
    logger.info(f"Resuming run {run_id}: {len(completed_steps)} done, {len(remaining)} remaining")

    return {"remaining_plan": remaining, "user_message": run_data["user_message"],
            "completed_count": len(completed_steps)}


# ── Main Entry Point ───────────────────────────────────

async def process_message(run_id: str, user_message: str) -> dict:
    """Main entry point — called by the /chat endpoint.

    Orchestrates:
    1. Check for interrupted plan to resume
    2. Classify intent
    3. Route to appropriate agent(s)
    4. For complex tasks: generate plan -> execute steps with checkpointing
    5. Return synthesized result
    """
    now = datetime.now()

    # The user_message already contains [SYSTEM_TIME], [ACTION SNAPSHOT],
    # and [PERSONALIZED PATTERNS] headers injected by chat.py.
    # We just pass it through to preserve the full context chain.
    contextualized_message = user_message

    # ── Check for resume ──
    # (This would be triggered by a special "resume" message or automatic detection)
    # For now, fresh runs go through normal flow.

    # Step 1: Classify intent (use raw message for classification, not full context)
    # Extract the actual user text after context headers
    raw_message = user_message
    if "\nUser request:" in user_message:
        raw_message = user_message.split("\nUser request:")[-1].strip()
    elif user_message.count("\n") > 3:
        # Context headers present — find the last non-header line
        lines = user_message.strip().split("\n")
        for i, line in enumerate(lines):
            if not line.startswith("[") and not line.startswith("  ") and line.strip():
                raw_message = "\n".join(lines[i:]).strip()
                break

    classification = await classify_intent(run_id, raw_message)
    intent = classification.get("intent", "general")
    complexity = classification.get("complexity", "simple")

    # Step 2: Simple -> GeneralAgent
    if complexity == "simple" or intent == "general":
        await mission_control.emit_narration(
            run_id, "primary_agent",
            "This is straightforward — I'll handle it directly."
        )
        agent = GeneralAgent(run_id)
        return await agent.execute(contextualized_message)

    # Step 3: Complex -> Plan-and-Execute with checkpointing
    plan = await generate_plan(run_id, raw_message, classification)
    return await _execute_plan(run_id, raw_message, plan, now)


async def resume_plan(run_id: str) -> dict:
    """Resume an interrupted plan. Called via POST /chat/resume."""
    resume_state = await resume_interrupted_plan(run_id)
    if not resume_state:
        return {"response": "No interrupted plan found to resume.", "tools_used": []}

    now = datetime.now()
    return await _execute_plan(
        run_id, resume_state["user_message"],
        resume_state["remaining_plan"], now,
        step_offset=resume_state["completed_count"],
    )


async def _execute_plan(run_id: str, user_message: str,
                        plan: list[dict], now: datetime,
                        step_offset: int = 0) -> dict:
    """Execute a plan's steps with per-step checkpointing and error isolation."""
    all_results = []
    all_tools = []

    for i, step_info in enumerate(plan):
        step_num = step_info.get("step", i + 1 + step_offset)
        agent_name = step_info.get("agent", "general_agent")
        action = step_info.get("action", user_message)

        friendly_to = AGENT_FRIENDLY_NAMES.get(agent_name, agent_name)
        await mission_control.emit_narration(
            run_id, "primary_agent",
            f"Step {step_num}: Handing off to {friendly_to} — {action[:80]}..."
        )
        await mission_control.emit_delegation(
            run_id, "primary_agent", agent_name,
            f"Step {step_num}: {action}"
        )
        await db.log_agent_step(run_id, "primary_agent", "delegation",
                                {"to": agent_name, "step": step_num, "action": action})

        sub_agent = get_sub_agent(agent_name, run_id)
        if not sub_agent:
            await mission_control.emit_narration(
                run_id, "primary_agent",
                f"No specialist for '{agent_name}' — I'll handle this step myself."
            )
            sub_agent = GeneralAgent(run_id)

        step_context = (
            f"[Current date/time: {now.isoformat()}]\n\n"
            f"You are executing step {step_num} of a larger plan.\n"
            f"Overall user request: {user_message}\n"
            f"Your specific task: {action}\n"
        )
        if all_results:
            step_context += "\nPrevious steps completed:\n"
            for j, prev in enumerate(all_results):
                step_context += f"  Step {j+1+step_offset} result summary: {prev[:200]}\n"

        # Execute with error isolation — one step failing doesn't kill the whole plan
        try:
            result = await sub_agent.execute(step_context)
            step_response = result.get("response", "")
            all_tools.extend(result.get("tools_used", []))
        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}")
            step_response = f"[Step {step_num} encountered an error: {str(e)[:200]}]"
            await mission_control.emit_error(run_id, "primary_agent",
                                             f"Step {step_num} failed: {e}")
            await mission_control.emit_narration(
                run_id, "primary_agent",
                f"Step {step_num} hit an issue, but I'm continuing with the rest of the plan."
            )

        all_results.append(step_response)

        # Checkpoint: record which steps are complete
        await db.update_agent_run(run_id, plan=json.dumps({
            "steps": plan,
            "completed_steps": list(range(1, len(all_results) + 1 + step_offset)),
        }))

    # Synthesize final response
    if len(all_results) == 1:
        final_response = all_results[0]
    else:
        await mission_control.emit_narration(
            run_id, "primary_agent",
            "All steps complete. Let me put together a summary..."
        )
        synthesis_prompt = (
            f"The user asked: {user_message}\n\n"
            f"The following steps were completed:\n"
        )
        for i, (step_info, result) in enumerate(zip(plan, all_results)):
            synthesis_prompt += f"\nStep {i+1} ({step_info.get('action', '?')}):\n{result}\n"
        synthesis_prompt += (
            "\nProvide a clear, unified summary of everything that was accomplished. "
            "Be specific about what was created (tasks, events, notes) and any important details."
        )

        try:
            synthesis_response = await llm_router.generate_content(
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=synthesis_prompt)])],
                system_instruction="You are NEXUS, summarizing the results of a multi-step operation. Be clear, concise, and specific.",
                temperature=0.1,
                agent_name="primary_agent",
                run_id=run_id,
            )
            # ── Defensive: synthesis_response.text may be None ──
            final_response = getattr(synthesis_response, "text", None) or "\n".join(all_results)
        except AllProvidersExhausted:
            # ── Deterministic fallback: concatenate raw step results ──
            logger.warning(f"[_synthesize] All providers exhausted — concatenating results")
            await mission_control.emit_narration(
                run_id, "primary_agent",
                "Compiling results using built-in summarization..."
            )
            final_response = "\n\n".join(
                f"Step {i+1}: {r}" for i, r in enumerate(all_results) if r
            )

    await mission_control.emit_final_answer(run_id, "primary_agent", final_response)
    return {"response": final_response, "tools_used": list(set(all_tools))}
