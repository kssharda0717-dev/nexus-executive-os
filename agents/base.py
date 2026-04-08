"""Base Agent — Gemini-powered agent with tool calling, Mission Control streaming,
centralized rate limiting, 429-aware retry, and state recovery.

Every agent in NEXUS inherits from this class. It provides:
1. A ReAct-style reasoning loop (think -> act -> observe -> repeat)
2. Automatic tool execution with result feeding
3. Real-time streaming of every thought and action to Mission Control
4. Centralized Gemini rate limiter (Semaphore + min-spacing + jitter)
5. 429 RESOURCE_EXHAUSTED-aware exponential backoff with retryDelay parsing
6. Agent step logging to the database for state recovery
"""

import asyncio
import json
import logging
import random
import re
import traceback
from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY, GEMINI_FLASH_MODEL, MAX_AGENT_STEPS,
    GEMINI_MAX_CONCURRENT, GEMINI_MIN_CALL_SPACING_S,
    GEMINI_429_BASE_DELAY_S, GEMINI_429_MAX_DELAY_S, GEMINI_JITTER_S,
)
from llm_router import AllProvidersExhausted
from mission_control import mission_control
import database as db

logger = logging.getLogger("nexus.agents")

# ── Lazy Gemini Client ─────────────────────────────────

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ── Centralized Gemini Rate Limiter ────────────────────

class GeminiRateLimiter:
    """Ensures ALL Gemini API calls across every agent are serialized with
    minimum spacing and random jitter. This prevents 429 RESOURCE_EXHAUSTED
    by respecting the Free Tier RPM (15 req/min = ~4s/req).

    Usage:
        await gemini_limiter.acquire()
        try:
            result = await make_gemini_call()
        finally:
            gemini_limiter.release()
    """

    def __init__(self):
        self._semaphore = asyncio.Semaphore(GEMINI_MAX_CONCURRENT)
        self._last_call_time: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait for the semaphore AND enforce minimum call spacing + jitter."""
        await self._semaphore.acquire()
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_call_time
            min_spacing = GEMINI_MIN_CALL_SPACING_S + random.uniform(0, GEMINI_JITTER_S)
            if elapsed < min_spacing:
                wait = min_spacing - elapsed
                logger.debug(f"Rate limiter: spacing wait {wait:.2f}s")
                await asyncio.sleep(wait)
            self._last_call_time = asyncio.get_event_loop().time()

    def release(self):
        self._semaphore.release()


# Module-level singleton — imported by primary.py and all agents
gemini_limiter = GeminiRateLimiter()


# ── 429-Aware Retry Logic ──────────────────────────────

def _is_rate_limit_error(e: Exception) -> bool:
    """Detect Gemini 429 / RESOURCE_EXHAUSTED errors."""
    msg = str(e).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg


def _parse_retry_delay(e: Exception) -> float | None:
    """Try to extract a retryDelay (in seconds) from the Gemini error message."""
    match = re.search(r'retry.*?(\d+\.?\d*)\s*s', str(e), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


async def retry_async(coro_factory, max_retries: int = 3,
                      base_delay: float = 1.0, max_delay: float = 15.0,
                      run_id: str = "", agent_name: str = "", label: str = ""):
    """Execute an async callable with exponential backoff.

    429/RESOURCE_EXHAUSTED errors get special treatment:
    - Longer base delay (GEMINI_429_BASE_DELAY_S)
    - Parses retryDelay from error message if available
    - More aggressive backoff cap (GEMINI_429_MAX_DELAY_S)
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                if _is_rate_limit_error(e):
                    # 429 path: respect the API's suggested delay or use our own
                    parsed = _parse_retry_delay(e)
                    delay = parsed if parsed else min(
                        GEMINI_429_BASE_DELAY_S * (2 ** attempt),
                        GEMINI_429_MAX_DELAY_S
                    )
                    logger.warning(
                        f"[{agent_name}] RATE LIMITED on {label} "
                        f"(attempt {attempt+1}/{max_retries}). "
                        f"Waiting {delay:.1f}s..."
                    )
                    await mission_control.emit_thought(
                        run_id, agent_name,
                        f"Rate limit hit — cooling down for {delay:.0f}s before retry..."
                    )
                else:
                    # Generic error path: standard exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"[{agent_name}] {label} attempt {attempt+1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await mission_control.emit_thought(
                        run_id, agent_name,
                        f"Hmm, {label} hiccup (attempt {attempt+1}/{max_retries}). "
                        f"Retrying in {delay:.0f}s..."
                    )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[{agent_name}] {label} failed after {max_retries} attempts: {e}"
                )
    raise last_error


# ── Personable Narration ───────────────────────────────

AGENT_FRIENDLY_NAMES = {
    "primary_agent": "NEXUS",
    "general_agent": "NEXUS",
    "planning_agent": "Schedule Architect",
    "learning_agent": "Learning Coach",
    "life_admin_agent": "Life Admin",
    "proactive_monitor": "Watchdog",
}

TOOL_NARRATIONS = {
    "list_events": "Scanning your calendar...",
    "create_event": "Booking time on your calendar...",
    "update_event": "Adjusting a calendar event...",
    "delete_event": "Clearing a calendar event...",
    "find_free_slots": "Hunting for open time slots...",
    "list_tasks": "Reviewing your task list...",
    "create_task": "Adding a new task to your list...",
    "update_task": "Updating a task...",
    "complete_task": "Marking a task as done!",
    "get_overdue_tasks": "Checking for overdue items...",
    "list_notes": "Browsing your notes...",
    "create_note": "Writing a new note...",
    "update_note": "Updating a note...",
    "search_notes": "Searching through your notes...",
    "append_to_note": "Adding to an existing note...",
    "web_search": "Researching live web data...",
    "linkedin_search": "Searching for LinkedIn profiles...",
    "generate_document": "Generating a document file...",
    "send_slack_message": "Sending a Slack message...",
    "fetch_outlook_emails": "Checking your Outlook inbox...",
    "respond_to_outlook_invite": "Responding to a calendar invite...",
    "fetch_outlook_calendar": "Checking your Outlook calendar...",
}


# ── Base Agent ─────────────────────────────────────────

class BaseAgent:
    """Base class for all NEXUS agents."""

    name: str = "base_agent"
    description: str = "Base agent"
    system_prompt: str = "You are a helpful AI assistant."
    model: str = GEMINI_FLASH_MODEL
    tools: list = []
    tool_handlers: dict = {}

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.conversation_history: list[types.Content] = []
        self.tools_used: list[str] = []
        self.friendly_name = AGENT_FRIENDLY_NAMES.get(self.name, self.name)

    async def think(self, thought: str):
        """Emit a thought to Mission Control and log it."""
        await mission_control.emit_thought(self.run_id, self.name, thought)
        await db.log_agent_step(self.run_id, self.name, "thought", {"thought": thought})

    async def narrate(self, message: str):
        """Emit a personable narration to Mission Control (separate from technical thoughts)."""
        await mission_control.emit_narration(self.run_id, self.name, message)

    async def _call_gemini(self, step: int) -> object:
        """Call Gemini through the LLM Router (multi-model failover).

        The LLMRouter handles: model selection based on agent tier,
        per-model rate limiting, 429 failover cascade, and cooldown tracking.
        """
        from llm_router import llm_router

        gemini_tools = self.tools if self.tools else None

        return await llm_router.generate_content(
            contents=self.conversation_history,
            system_instruction=self.system_prompt,
            tools=gemini_tools,
            temperature=0.2,
            agent_name=self.name,
            run_id=self.run_id,
        )

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Execute a single tool with guardrail validation, retry logic, and narration.

        Pre-flight check flow:
        1. Run guardrails/validator against (tool_name, tool_args)
        2. If blocked → return the rejection as a tool result so the LLM can self-correct
        3. If passed → proceed with execution
        """
        narration = TOOL_NARRATIONS.get(tool_name, f"Using {tool_name}...")
        await self.narrate(f"{self.friendly_name}: {narration}")

        await mission_control.emit_tool_call(self.run_id, self.name, tool_name, tool_args)
        await db.log_agent_step(self.run_id, self.name, "tool_call",
                                {"tool": tool_name, "args": tool_args})

        if tool_name not in self.tool_handlers:
            result = {"error": f"Unknown tool: {tool_name}"}
            await mission_control.emit_error(self.run_id, self.name, f"Unknown tool: {tool_name}")
            return result

        # ── GUARDRAIL PRE-FLIGHT CHECK ──
        try:
            from guardrails.validator import validator, ValidationContext
            # Build context with current time + recent action snapshot
            snapshot = await db.get_action_snapshot(limit=5)
            ctx = ValidationContext(action_snapshot=snapshot)
            check = await validator.check(tool_name, tool_args, ctx)

            if not check.passed:
                # Guardrail BLOCKED — return structured rejection to the LLM
                await self.think(
                    f"GUARDRAIL [{check.blocked_by}]: {check.message} "
                    f"Suggestion: {check.suggestion}"
                )
                await self.narrate(
                    f"{self.friendly_name}: Hold on — {check.message}"
                )
                await mission_control.emit_thought(
                    self.run_id, self.name,
                    f"Pre-flight check failed [{check.category}/{check.blocked_by}]: "
                    f"{check.message}"
                )
                result = check.to_tool_result()
                self.tools_used.append(tool_name)
                await mission_control.emit_tool_result(
                    self.run_id, self.name, tool_name, result
                )
                await db.log_agent_step(self.run_id, self.name, "tool_result",
                                        {"tool": tool_name, "result": result,
                                         "guardrail_blocked": True})
                return result
        except ImportError:
            pass  # Guardrails module not available — proceed without checks
        except Exception as e:
            # Guardrails must NEVER crash the tool call
            logger.debug(f"Guardrail check skipped for {tool_name}: {e}")

        # ── EXECUTE TOOL ──
        async def _run_tool():
            return await self.tool_handlers[tool_name](self.run_id, self.name, **tool_args)

        try:
            result = await retry_async(
                _run_tool,
                max_retries=2, base_delay=0.5, max_delay=5.0,
                run_id=self.run_id, agent_name=self.name,
                label=f"tool:{tool_name}",
            )
            if not isinstance(result, (dict, list)):
                result = {"result": str(result)}
            elif isinstance(result, list):
                result = {"items": result, "count": len(result)}
        except Exception as e:
            result = {"error": str(e), "traceback": traceback.format_exc()[-500:]}
            await mission_control.emit_error(
                self.run_id, self.name, f"Tool {tool_name} failed after retries: {e}"
            )
            await self.narrate(f"{self.friendly_name}: Ran into an issue with {tool_name}, but I'm working around it.")

        self.tools_used.append(tool_name)
        await mission_control.emit_tool_result(self.run_id, self.name, tool_name, result)
        await db.log_agent_step(self.run_id, self.name, "tool_result",
                                {"tool": tool_name, "result": result})
        return result

    async def execute(self, user_message: str) -> dict:
        """Run the agent's reasoning loop until it produces a final answer.

        ReAct loop:
        1. Send message + conversation history to Gemini (rate-limited + retried)
        2. If Gemini returns tool calls -> execute them (with retries), feed results back
        3. If Gemini returns text -> that's the final answer
        4. Repeat up to MAX_AGENT_STEPS

        Defensive guarantees:
        - candidate.content can be None (blocked/empty response) → skip gracefully
        - candidate.content.parts can be None (not []) → coerce to []
        - Never appends None into conversation_history
        - Every part is checked with getattr before accessing .text / .function_call
        """
        await self.think(f"Starting execution. Input: {user_message[:200]}")
        await self.narrate(f"{self.friendly_name} is on it...")

        self.conversation_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        )

        for step in range(MAX_AGENT_STEPS):
            try:
                response = await self._call_gemini(step)
            except AllProvidersExhausted as e:
                # ── Structured degradation: no raw tracebacks ──
                await mission_control.emit_narration(
                    self.run_id, self.name,
                    f"{self.friendly_name}: High demand on AI infrastructure. "
                    f"Completing with what I have so far."
                )
                await db.log_agent_step(self.run_id, self.name, "degraded",
                                        {"reason": str(e), "step": step})
                # Return best-effort from conversation history
                last_texts = []
                for content in reversed(self.conversation_history):
                    if content and getattr(content, "parts", None):
                        for p in content.parts:
                            if getattr(p, "text", None):
                                last_texts.append(p.text)
                        if last_texts:
                            break
                fallback = "\n".join(last_texts) if last_texts else (
                    "I'm experiencing high demand right now. "
                    "Your request is noted — please try again shortly."
                )
                return {"response": fallback, "tools_used": self.tools_used}
            except Exception as e:
                error_msg = f"Gemini API error after retries: {str(e)}"
                await mission_control.emit_error(self.run_id, self.name, error_msg)
                await db.log_agent_step(self.run_id, self.name, "error", {"error": error_msg})
                await self.narrate(f"{self.friendly_name}: I hit a wall talking to my brain. Sorry about that.")
                return {"response": f"I encountered an error: {error_msg}", "tools_used": self.tools_used}

            # ── Guard 1: response.candidates may be None or empty ──
            candidates = getattr(response, "candidates", None) or []
            candidate = candidates[0] if candidates else None
            if not candidate:
                await self.think("No response from Gemini — ending loop.")
                return {"response": "I wasn't able to generate a response.", "tools_used": self.tools_used}

            # ── Guard 2: candidate.content may be None (blocked/empty) ──
            content = getattr(candidate, "content", None)
            if content is None:
                await self.think("Gemini returned empty content — treating as 'Task completed.'")
                return {"response": "Task completed.", "tools_used": self.tools_used}

            # ── Guard 3: content.parts may be None (not []) ──
            parts = getattr(content, "parts", None) or []

            # Only append valid Content objects to history — never None
            self.conversation_history.append(content)

            # ── Guard 4: each part's .text / .function_call may be absent ──
            text_parts = [
                p.text for p in parts
                if getattr(p, "text", None)
            ]
            function_calls = [
                p for p in parts
                if getattr(p, "function_call", None)
            ]

            if function_calls:
                function_responses = []
                for fc_part in function_calls:
                    fc = fc_part.function_call
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    result = await self._execute_tool(tool_name, tool_args)

                    function_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response=result,
                        )
                    )

                self.conversation_history.append(
                    types.Content(role="user", parts=function_responses)
                )
                continue

            # Final text answer
            final_text = "\n".join(text_parts) if text_parts else "Task completed."
            await self.think(f"Reached final answer after {step + 1} steps.")
            await self.narrate(f"{self.friendly_name}: All done!")
            await mission_control.emit_final_answer(self.run_id, self.name, final_text)
            await db.log_agent_step(self.run_id, self.name, "final_answer",
                                    {"answer": final_text[:1000]})

            return {"response": final_text, "tools_used": list(set(self.tools_used))}

        # Hit max steps
        await self.think(f"Hit max steps ({MAX_AGENT_STEPS}). Returning best effort.")
        await self.narrate(f"{self.friendly_name}: That was a complex one! Here's what I've got so far.")
        last_texts = []
        for content in reversed(self.conversation_history):
            if content and getattr(content, "parts", None):
                for p in content.parts:
                    if getattr(p, "text", None):
                        last_texts.append(p.text)
                if last_texts:
                    break
        fallback = "\n".join(last_texts) if last_texts else "Reached step limit."
        return {"response": fallback, "tools_used": list(set(self.tools_used))}
