"""Web Search Tool — Tavily API integration for NEXUS.

Provides real-time web search capability so the agent mesh can answer
questions about current events, stock prices, news, and any data not
stored in the local Vault.

MCP Pattern:
    - Typed inputs via Pydantic models
    - Structured JSON outputs
    - Built-in error boundary (never raises to caller)
    - Graceful degradation when TAVILY_API_KEY is missing

Zero-Token Fail-Safe:
    If TAVILY_API_KEY is not set, all methods return structured
    "disabled" responses. The core NEXUS system is NEVER affected.
"""

import logging
import os
import time
from collections import deque
from typing import Optional

from pydantic import BaseModel, Field

from mission_control import mission_control

logger = logging.getLogger("nexus.tools.web_search")

# ── Configuration ─────────────────────────────────────────
# Global shared Tavily key — one key for all users, set in server .env
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── Per-User Rate Limiting ────────────────────────────────
# Protects API budget: 10 searches per user per hour
_SEARCH_RATE_LIMIT = 10       # max searches per window
_SEARCH_RATE_WINDOW = 3600    # window in seconds (1 hour)
_user_search_log: dict[str, deque] = {}  # user_id → deque of timestamps


def _check_search_rate(user_id: str) -> bool:
    """Check if user is within their search rate limit. Returns True if allowed."""
    now = time.monotonic()
    if user_id not in _user_search_log:
        _user_search_log[user_id] = deque()

    log = _user_search_log[user_id]

    # Evict timestamps outside the window
    while log and (now - log[0]) > _SEARCH_RATE_WINDOW:
        log.popleft()

    return len(log) < _SEARCH_RATE_LIMIT


def _record_search(user_id: str):
    """Record a search timestamp for rate limiting."""
    now = time.monotonic()
    if user_id not in _user_search_log:
        _user_search_log[user_id] = deque()
    _user_search_log[user_id].append(now)


# ── Pydantic Models ──────────────────────────────────────

class WebSearchRequest(BaseModel):
    """Input schema for web search."""
    query: str = Field(..., description="The search query")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum results to return")
    search_depth: str = Field(default="basic", description="'basic' or 'advanced'")
    include_domains: Optional[list[str]] = Field(default=None, description="Restrict to these domains")
    exclude_domains: Optional[list[str]] = Field(default=None, description="Exclude these domains")


class WebSearchResult(BaseModel):
    """A single search result."""
    title: str
    url: str
    content: str
    score: float = 0.0


class WebSearchResponse(BaseModel):
    """Output schema for web search."""
    status: str  # "ok", "disabled", "error"
    query: str
    results: list[WebSearchResult] = []
    answer: Optional[str] = None  # Tavily's AI-generated answer
    count: int = 0
    message: Optional[str] = None


# ── Core Search Function ─────────────────────────────────

def _is_enabled() -> bool:
    """Check if web search is available."""
    return bool(TAVILY_API_KEY)


async def web_search(
    run_id: str,
    agent_name: str,
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    user_id: str = "default",
) -> dict:
    """Search the live web using Tavily API.

    This is the tool function called by the agent system.
    Returns a plain dict (JSON-serializable) for the LLM.

    Args:
        run_id: Current agent run ID
        agent_name: Name of the calling agent
        query: Search query
        max_results: Number of results (1-10)
        search_depth: "basic" (fast) or "advanced" (thorough)
        include_domains: Optional domain whitelist
        exclude_domains: Optional domain blacklist
        user_id: User ID for per-user rate limiting

    Returns:
        dict with keys: status, query, results, answer, count, message
    """
    await mission_control.emit_tool_call(
        run_id, agent_name, "web_search.search",
        {"query": query, "max_results": max_results, "depth": search_depth}
    )

    # ── Per-user rate limiting ──
    if not _check_search_rate(user_id):
        result = WebSearchResponse(
            status="rate_limited",
            query=query,
            message="Web search limit reached for this hour. Please try again later.",
        )
        await mission_control.emit_tool_result(
            run_id, agent_name, "web_search.search", result.model_dump()
        )
        logger.warning(f"Rate limit hit for user '{user_id}' — query: {query}")
        return result.model_dump()

    if not _is_enabled():
        result = WebSearchResponse(
            status="disabled",
            query=query,
            message="Web search not configured. Set TAVILY_API_KEY in .env to enable.",
        )
        await mission_control.emit_tool_result(
            run_id, agent_name, "web_search.search", result.model_dump()
        )
        return result.model_dump()

    try:
        import httpx

        async with httpx.AsyncClient(timeout=15.0) as client:
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": True,
            }
            if include_domains:
                payload["include_domains"] = include_domains
            if exclude_domains:
                payload["exclude_domains"] = exclude_domains

            resp = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )

            if resp.status_code != 200:
                error_msg = f"Tavily API returned {resp.status_code}: {resp.text[:200]}"
                logger.error(error_msg)
                result = WebSearchResponse(
                    status="error", query=query, message=error_msg
                )
                await mission_control.emit_tool_result(
                    run_id, agent_name, "web_search.search", result.model_dump()
                )
                return result.model_dump()

            data = resp.json()

            # Parse results
            search_results = []
            for item in data.get("results", []):
                search_results.append(WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                ))

            result = WebSearchResponse(
                status="ok",
                query=query,
                results=search_results,
                answer=data.get("answer"),
                count=len(search_results),
            )

            await mission_control.emit_tool_result(
                run_id, agent_name, "web_search.search",
                {"count": result.count, "answer_preview": (result.answer or "")[:200]}
            )

            _record_search(user_id)
            logger.info(f"Web search: '{query}' → {result.count} results (user={user_id})")
            return result.model_dump()

    except ImportError:
        logger.error("httpx not installed — web search unavailable")
        result = WebSearchResponse(
            status="error", query=query,
            message="httpx not installed. Run: pip install httpx"
        )
        await mission_control.emit_tool_result(
            run_id, agent_name, "web_search.search", result.model_dump()
        )
        return result.model_dump()

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        result = WebSearchResponse(
            status="error", query=query, message=str(e)
        )
        await mission_control.emit_tool_result(
            run_id, agent_name, "web_search.search", result.model_dump()
        )
        return result.model_dump()


def get_status() -> dict:
    """Get web search tool status for the Neural Link UI."""
    return {
        "name": "Web Search (Tavily)",
        "enabled": _is_enabled(),
        "provider": "Tavily",
    }
