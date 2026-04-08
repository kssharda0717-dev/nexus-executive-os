"""LinkedIn Networking Agent — Profile search, synthesis, and outreach drafting.

This is the "Winning Edge" feature: a networking tool that allows NEXUS to:
1. Search for LinkedIn profiles based on role/industry/location criteria
2. Synthesize profile data (name, role, company, URL)
3. Generate high-context "Conversation Starter" messages based on the
   user's resume/profile and the target's role
4. Present results in a clean Markdown table

MCP Pattern:
    - Typed inputs/outputs via Pydantic models
    - Built-in error boundary
    - Graceful degradation when credentials are missing

Zero-Token Fail-Safe:
    If TAVILY_API_KEY is not set (used for search), the tool returns
    structured "disabled" responses. Core NEXUS is NEVER affected.

Note: This uses web search (Tavily) to find public LinkedIn profiles
rather than the official LinkedIn API, since OAuth approval requires
a verified LinkedIn Partner app. This approach works immediately
and demonstrates the full agentic workflow.
"""

import logging
import os
import time
from collections import deque
from typing import Optional

from pydantic import BaseModel, Field

from mission_control import mission_control

logger = logging.getLogger("nexus.tools.linkedin_agent")

# ── Configuration ─────────────────────────────────────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── Per-User Rate Limiting ────────────────────────────────
# Shares the same budget concept as web_search: 10 searches per user per hour
_SEARCH_RATE_LIMIT = 10       # max searches per window
_SEARCH_RATE_WINDOW = 3600    # window in seconds (1 hour)
_user_search_log: dict[str, deque] = {}  # user_id → deque of timestamps


def _check_search_rate(user_id: str) -> bool:
    """Check if user is within their search rate limit. Returns True if allowed."""
    now = time.monotonic()
    if user_id not in _user_search_log:
        _user_search_log[user_id] = deque()
    log = _user_search_log[user_id]
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

class LinkedInProfile(BaseModel):
    """A parsed LinkedIn profile from search results."""
    name: str = ""
    role: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    snippet: str = ""


class NetworkingRequest(BaseModel):
    """Input for the networking search."""
    query: str = Field(..., description="Search criteria, e.g. 'HCM recruiters in London'")
    user_context: Optional[str] = Field(
        default=None,
        description="User's background/resume summary for personalized outreach"
    )
    max_results: int = Field(default=5, ge=1, le=10)


class ConversationStarter(BaseModel):
    """A generated outreach message for a specific profile."""
    profile: LinkedInProfile
    message: str
    approach: str  # e.g. "mutual industry", "skill complement", "company interest"


class NetworkingResponse(BaseModel):
    """Output from the networking agent."""
    status: str  # "ok", "disabled", "error"
    query: str
    profiles: list[LinkedInProfile] = []
    outreach_drafts: list[ConversationStarter] = []
    markdown_table: str = ""
    count: int = 0
    message: Optional[str] = None


# ── Profile Parsing ──────────────────────────────────────

def _parse_linkedin_result(result: dict) -> LinkedInProfile:
    """Parse a web search result into a LinkedInProfile.

    Extracts structured data from Tavily search results that point
    to linkedin.com profiles.
    """
    url = result.get("url", "")
    title = result.get("title", "")
    content = result.get("content", "")

    # Parse name and role from LinkedIn title format: "Name - Role - Company | LinkedIn"
    name = ""
    role = ""
    company = ""

    if " - " in title:
        parts = title.replace(" | LinkedIn", "").split(" - ")
        if len(parts) >= 1:
            name = parts[0].strip()
        if len(parts) >= 2:
            role = parts[1].strip()
        if len(parts) >= 3:
            company = parts[2].strip()
    elif " | " in title:
        name = title.split(" | ")[0].strip()

    # Try to extract location from content
    location = ""
    content_lower = content.lower()
    location_markers = ["location:", "based in", "located in"]
    for marker in location_markers:
        idx = content_lower.find(marker)
        if idx != -1:
            loc_text = content[idx + len(marker):].strip()
            location = loc_text.split(".")[0].split(",")[0].strip()[:50]
            break

    return LinkedInProfile(
        name=name or title[:60],
        role=role,
        company=company,
        location=location,
        url=url,
        snippet=content[:200],
    )


def _build_markdown_table(profiles: list[LinkedInProfile]) -> str:
    """Build a clean Markdown table from profiles."""
    if not profiles:
        return "_No profiles found._"

    lines = [
        "| # | Name | Role | Company | Profile |",
        "|---|------|------|---------|---------|",
    ]
    for i, p in enumerate(profiles, 1):
        name = p.name or "Unknown"
        role = p.role or "—"
        company = p.company or "—"
        link = f"[View]({p.url})" if p.url else "—"
        lines.append(f"| {i} | {name} | {role} | {company} | {link} |")

    return "\n".join(lines)


def _generate_conversation_starter(
    profile: LinkedInProfile,
    user_context: str = "",
) -> ConversationStarter:
    """Generate a personalized conversation starter for a profile.

    Uses template-based generation (no LLM call) to avoid consuming
    Gemini quota on outreach drafting. The templates are high-quality
    and contextually aware.
    """
    name = profile.name.split()[0] if profile.name else "there"
    role = profile.role or "your current role"
    company = profile.company or "your organization"

    # Determine approach based on available data
    if user_context and profile.company:
        approach = "mutual industry"
        message = (
            f"Hi {name}, I came across your profile and was impressed by your work "
            f"as {role} at {company}. {user_context[:100] if user_context else ''} "
            f"I'd love to connect and explore potential synergies between our backgrounds. "
            f"Would you be open to a brief conversation?"
        )
    elif profile.company:
        approach = "company interest"
        message = (
            f"Hi {name}, your trajectory at {company} caught my attention — "
            f"particularly your experience in {role}. I'm exploring opportunities "
            f"in this space and would value your perspective. "
            f"Would you have 15 minutes for a virtual coffee?"
        )
    else:
        approach = "skill complement"
        message = (
            f"Hi {name}, your background in {role} resonates with my professional "
            f"interests. I believe there could be meaningful overlap in our expertise. "
            f"I'd appreciate the chance to connect and exchange insights."
        )

    return ConversationStarter(
        profile=profile,
        message=message.strip(),
        approach=approach,
    )


# ── Core Tool Function ───────────────────────────────────

def _is_enabled() -> bool:
    """Check if LinkedIn networking tool is available."""
    return bool(TAVILY_API_KEY)


async def _tavily_search_profiles(client, query: str, max_results: int) -> list[LinkedInProfile]:
    """Execute a single Tavily search and return parsed LinkedIn profiles."""
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": f"site:linkedin.com/in {query}",
        "max_results": max_results,
        "search_depth": "advanced",
        "include_domains": ["linkedin.com"],
        "include_answer": False,
    }
    resp = await client.post("https://api.tavily.com/search", json=payload)
    if resp.status_code != 200:
        logger.warning(f"Tavily search returned {resp.status_code} for query: {query}")
        return []
    data = resp.json()
    profiles = []
    for item in data.get("results", []):
        url = item.get("url", "")
        if "linkedin.com/in/" in url:
            profiles.append(_parse_linkedin_result(item))
    return profiles


def _generate_broadened_queries(query: str) -> list[str]:
    """Generate broadened/alternative search queries from the original query.

    Splits compound queries into individual facets and generates semantic
    variants to maximize profile coverage when the original query is too narrow.
    """
    variants = []
    q_lower = query.lower()

    # Extract meaningful tokens (skip common prepositions/articles)
    stop_words = {"in", "at", "for", "the", "a", "an", "and", "or", "with", "on", "of", "to"}
    tokens = [t for t in query.split() if t.lower() not in stop_words and len(t) > 1]

    # Strategy 1: Drop location qualifier (keep role + company/industry)
    location_markers = ["in ", "at ", "based in ", "located in ", "from "]
    for marker in location_markers:
        idx = q_lower.find(marker)
        if idx > 0:
            role_part = query[:idx].strip()
            if len(role_part) > 3:
                variants.append(role_part)

    # Strategy 2: If multiple nouns, split into sub-queries
    # e.g. "Tech Recruiter Oracle" → "Tech Recruiter", "Oracle Recruiter"
    if len(tokens) >= 3:
        variants.append(f"{tokens[0]} {tokens[1]}")
        variants.append(f"{tokens[-1]} {tokens[0]}")

    # Strategy 3: Add synonym expansions for common role terms
    role_synonyms = {
        "recruiter": ["talent acquisition", "hiring manager", "talent partner"],
        "engineer": ["developer", "software engineer", "SDE"],
        "manager": ["lead", "director", "head of"],
        "designer": ["UX designer", "product designer", "design lead"],
        "data": ["data science", "analytics", "machine learning"],
        "ai": ["artificial intelligence", "machine learning", "deep learning"],
    }
    for token in tokens:
        synonyms = role_synonyms.get(token.lower(), [])
        if synonyms:
            variants.append(synonyms[0])  # Use top synonym only
            break

    # Deduplicate while preserving order, skip if too similar to original
    seen = {query.lower()}
    unique = []
    for v in variants:
        v_clean = v.strip()
        if v_clean.lower() not in seen and len(v_clean) > 3:
            seen.add(v_clean.lower())
            unique.append(v_clean)
    return unique[:3]  # Cap at 3 broadened queries


async def linkedin_search(
    run_id: str,
    agent_name: str,
    query: str,
    user_context: str = "",
    max_results: int = 5,
    user_id: str = "default",
) -> dict:
    """Search for LinkedIn profiles with iterative broadening.

    If the initial query returns fewer than 3 results, automatically
    generates broadened queries, searches in parallel, deduplicates
    by profile URL, and returns the combined set up to max_results.

    Args:
        run_id: Current agent run ID
        agent_name: Calling agent name
        query: Search criteria (e.g. "HCM recruiters in London")
        user_context: User's background for personalized outreach
        max_results: Number of profiles to find (1-10)

    Returns:
        dict with: status, profiles, outreach_drafts, markdown_table, count
    """
    await mission_control.emit_tool_call(
        run_id, agent_name, "linkedin.search",
        {"query": query, "max_results": max_results}
    )

    await mission_control.emit_narration(
        run_id, agent_name,
        f"Researching LinkedIn profiles: {query}..."
    )

    # ── Per-user rate limiting ──
    if not _check_search_rate(user_id):
        result = NetworkingResponse(
            status="rate_limited",
            query=query,
            message="Web search limit reached for this hour. Please try again later.",
        )
        await mission_control.emit_tool_result(
            run_id, agent_name, "linkedin.search", result.model_dump()
        )
        logger.warning(f"Rate limit hit for user '{user_id}' — query: {query}")
        return result.model_dump()

    if not _is_enabled():
        result = NetworkingResponse(
            status="disabled",
            query=query,
            message="LinkedIn search not configured. Set TAVILY_API_KEY in .env to enable web-powered profile search.",
        )
        await mission_control.emit_tool_result(
            run_id, agent_name, "linkedin.search", result.model_dump()
        )
        return result.model_dump()

    try:
        import asyncio as _aio
        import httpx

        async with httpx.AsyncClient(timeout=15.0) as client:
            # ── Phase 1: Primary search ──
            profiles = await _tavily_search_profiles(client, query, max_results)

            # ── Phase 2: Iterative broadening if results are thin ──
            BROADENING_THRESHOLD = 3
            if len(profiles) < BROADENING_THRESHOLD and len(profiles) < max_results:
                broadened_queries = _generate_broadened_queries(query)
                if broadened_queries:
                    await mission_control.emit_narration(
                        run_id, agent_name,
                        f"Found only {len(profiles)} profiles. "
                        f"Broadening search with {len(broadened_queries)} alternative queries..."
                    )

                    # Fetch broadened results in parallel
                    remaining = max_results - len(profiles)
                    tasks = [
                        _tavily_search_profiles(client, bq, remaining)
                        for bq in broadened_queries
                    ]
                    broadened_results = await _aio.gather(*tasks, return_exceptions=True)

                    # Deduplicate by profile URL
                    seen_urls = {p.url for p in profiles if p.url}
                    for batch in broadened_results:
                        if isinstance(batch, Exception):
                            logger.warning(f"Broadened search failed: {batch}")
                            continue
                        for profile in batch:
                            if profile.url and profile.url not in seen_urls:
                                seen_urls.add(profile.url)
                                profiles.append(profile)
                                if len(profiles) >= max_results:
                                    break
                        if len(profiles) >= max_results:
                            break

            # Trim to requested count
            profiles = profiles[:max_results]

        # Generate conversation starters
        outreach_drafts = []
        for profile in profiles:
            draft = _generate_conversation_starter(profile, user_context)
            outreach_drafts.append(draft)

        # Build markdown table
        markdown_table = _build_markdown_table(profiles)

        result = NetworkingResponse(
            status="ok",
            query=query,
            profiles=profiles,
            outreach_drafts=outreach_drafts,
            markdown_table=markdown_table,
            count=len(profiles),
        )

        await mission_control.emit_narration(
            run_id, agent_name,
            f"Found {len(profiles)} LinkedIn profiles matching '{query}'. "
            f"Generating personalized outreach drafts..."
        )

        await mission_control.emit_tool_result(
            run_id, agent_name, "linkedin.search",
            {"count": result.count, "profiles": [p.name for p in profiles]}
        )

        _record_search(user_id)
        logger.info(f"LinkedIn search: '{query}' → {result.count} profiles (user={user_id})")
        return result.model_dump()

    except ImportError:
        logger.error("httpx not installed — LinkedIn search unavailable")
        result = NetworkingResponse(
            status="error", query=query,
            message="httpx not installed. Run: pip install httpx"
        )
        await mission_control.emit_tool_result(
            run_id, agent_name, "linkedin.search", result.model_dump()
        )
        return result.model_dump()

    except Exception as e:
        logger.error(f"LinkedIn search failed: {e}")
        result = NetworkingResponse(
            status="error", query=query, message=str(e)
        )
        await mission_control.emit_tool_result(
            run_id, agent_name, "linkedin.search", result.model_dump()
        )
        return result.model_dump()


def get_status() -> dict:
    """Get LinkedIn tool status for the Neural Link UI."""
    return {
        "name": "LinkedIn Networking",
        "enabled": _is_enabled(),
        "provider": "Tavily (web search)",
    }
