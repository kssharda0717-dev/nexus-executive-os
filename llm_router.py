"""LLM Router — Industrial-grade multi-model, multi-project failover for NEXUS.

Architecture (Post-RCA 2026-04-02):
    Phase 1 — Graceful Degradation:
        Custom AllProvidersExhausted exception (no raw tracebacks ever)
        Circuit Breaker: after consecutive full-cascade failures, short-circuits
        to deterministic mode for CIRCUIT_BREAKER_COOLDOWN_S (5 min) to let the
        API breathe. Prevents "4 failing requests deepening the hole."

    Phase 2 — Multi-Key Rotation ("Double Tank"):
        Two GCP projects (GEMINI_API_KEY + GEMINI_API_KEY_SECONDARY), each with
        4 models = 8 independent provider endpoints. True physical redundancy:
        when Project A's aggregate quota is hit, Project B is a separate bucket.

    NOT OpenRouter. NOT LiteLLM. Zero new dependencies. Native google-genai SDK
    throughout. Full FunctionDeclaration + system_instruction support preserved.

Failover cascade (8 providers, 2 projects):
    Project A: 2.5-flash → 2.5-flash-lite → 2.0-flash → 2.0-flash-lite
    Project B: 2.5-flash → 2.5-flash-lite → 2.0-flash → 2.0-flash-lite
    Circuit Breaker → AllProvidersExhausted (caught by callers)
"""

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY, GEMINI_API_KEY_SECONDARY,
    GEMINI_MAX_CONCURRENT, GEMINI_MIN_CALL_SPACING_S,
    GEMINI_429_BASE_DELAY_S, GEMINI_429_MAX_DELAY_S, GEMINI_JITTER_S,
    CIRCUIT_BREAKER_COOLDOWN_S, CIRCUIT_BREAKER_TRIP_THRESHOLD,
)

logger = logging.getLogger("nexus.llm_router")


# ═══════════════════════════════════════════════════════════════
# CUSTOM EXCEPTION — replaces raw ClientError propagation
# ═══════════════════════════════════════════════════════════════

class AllProvidersExhausted(Exception):
    """Raised when every provider in every project has been exhausted.

    This is a STRUCTURED signal — callers catch it and fall back to
    deterministic defaults. It NEVER reaches the user as a raw traceback.
    """

    def __init__(self, message: str = "All LLM providers exhausted",
                 last_error: Exception | None = None,
                 circuit_breaker_tripped: bool = False):
        self.last_error = last_error
        self.circuit_breaker_tripped = circuit_breaker_tripped
        super().__init__(message)


# ═══════════════════════════════════════════════════════════════
# MODEL PROVIDER — Per-model rate limiter + health state
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelProvider:
    """A single Gemini model endpoint with its own rate-limit state."""
    model_name: str
    project: str                       # "primary" or "secondary"
    tier: str                          # "primary", "secondary", "fallback"
    quality: int = 100                 # 100=best, 50=good, 25=basic
    rpm_limit: int = 10                # conservative default
    rpd_limit: int = 1500

    # Mutable state
    _call_count: int = field(default=0, repr=False)
    _daily_calls: int = field(default=0, repr=False)
    _last_call_time: float = field(default=0.0, repr=False)
    _cooldown_until: float = field(default=0.0, repr=False)
    _consecutive_429s: int = field(default=0, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def display_name(self) -> str:
        """Human-readable name for narration."""
        tag = "A" if self.project == "primary" else "B"
        return f"{self.model_name}@{tag}"

    @property
    def is_available(self) -> bool:
        """Check if this provider is currently usable (not in cooldown)."""
        if self._cooldown_until > 0:
            if time.monotonic() < self._cooldown_until:
                return False
            # Cooldown expired — reset
            self._cooldown_until = 0.0
            self._consecutive_429s = 0
        return True

    def mark_success(self):
        """Record a successful call."""
        self._consecutive_429s = 0
        self._call_count += 1
        self._daily_calls += 1
        self._last_call_time = time.monotonic()

    def mark_rate_limited(self, retry_after: float | None = None):
        """Record a 429 and enter cooldown.

        Cooldown strategy:
            - If the API tells us a retryDelay, use it + 2s buffer
            - Otherwise, exponential: 15s, 30s, 60s, 120s per consecutive 429
            - After 4 consecutive 429s, disable this provider for 5 minutes
        """
        self._consecutive_429s += 1

        if retry_after:
            cooldown = retry_after + 2.0
        elif self._consecutive_429s >= 4:
            cooldown = 300.0  # 5 minutes — this model's quota is likely gone
        else:
            cooldown = min(15.0 * (2 ** (self._consecutive_429s - 1)), 120.0)

        self._cooldown_until = time.monotonic() + cooldown
        logger.warning(
            f"[{self.display_name}] 429 #{self._consecutive_429s} — "
            f"cooldown {cooldown:.0f}s"
        )

    async def enforce_spacing(self):
        """Wait for minimum spacing between calls to this provider."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            min_spacing = GEMINI_MIN_CALL_SPACING_S + random.uniform(0, GEMINI_JITTER_S)
            if elapsed < min_spacing:
                wait = min_spacing - elapsed
                logger.debug(f"[{self.display_name}] spacing wait {wait:.2f}s")
                await asyncio.sleep(wait)


# ═══════════════════════════════════════════════════════════════
# SMART TIERS — Task-type → model quality mapping
# ═══════════════════════════════════════════════════════════════

class TaskTier:
    """Maps agent roles to minimum model quality requirements."""
    ORCHESTRATOR = 90   # Primary agent: classify, plan, synthesize
    SUB_AGENT = 50      # Planning/Learning/LifeAdmin: tool-calling loops
    PROACTIVE = 25      # Background checks: overdue, briefings


# Agent name → task tier
AGENT_TIER_MAP = {
    "primary_agent": TaskTier.ORCHESTRATOR,
    "general_agent": TaskTier.ORCHESTRATOR,
    "planning_agent": TaskTier.SUB_AGENT,
    "learning_agent": TaskTier.SUB_AGENT,
    "life_admin_agent": TaskTier.SUB_AGENT,
    "proactive_monitor": TaskTier.PROACTIVE,
}


# ═══════════════════════════════════════════════════════════════
# CIRCUIT BREAKER — Project-level quota protection
# ═══════════════════════════════════════════════════════════════

class CircuitBreaker:
    """Tracks consecutive full-cascade failures and trips to prevent
    hammering an already-exhausted API.

    States:
        CLOSED  — normal operation, cascade runs fully
        OPEN    — tripped, all calls short-circuit to AllProvidersExhausted
        (auto-resets after CIRCUIT_BREAKER_COOLDOWN_S)
    """

    def __init__(self):
        self._consecutive_failures: int = 0
        self._tripped_until: float = 0.0
        self._trip_count: int = 0  # lifetime trip counter for observability

    @property
    def is_tripped(self) -> bool:
        if self._tripped_until > 0:
            if time.monotonic() < self._tripped_until:
                return True
            # Auto-reset
            self._tripped_until = 0.0
            self._consecutive_failures = 0
        return False

    @property
    def remaining_cooldown_s(self) -> float:
        if self._tripped_until > 0:
            remaining = self._tripped_until - time.monotonic()
            return max(0.0, remaining)
        return 0.0

    def record_cascade_failure(self):
        """Record a full-cascade failure. Trip the breaker if threshold met."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= CIRCUIT_BREAKER_TRIP_THRESHOLD:
            self._tripped_until = time.monotonic() + CIRCUIT_BREAKER_COOLDOWN_S
            self._trip_count += 1
            logger.critical(
                f"CIRCUIT BREAKER TRIPPED (trip #{self._trip_count}). "
                f"All providers exhausted {self._consecutive_failures} times. "
                f"Entering deterministic mode for {CIRCUIT_BREAKER_COOLDOWN_S:.0f}s."
            )
            return True
        return False

    def record_success(self):
        """A successful call resets the failure counter."""
        self._consecutive_failures = 0

    def get_status(self) -> dict:
        return {
            "state": "OPEN" if self.is_tripped else "CLOSED",
            "consecutive_failures": self._consecutive_failures,
            "lifetime_trips": self._trip_count,
            "remaining_cooldown_s": round(self.remaining_cooldown_s, 1),
        }


# ═══════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════

def _is_rate_limit_error(e: Exception) -> bool:
    """Detect 429 / RESOURCE_EXHAUSTED errors."""
    msg = str(e).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg


def _parse_retry_delay(e: Exception) -> float | None:
    """Extract retryDelay (seconds) from Gemini error message."""
    match = re.search(r'retry.*?(\d+\.?\d*)\s*s', str(e), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


# ═══════════════════════════════════════════════════════════════
# LLM ROUTER — The core failover engine (Phase 1 + Phase 2)
# ═══════════════════════════════════════════════════════════════

# Models available in each project
_MODEL_SPECS = [
    ("gemini-2.5-flash",      "primary",   100, 10, 500),
    ("gemini-2.5-flash-lite", "secondary",  75, 10, 500),
    ("gemini-2.0-flash",      "fallback",   60, 15, 1500),
    ("gemini-2.0-flash-lite", "fallback",   40, 15, 1500),
]


class LLMRouter:
    """Multi-model, multi-project failover router using native Gemini SDK.

    Cascade order (up to 8 providers):
        Project A: gemini-2.5-flash → 2.5-flash-lite → 2.0-flash → 2.0-flash-lite
        Project B: gemini-2.5-flash → 2.5-flash-lite → 2.0-flash → 2.0-flash-lite

    Each model × project combination has INDEPENDENT rate limits.
    Circuit breaker trips after repeated full-cascade failures.
    """

    def __init__(self):
        self._clients: dict[str, genai.Client] = {}
        self._semaphore = asyncio.Semaphore(GEMINI_MAX_CONCURRENT)
        self._circuit_breaker = CircuitBreaker()

        # Build provider cascade: Project A first, then Project B
        self.providers: list[ModelProvider] = []

        # Project A — always available (primary key)
        if GEMINI_API_KEY:
            for model, tier, quality, rpm, rpd in _MODEL_SPECS:
                self.providers.append(ModelProvider(
                    model_name=model, project="primary",
                    tier=tier, quality=quality, rpm_limit=rpm, rpd_limit=rpd,
                ))

        # Project B — only if secondary key is configured
        if GEMINI_API_KEY_SECONDARY:
            for model, tier, quality, rpm, rpd in _MODEL_SPECS:
                self.providers.append(ModelProvider(
                    model_name=model, project="secondary",
                    tier=tier, quality=quality, rpm_limit=rpm, rpd_limit=rpd,
                ))
            logger.info(
                f"LLM Router initialized: {len(self.providers)} providers "
                f"across 2 projects (Double Tank active)"
            )
        else:
            logger.info(
                f"LLM Router initialized: {len(self.providers)} providers "
                f"(single project — set GEMINI_API_KEY_SECONDARY for redundancy)"
            )

        # Track which provider last served each agent (for narration)
        self._active_provider: dict[str, str] = {}

    def _get_client(self, project: str) -> genai.Client:
        """Get or create the genai.Client for a project."""
        if project not in self._clients:
            key = GEMINI_API_KEY if project == "primary" else GEMINI_API_KEY_SECONDARY
            if not key:
                raise AllProvidersExhausted(
                    f"No API key configured for project '{project}'"
                )
            self._clients[project] = genai.Client(api_key=key)
        return self._clients[project]

    def get_providers_for_tier(self, min_quality: int) -> list[ModelProvider]:
        """Get providers ordered by preference: qualified first, then fallbacks.

        Always returns the FULL cascade so failover works even when
        the best-quality model 429s. Order: qualified+available → other available → all.
        """
        qualified = [p for p in self.providers if p.quality >= min_quality and p.is_available]
        fallbacks = [p for p in self.providers if p.is_available and p not in qualified]

        if qualified or fallbacks:
            return qualified + fallbacks

        # Nuclear fallback: all providers, least-cooldown first
        return sorted(self.providers, key=lambda p: p._cooldown_until)

    async def generate_content(
        self,
        *,
        contents: list,
        system_instruction: str = "",
        tools: list | None = None,
        temperature: float = 0.2,
        agent_name: str = "unknown",
        run_id: str = "",
    ) -> object:
        """Route a generate_content call through the failover chain.

        This is the SINGLE entry point for ALL LLM calls in the system.

        Returns: A native google.genai response object.
        Raises: AllProvidersExhausted if every provider fails (never a raw ClientError).
        """
        from mission_control import mission_control

        # ── Circuit Breaker Check ──
        if self._circuit_breaker.is_tripped:
            remaining = self._circuit_breaker.remaining_cooldown_s
            logger.warning(
                f"[{agent_name}] Circuit breaker OPEN — "
                f"{remaining:.0f}s remaining. Short-circuiting to deterministic mode."
            )
            await mission_control.emit_narration(
                run_id, agent_name,
                f"High demand detected. Operating in resilient autonomous mode "
                f"({remaining:.0f}s cooldown remaining)."
            )
            raise AllProvidersExhausted(
                "Circuit breaker tripped — deterministic mode active",
                circuit_breaker_tripped=True,
            )

        min_quality = AGENT_TIER_MAP.get(agent_name, TaskTier.SUB_AGENT)
        providers = self.get_providers_for_tier(min_quality)

        if not providers:
            raise AllProvidersExhausted("No providers configured")

        last_error: Exception | None = None
        current_project: str | None = None
        project_switch_narrated = False

        for provider in providers:
            # ── Narrate project switch (A → B) ──
            if current_project and provider.project != current_project and not project_switch_narrated:
                project_label = "backup" if provider.project == "secondary" else "primary"
                await mission_control.emit_narration(
                    run_id, agent_name,
                    f"Primary cluster exhausted. Engaging {project_label} infrastructure..."
                )
                await mission_control.emit_thought(
                    run_id, agent_name,
                    f"Switching from Project {current_project.upper()} to "
                    f"Project {provider.project.upper()}"
                )
                project_switch_narrated = True
            current_project = provider.project

            # Global concurrency gate
            await self._semaphore.acquire()
            try:
                # Per-provider spacing
                await provider.enforce_spacing()

                try:
                    client = self._get_client(provider.project)
                    response = client.models.generate_content(
                        model=provider.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction or None,
                            tools=tools,
                            temperature=temperature,
                        ),
                    )

                    provider.mark_success()
                    self._circuit_breaker.record_success()

                    # Track active provider for observability
                    prev = self._active_provider.get(agent_name)
                    self._active_provider[agent_name] = provider.display_name
                    if prev and prev != provider.display_name:
                        await mission_control.emit_thought(
                            run_id, agent_name,
                            f"Switched to {provider.display_name} (was: {prev})"
                        )

                    return response

                except Exception as e:
                    last_error = e

                    if _is_rate_limit_error(e):
                        retry_after = _parse_retry_delay(e)
                        provider.mark_rate_limited(retry_after)

                        await mission_control.emit_thought(
                            run_id, agent_name,
                            f"{provider.display_name} hit rate limit — "
                            f"failing over to next model..."
                        )
                        logger.warning(
                            f"[{agent_name}] {provider.display_name} → 429. "
                            f"Trying next provider..."
                        )
                        continue  # Try next provider

                    else:
                        # Non-rate-limit error — don't cascade, just wrap and raise
                        logger.error(
                            f"[{agent_name}] {provider.display_name} → "
                            f"non-429 error: {e}"
                        )
                        raise AllProvidersExhausted(
                            f"Non-rate-limit error from {provider.display_name}: {e}",
                            last_error=e,
                        )

            finally:
                self._semaphore.release()

        # ── All providers exhausted ──
        # Last resort: wait for the shortest cooldown and retry once
        if last_error and _is_rate_limit_error(last_error):
            best_provider = min(self.providers, key=lambda p: p._cooldown_until)
            wait_time = max(0, best_provider._cooldown_until - time.monotonic())

            if 0 < wait_time <= GEMINI_429_MAX_DELAY_S:
                await mission_control.emit_narration(
                    run_id, agent_name,
                    f"All models across all clusters are cooling down. "
                    f"Waiting {wait_time:.0f}s for {best_provider.display_name} to recover..."
                )
                logger.warning(
                    f"[{agent_name}] All providers 429'd. "
                    f"Waiting {wait_time:.1f}s for {best_provider.display_name}..."
                )
                await asyncio.sleep(wait_time + 1.0)

                # One final attempt
                await self._semaphore.acquire()
                try:
                    await best_provider.enforce_spacing()
                    client = self._get_client(best_provider.project)
                    response = client.models.generate_content(
                        model=best_provider.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction or None,
                            tools=tools,
                            temperature=temperature,
                        ),
                    )
                    best_provider.mark_success()
                    self._circuit_breaker.record_success()
                    self._active_provider[agent_name] = best_provider.display_name
                    return response
                except Exception as e:
                    last_error = e
                finally:
                    self._semaphore.release()

        # ── Total failure — trip circuit breaker ──
        tripped = self._circuit_breaker.record_cascade_failure()
        if tripped:
            await mission_control.emit_narration(
                run_id, agent_name,
                "High demand detected. Switching to resilient autonomous planning mode. "
                "I'll use my built-in intelligence while the AI infrastructure recovers."
            )

        provider_count = len(self.providers)
        project_count = len(set(p.project for p in self.providers))
        raise AllProvidersExhausted(
            f"All {provider_count} providers across {project_count} project(s) exhausted. "
            f"Circuit breaker: {'TRIPPED' if tripped else 'armed'}.",
            last_error=last_error,
        )

    def get_status(self) -> dict:
        """Get current provider health status (for /health endpoint)."""
        return {
            "providers": [
                {
                    "model": p.model_name,
                    "project": p.project,
                    "tier": p.tier,
                    "quality": p.quality,
                    "available": p.is_available,
                    "total_calls": p._call_count,
                    "consecutive_429s": p._consecutive_429s,
                }
                for p in self.providers
            ],
            "active_assignments": dict(self._active_provider),
            "circuit_breaker": self._circuit_breaker.get_status(),
            "dual_project": bool(GEMINI_API_KEY_SECONDARY),
        }


# ── Module-level singleton ────────────────────────────────────
llm_router = LLMRouter()
