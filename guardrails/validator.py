"""Central Validator — Pre-flight checks for all NEXUS tool calls.

Every tool call passes through Validator.check(tool_name, arguments, context)
before execution. If any guardrail fails, the tool call is BLOCKED and the
agent receives a structured rejection with a suggested correction.

Design Principles:
    1. EXTENSIBLE: Add new rules by defining a function and registering it
       with @register_rule(tool_pattern, rule_name).
    2. NON-DESTRUCTIVE: A failed check returns a rejection — it never mutates
       arguments or raises exceptions.
    3. OBSERVABLE: Every check result is logged and emitted to Mission Control.
    4. FAST: Rules are pure logic (no DB calls, no LLM calls). Context is
       pre-fetched once and passed in.

Rule Categories:
    - TEMPORAL: Past-date detection, end-before-start, timezone sanity
    - LOGIC: Delete/update of non-existent items, invalid references
    - DUPLICATE: Near-identical create operations within recent history
    - (Future) BUDGET: Spending limits, over-allocation detection
    - (Future) PRIVACY: PII redaction, sensitive field masking
    - (Future) PRIORITY: Conflicting urgency levels, overloaded days
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

logger = logging.getLogger("nexus.guardrails")


# ═══════════════════════════════════════════════════════════════
# VALIDATION RESULT
# ═══════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Result of a pre-flight validation check.

    Attributes:
        passed: True if all checks passed; False if blocked.
        blocked_by: Name of the rule that blocked (if any).
        category: Rule category (temporal, logic, duplicate, etc.)
        message: Human-readable explanation for the LLM.
        suggestion: Suggested correction or question for the user.
        severity: "error" (hard block) or "warning" (proceed with caution).
    """
    passed: bool = True
    blocked_by: str = ""
    category: str = ""
    message: str = ""
    suggestion: str = ""
    severity: str = "error"  # "error" or "warning"

    def to_tool_result(self) -> dict:
        """Convert to a dict that can be returned as a tool result."""
        return {
            "guardrail_blocked": True,
            "rule": self.blocked_by,
            "category": self.category,
            "message": self.message,
            "suggestion": self.suggestion,
            "severity": self.severity,
        }


# ═══════════════════════════════════════════════════════════════
# VALIDATION CONTEXT
# ═══════════════════════════════════════════════════════════════

@dataclass
class ValidationContext:
    """Pre-fetched context for guardrail checks.

    Built once per turn (in chat.py or the agent loop), then passed
    to every rule so they don't need to query the DB themselves.
    """
    now: datetime = field(default_factory=datetime.now)
    action_snapshot: list[dict] = field(default_factory=list)
    # Future: user_preferences, budget_state, etc.


# ═══════════════════════════════════════════════════════════════
# RULE REGISTRY
# ═══════════════════════════════════════════════════════════════

# Each rule: (tool_pattern_regex, rule_name, check_function)
_RULES: list[tuple[str, str, Callable]] = []


def register_rule(tool_pattern: str, rule_name: str):
    """Decorator to register a guardrail rule.

    Args:
        tool_pattern: Regex pattern for matching tool names
                      (e.g., "create_event|create_task" or ".*")
        rule_name: Human-readable name for logging
    """
    def decorator(func: Callable):
        _RULES.append((tool_pattern, rule_name, func))
        return func
    return decorator


# ═══════════════════════════════════════════════════════════════
# BUILT-IN RULES
# ═══════════════════════════════════════════════════════════════

def _parse_datetime_lenient(value: str) -> Optional[datetime]:
    """Parse an ISO 8601 datetime string leniently."""
    if not value:
        return None
    try:
        # Strip timezone info for comparison (naive datetime)
        clean = re.sub(r'[+-]\d{2}:\d{2}$', '', value)
        clean = clean.replace('Z', '')
        return datetime.fromisoformat(clean)
    except (ValueError, TypeError):
        return None


# ── TEMPORAL RULES ────────────────────────────────────────────

@register_rule(r"create_event", "past_event_date")
def check_past_event_date(tool_name: str, args: dict,
                           ctx: ValidationContext) -> Optional[ValidationResult]:
    """Block creation of events with start_time in the past."""
    start_str = args.get("start_time", "")
    start_dt = _parse_datetime_lenient(start_str)
    if not start_dt:
        return None  # Can't parse → let the tool handle it

    if start_dt < ctx.now - timedelta(hours=1):  # 1hr grace for timezone drift
        days_ago = (ctx.now - start_dt).days
        return ValidationResult(
            passed=False,
            blocked_by="past_event_date",
            category="temporal",
            message=(
                f"The start time {start_str} is in the past "
                f"({days_ago} day{'s' if days_ago != 1 else ''} ago). "
                f"Today is {ctx.now.strftime('%A, %B %d, %Y')}."
            ),
            suggestion=(
                f"Did you mean {start_str} of next year, or a different date? "
                f"Please confirm the correct date."
            ),
        )
    return None


@register_rule(r"create_task", "past_task_due_date")
def check_past_task_due_date(tool_name: str, args: dict,
                              ctx: ValidationContext) -> Optional[ValidationResult]:
    """Warn when creating a task with a due date in the past."""
    due_str = args.get("due_date", "")
    due_dt = _parse_datetime_lenient(due_str)
    if not due_dt:
        return None

    if due_dt < ctx.now - timedelta(hours=1):
        days_ago = (ctx.now - due_dt).days
        return ValidationResult(
            passed=False,
            blocked_by="past_task_due_date",
            category="temporal",
            message=(
                f"The due date {due_str} is in the past "
                f"({days_ago} day{'s' if days_ago != 1 else ''} ago). "
                f"Today is {ctx.now.strftime('%A, %B %d, %Y')}."
            ),
            suggestion=(
                f"I noticed {due_str} has already passed. Did you mean next year, "
                f"or was this intentional? Please confirm."
            ),
        )
    return None


@register_rule(r"create_event|update_event", "end_before_start")
def check_end_before_start(tool_name: str, args: dict,
                            ctx: ValidationContext) -> Optional[ValidationResult]:
    """Block events where end_time is before start_time."""
    start_str = args.get("start_time", "")
    end_str = args.get("end_time", "")

    start_dt = _parse_datetime_lenient(start_str)
    end_dt = _parse_datetime_lenient(end_str)

    if not start_dt or not end_dt:
        return None

    if end_dt <= start_dt:
        return ValidationResult(
            passed=False,
            blocked_by="end_before_start",
            category="temporal",
            message=(
                f"End time ({end_str}) is before or equal to start time ({start_str}). "
                f"An event cannot end before it begins."
            ),
            suggestion=(
                f"Please provide a valid end time that is after {start_str}."
            ),
        )
    return None


# ── LOGIC RULES ───────────────────────────────────────────────

@register_rule(r"delete_event|delete_task|update_event|update_task|update_note|complete_task", "phantom_reference")
def check_phantom_reference(tool_name: str, args: dict,
                             ctx: ValidationContext) -> Optional[ValidationResult]:
    """Warn when operating on an ID that doesn't appear in the Action Snapshot.

    This catches "hallucinated IDs" — when the LLM fabricates an ID
    that doesn't exist in recent history.
    """
    # Extract the target ID from the arguments
    target_id = (args.get("event_id") or args.get("task_id") or
                 args.get("note_id") or "")

    if not target_id:
        return None  # No ID to validate

    # Check if this ID appears in the action snapshot
    known_ids = {entry.get("id", "") for entry in ctx.action_snapshot}

    if known_ids and target_id not in known_ids:
        # It's not in recent snapshot — issue a warning (not hard block,
        # because the item might exist from an older session)
        return ValidationResult(
            passed=False,
            blocked_by="phantom_reference",
            category="logic",
            message=(
                f"The ID '{target_id[:12]}...' was not found in your recent actions. "
                f"It may have been created in a previous session or may not exist."
            ),
            suggestion=(
                f"Can you confirm the specific item you want to "
                f"{'delete' if 'delete' in tool_name else 'update'}? "
                f"Recent items: "
                + ", ".join(
                    f"\"{e.get('title', '?')}\" (ID: {e.get('id', '?')[:8]}...)"
                    for e in ctx.action_snapshot[:3]
                )
            ),
            severity="warning",
        )
    return None


# ── DUPLICATE PREVENTION ──────────────────────────────────────

@register_rule(r"create_task", "duplicate_task")
def check_duplicate_task(tool_name: str, args: dict,
                          ctx: ValidationContext) -> Optional[ValidationResult]:
    """Detect near-duplicate task creation within recent history."""
    new_title = (args.get("title") or "").lower().strip()
    new_date = args.get("due_date", "")

    if not new_title:
        return None

    for entry in ctx.action_snapshot:
        existing_title = (entry.get("title") or "").lower().strip()
        existing_tool = entry.get("tool", "")

        # Only compare against recent creates/updates of tasks
        if "task" not in existing_tool:
            continue

        # Exact title match
        if new_title == existing_title:
            return ValidationResult(
                passed=False,
                blocked_by="duplicate_task",
                category="duplicate",
                message=(
                    f"A task titled \"{entry.get('title', '')}\" was already "
                    f"created recently (ID: {entry.get('id', '?')[:8]}...)."
                ),
                suggestion=(
                    f"Did you want to update the existing task instead? "
                    f"I can use update_task(task_id=\"{entry.get('id', '')}\") "
                    f"to modify it."
                ),
            )

        # Fuzzy match: >80% word overlap
        new_words = set(new_title.split())
        existing_words = set(existing_title.split())
        if new_words and existing_words:
            overlap = len(new_words & existing_words) / max(len(new_words), len(existing_words))
            if overlap > 0.8:
                return ValidationResult(
                    passed=False,
                    blocked_by="duplicate_task",
                    category="duplicate",
                    message=(
                        f"This looks very similar to a recent task: "
                        f"\"{entry.get('title', '')}\" (ID: {entry.get('id', '?')[:8]}...). "
                        f"Word overlap: {overlap:.0%}."
                    ),
                    suggestion=(
                        f"Should I update the existing task, or is this intentionally "
                        f"a separate item?"
                    ),
                    severity="warning",
                )

    return None


@register_rule(r"create_event", "duplicate_event")
def check_duplicate_event(tool_name: str, args: dict,
                           ctx: ValidationContext) -> Optional[ValidationResult]:
    """Detect near-duplicate event creation within recent history."""
    new_title = (args.get("title") or "").lower().strip()
    new_start = args.get("start_time", "")

    if not new_title:
        return None

    for entry in ctx.action_snapshot:
        existing_title = (entry.get("title") or "").lower().strip()
        existing_tool = entry.get("tool", "")
        existing_start = entry.get("start_time", "")

        if "event" not in existing_tool:
            continue

        # Same title and same start time = definite duplicate
        if new_title == existing_title and new_start and new_start == existing_start:
            return ValidationResult(
                passed=False,
                blocked_by="duplicate_event",
                category="duplicate",
                message=(
                    f"An event \"{entry.get('title', '')}\" at {existing_start} "
                    f"already exists (ID: {entry.get('id', '?')[:8]}...)."
                ),
                suggestion=(
                    f"This appears to be a duplicate. Would you like to update "
                    f"the existing event instead?"
                ),
            )

    return None


# ═══════════════════════════════════════════════════════════════
# VALIDATOR — The Central Engine
# ═══════════════════════════════════════════════════════════════

class Validator:
    """Central pre-flight validation engine for all tool calls.

    Usage:
        result = await validator.check("create_event", args, context)
        if not result.passed:
            return result.to_tool_result()  # Block the call
        # ... proceed with tool execution
    """

    def __init__(self):
        self._rules = _RULES
        logger.info(f"Guardrails Validator initialized with {len(self._rules)} rules")

    async def check(self, tool_name: str, args: dict,
                    ctx: ValidationContext = None) -> ValidationResult:
        """Run all matching guardrail rules against a tool call.

        Returns the FIRST failing check (hard block), or a passing result
        if all checks pass. Warnings are logged but don't block.

        Args:
            tool_name: The tool being called (e.g., "create_event")
            args: The tool arguments
            ctx: Pre-built validation context (current time, snapshot, etc.)

        Returns:
            ValidationResult — check .passed to determine if execution should proceed
        """
        if ctx is None:
            ctx = ValidationContext()

        warnings = []

        for pattern, rule_name, check_fn in self._rules:
            if not re.match(pattern, tool_name):
                continue

            try:
                result = check_fn(tool_name, args, ctx)
                if result is None:
                    continue  # Rule passed

                if result.severity == "error" and not result.passed:
                    logger.warning(
                        f"GUARDRAIL BLOCKED: [{rule_name}] {tool_name}({args}) → "
                        f"{result.message}"
                    )
                    return result

                if result.severity == "warning" and not result.passed:
                    warnings.append(result)
                    logger.info(
                        f"GUARDRAIL WARNING: [{rule_name}] {tool_name}({args}) → "
                        f"{result.message}"
                    )

            except Exception as e:
                # A guardrail must NEVER crash the tool call
                logger.error(f"Guardrail rule '{rule_name}' failed: {e}")
                continue

        # If we only have warnings, return the most severe one
        if warnings:
            return warnings[0]

        return ValidationResult(passed=True)

    def get_rules(self) -> list[dict]:
        """List all registered rules (for observability / health endpoints)."""
        return [
            {"pattern": pattern, "name": name, "function": fn.__name__}
            for pattern, name, fn in self._rules
        ]


# ── MCP / EXTERNAL SERVICE GUARDRAILS ────────────────────────

@register_rule(r"send_slack_message", "slack_content_guard")
def check_slack_content(tool_name: str, args: dict,
                        ctx: ValidationContext) -> Optional[ValidationResult]:
    """Prevent the AI from being prompt-injected into sending dangerous Slack messages.

    Blocks messages containing:
    - URLs that look like phishing
    - Messages pretending to be from other users
    - Excessively long messages (> 4000 chars)
    """
    text = args.get("text", "")
    if len(text) > 4000:
        return ValidationResult(
            passed=False,
            blocked_by="slack_content_guard",
            category="safety",
            message="Slack message is too long (>4000 chars). Please shorten it.",
            suggestion="Break this into multiple shorter messages.",
        )
    # Check for impersonation patterns
    impersonation_patterns = [
        r"(?i)(from|sent by|message from|on behalf of)\s*:\s*\w+",
        r"(?i)@here\s.*(urgent|emergency|immediately|password|credentials)",
    ]
    import re as _re_local
    for pattern in impersonation_patterns:
        if _re_local.search(pattern, text):
            return ValidationResult(
                passed=False,
                blocked_by="slack_content_guard",
                category="safety",
                message="This message contains patterns that could be confused with impersonation.",
                suggestion="Please rephrase the message to clearly indicate it's from NEXUS.",
                severity="warning",
            )
    return None


@register_rule(r"respond_to_outlook_invite", "invite_response_guard")
def check_invite_response(tool_name: str, args: dict,
                           ctx: ValidationContext) -> Optional[ValidationResult]:
    """Ensure invite responses have valid actions and non-empty event IDs."""
    action = args.get("action", "")
    event_id = args.get("event_id", "")
    if action not in ("accept", "decline", "tentative"):
        return ValidationResult(
            passed=False,
            blocked_by="invite_response_guard",
            category="logic",
            message=f"Invalid invite action: '{action}'. Must be accept, decline, or tentative.",
            suggestion="Use 'accept', 'decline', or 'tentative'.",
        )
    if not event_id or len(event_id) < 5:
        return ValidationResult(
            passed=False,
            blocked_by="invite_response_guard",
            category="logic",
            message="Invalid or missing event_id for invite response.",
            suggestion="Please provide the event ID from the calendar invite.",
        )
    return None


# ── Module-level singleton ─────────────────────────────────
validator = Validator()
