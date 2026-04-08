"""NEXUS Guardrails — Neurosymbolic validation layer.

Provides pre-flight checks for all tool calls to prevent:
    - Temporal blindness (past-date events, inverted time ranges)
    - Hallucinated compliance (deleting non-existent items)
    - Duplicate creation (identical title+date within recent history)
    - Logic inconsistency (end_time before start_time)

Architecture:
    Extensible rule-based system. Each guardrail is a self-contained
    validation function. New rules (budget limits, privacy redaction,
    priority conflicts) can be added without modifying the agent code.
"""

from guardrails.validator import Validator, ValidationResult, validator

__all__ = ["Validator", "ValidationResult", "validator"]
