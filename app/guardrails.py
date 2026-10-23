"""Defense-in-depth guardrails for the Data Agents metadata capstone.

Drop-in compatible with existing imports:
    from app.guardrails import validate_question, filter_tools

Additional guards must be explicitly wired into UI/agent/MCP server paths.
Standalone SQL commands are also checked by validate_question(), provided
the caller invokes it. Any SQL execution tool MUST independently call
validate_sql_query() at its trusted MCP execution boundary. Regex checks are defense in depth,
not a substitute for SQL parsing and read-only database permissions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, TypeVar

LOGGER = logging.getLogger(__name__)

SAFE_TOOLS = frozenset({
    "search_catalog",
    "recommend_data_products",
    "get_asset",
    "analyze_impact",
    "detect_schema_drift",
    "semantic_search",
    "search_glossary",
    "propose_schema_change",
})
BLOCKED_TOOLS = frozenset({"resolve_schema_change"})
MAX_QUESTION = 4000

# Block obvious data modification and administrative SQL operations.
# Apply to SQL destined for execution, NOT natural-language user questions.
FORBIDDEN_SQL = re.compile(
    r"\b(?:DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|CREATE|"
    r"GRANT|REVOKE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|COPY)\b",
    re.IGNORECASE,
)
_SQL_START = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)

# This is a warning signal, NOT a complete prompt-injection defense.
# It deliberately avoids blocking normal questions about injection attacks.
_INJECTION_PATTERNS = (
    re.compile(r"ignore (?:all |any )?(?:previous|prior|system|developer) instructions", re.I),
    re.compile(r"(?:reveal|print|show|disclose) (?:the |your )?(?:system prompt|developer prompt|api key|secret key)", re.I),
    re.compile(r"(?:disable|bypass|override) (?:all |the )?(?:guardrails|safety checks|approval checks)", re.I),
)
_SECRET_PATTERNS = (
    re.compile(r"\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*[\"']?([A-Za-z0-9_./+=-]{8,})", re.I),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
)
_MAX_TOOL_RESULT_CHARS = 6000
_MAX_TOOL_CALLS = 12
_AGENT_TIMEOUT_SECONDS = 45


class GuardrailError(ValueError):
    """A guardrail rejected an operation."""


@dataclass(frozen=True)
class GuardrailConfig:
    max_question_chars: int = MAX_QUESTION
    max_tool_result_chars: int = _MAX_TOOL_RESULT_CHARS
    max_tool_calls: int = _MAX_TOOL_CALLS
    agent_timeout_seconds: float = _AGENT_TIMEOUT_SECONDS
    block_suspicious_questions: bool = False


def validate_sql_query(sql: str) -> str:
    """Reject obvious unsafe SQL; return a normalized read-only SQL query.

    This conservative regex-based validator is intended as an extra safeguard.
    For actual SQL execution, also enforce read-only database permissions,
    a single-statement SQL parser, and database-side resource limits.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise GuardrailError("SQL query cannot be empty")
    sql = sql.strip()
    if FORBIDDEN_SQL.search(sql):
        raise GuardrailError(
            "Unsafe SQL operation detected. Only read-only queries are permitted."
        )
    if not _SQL_START.match(sql):
        raise GuardrailError("SQL query must start with SELECT or WITH")
    # Permit one optional terminal semicolon, never interior semicolons.
    statement = sql[:-1].rstrip() if sql.endswith(";") else sql
    if ";" in statement:
        raise GuardrailError("Multiple SQL statements are not allowed")
    return statement


def detect_prompt_injection(text: str) -> list[str]:
    """Return heuristic signals; absence of signals does not imply safety."""
    if not isinstance(text, str):
        return []
    return [f"pattern_{i}" for i, pat in enumerate(_INJECTION_PATTERNS, 1) if pat.search(text)]


def looks_like_sql_command(text: str) -> bool:
    """Identify standalone SQL commands, not natural-language impact questions."""
    return bool(re.match(
        r"^\s*(?:SELECT|WITH|DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|CREATE|"
        r"GRANT|REVOKE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|COPY)\b",
        text, re.IGNORECASE,
    ))


def validate_question(question: str, cfg: GuardrailConfig | None = None) -> str:
    """Preserves original API: returns normalized question or raises ValueError."""
    cfg = cfg or GuardrailConfig()
    if not isinstance(question, str) or not question.strip():
        raise GuardrailError("Question is empty")
    question = question.strip()
    if len(question) > cfg.max_question_chars:
        raise GuardrailError(f"Question exceeds {cfg.max_question_chars} characters")
    if any(ord(c) < 32 and c not in "\n\r\t" for c in question):
        raise GuardrailError("Question contains disallowed control characters")
    # Block standalone SQL commands at the chat input boundary. Natural-language
    # questions such as "What happens if we DROP TABLE customer?" remain allowed.
    if looks_like_sql_command(question):
        validate_sql_query(question)
    signals = detect_prompt_injection(question)
    if signals:
        LOGGER.warning("Suspicious user input detected: %s", ", ".join(signals))
        if cfg.block_suspicious_questions:
            raise GuardrailError("Question contains suspicious instruction patterns")
    return question


def filter_tools(tools: Iterable[Any]) -> list[Any]:
    """Only expose approved tools to an LLM; never expose human approval tool."""
    return [
        tool for tool in tools
        if getattr(tool, "name", None) in SAFE_TOOLS
        and getattr(tool, "name", None) not in BLOCKED_TOOLS
    ]


def require_agent_tool(tool_name: str) -> None:
    """Enforce tool authorization at a trusted invocation boundary."""
    if tool_name not in SAFE_TOOLS or tool_name in BLOCKED_TOOLS:
        raise GuardrailError(f"Agent cannot invoke tool: {tool_name}")


def validate_asset_access(
    asset_id: str,
    allowed_asset_ids: set[str] | frozenset[str],
) -> str:
    """Example asset-level allowlist; caller must derive allowed IDs from identity."""
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise GuardrailError("Asset ID is required")
    if asset_id not in allowed_asset_ids:
        raise GuardrailError("Access denied for requested asset")
    return asset_id


def validate_schema_change_decision(
    *,
    request: dict[str, Any] | None,
    decision: str,
    reviewer_id: str,
    authorized_reviewers: set[str] | frozenset[str],
    expected_version: int | str | None = None,
) -> None:
    """Server-side checks for human approval; call from MCP resolution handler.

    authorized_reviewers must come from verified identity/roles, not a text box.
    Caller must enforce atomic compare-and-set when saving the catalog.
    """
    if not request or request.get("status") != "PENDING":
        raise GuardrailError("Schema change request is not pending")
    if decision not in {"APPROVE", "REJECT"}:
        raise GuardrailError("Invalid approval decision")
    if not reviewer_id or reviewer_id not in authorized_reviewers:
        raise GuardrailError("Reviewer is not authorized")
    if expected_version is not None and request.get("version") != expected_version:
        raise GuardrailError("Proposal version changed since review")


def redact_sensitive_text(text: str) -> str:
    """Best-effort redaction of obvious credential patterns, not full DLP."""
    if not isinstance(text, str):
        raise GuardrailError("Output must be text")
    result = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            result = pattern.sub(lambda m: m.group(0).replace(m.group(1), "[REDACTED]"), result)
        else:
            result = pattern.sub("[REDACTED]", result)
    return result


def sanitize_tool_result(result: Any, cfg: GuardrailConfig | None = None) -> str:
    """Bound tool-output size; treat returned text as untrusted reference data."""
    cfg = cfg or GuardrailConfig()
    if isinstance(result, str):
        content = result
    else:
        content = json.dumps(result, ensure_ascii=False, default=str)
    content = redact_sensitive_text(content)
    if len(content) > cfg.max_tool_result_chars:
        content = content[:cfg.max_tool_result_chars] + "\n[TRUNCATED]"
    return content


def validate_tool_call_budget(call_count: int, cfg: GuardrailConfig | None = None) -> None:
    cfg = cfg or GuardrailConfig()
    if call_count >= cfg.max_tool_calls:
        raise GuardrailError(f"Maximum tool calls ({cfg.max_tool_calls}) reached")


T = TypeVar("T")


async def run_with_timeout(
    operation: Awaitable[T], cfg: GuardrailConfig | None = None
) -> T:
    """Wrap an awaitable at its actual execution point to enforce timeout."""
    cfg = cfg or GuardrailConfig()
    try:
        return await asyncio.wait_for(operation, timeout=cfg.agent_timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise GuardrailError(
            f"Agent execution exceeded {cfg.agent_timeout_seconds:g} seconds"
        ) from exc


def audit_event(event: str, **fields: Any) -> None:
    """Emit minimal structured events; do not log prompts, secrets or PII."""
    allowed = {"agent", "tool", "outcome", "request_id", "duration_ms"}
    payload = {key: value for key, value in fields.items() if key in allowed}
    LOGGER.info("guardrail_audit %s", json.dumps({"event": event, **payload}, default=str))
