"""
air-crewai-trust — Consent Gate

Intercepts destructive or sensitive tool calls and holds them
pending user approval. Classifies tools by risk level and
prompts the user for consent via the console.

Flow:
1. before_tool_call fires
2. ConsentGate checks if tool requires consent
3. If yes: prompts user for approval
4. If approved: tool executes normally
5. If rejected/timeout: tool call is blocked
6. All decisions are logged to the audit ledger
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from .audit_ledger import AuditLedger
from .config import RISK_ORDER, ConsentGateConfig, RiskLevel

# Default risk classification for common tool patterns
TOOL_RISK_MAP: dict[str, RiskLevel] = {
    # Critical — arbitrary code execution
    "exec": RiskLevel.CRITICAL,
    "spawn": RiskLevel.CRITICAL,
    "shell": RiskLevel.CRITICAL,
    "run_command": RiskLevel.CRITICAL,
    "execute": RiskLevel.CRITICAL,
    # High — filesystem writes, destructive actions
    "fs_write": RiskLevel.HIGH,
    "fs_delete": RiskLevel.HIGH,
    "file_write": RiskLevel.HIGH,
    "file_delete": RiskLevel.HIGH,
    "apply_patch": RiskLevel.HIGH,
    "rm": RiskLevel.HIGH,
    "rmdir": RiskLevel.HIGH,
    "git_push": RiskLevel.HIGH,
    "deploy": RiskLevel.HIGH,
    # Medium — communication, network
    "send_email": RiskLevel.MEDIUM,
    "email_send": RiskLevel.MEDIUM,
    "slack_send": RiskLevel.MEDIUM,
    "http_request": RiskLevel.MEDIUM,
    "api_call": RiskLevel.MEDIUM,
    # Low — reads, queries
    "fs_read": RiskLevel.LOW,
    "file_read": RiskLevel.LOW,
    "search": RiskLevel.LOW,
    "query": RiskLevel.LOW,
}


class ConsentRequest:
    """Tracks a pending consent request."""

    def __init__(
        self,
        id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        risk_level: RiskLevel,
        reason: str,
    ) -> None:
        self.id = id
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.risk_level = risk_level
        self.reason = reason
        self.status: str = "pending"
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.resolved_at: str | None = None


class ConsentGate:
    """
    Risk classification and user consent for tool calls.

    Classifies tools by risk level and blocks high-risk calls
    until the user approves them.
    """

    def __init__(self, config: ConsentGateConfig, ledger: AuditLedger) -> None:
        self.config = config
        self._ledger = ledger

    def classify_risk(self, tool_name: str) -> RiskLevel:
        """Classify risk level for a tool."""
        # Exact match first
        if tool_name in TOOL_RISK_MAP:
            return TOOL_RISK_MAP[tool_name]

        # Partial match — check if tool name contains any risk keyword
        lower = tool_name.lower()
        for pattern, level in TOOL_RISK_MAP.items():
            if pattern in lower:
                return level

        return RiskLevel.LOW

    def requires_consent(self, tool_name: str) -> bool:
        """Check if a tool call requires consent."""
        # Explicit never-require list
        if tool_name in self.config.never_require:
            return False

        # Explicit always-require list
        if tool_name in self.config.always_require:
            return True

        # Risk threshold check
        risk = self.classify_risk(tool_name)
        return RISK_ORDER[risk] >= RISK_ORDER[self.config.risk_threshold]

    def intercept(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        prompt_fn: Any | None = None,
    ) -> dict[str, Any]:
        """
        Intercept a tool call. If consent is needed, prompt user.

        Returns dict with 'blocked' (bool) and optional 'reason'.
        prompt_fn: optional callable that takes a message and returns bool.
                   If None, uses console input.
        """
        if not self.requires_consent(tool_name):
            return {"blocked": False}

        risk = self.classify_risk(tool_name)

        request = ConsentRequest(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            tool_args=tool_args,
            risk_level=risk,
            reason=f'Tool "{tool_name}" classified as {risk.value} risk',
        )

        # Format and display consent message
        message = self.format_consent_message(request)

        # Get approval
        if prompt_fn is not None:
            approved = prompt_fn(message)
        else:
            approved = self._console_prompt(message)

        # Update request status
        request.status = "approved" if approved else "rejected"
        request.resolved_at = datetime.now(timezone.utc).isoformat()

        # Log to audit ledger
        self._ledger.append(
            action=f"consent_{request.status}",
            tool_name=tool_name,
            risk_level=risk.value,
            consent_required=True,
            consent_granted=approved,
            metadata={
                "consent_id": request.id,
                "tool_args": tool_args,
            },
        )

        if not approved:
            return {
                "blocked": True,
                "reason": "Tool call rejected by user",
            }

        return {"blocked": False}

    def format_consent_message(self, request: ConsentRequest) -> str:
        """Format a human-readable consent message."""
        risk_emoji: dict[RiskLevel, str] = {
            RiskLevel.CRITICAL: "\U0001f6a8",
            RiskLevel.HIGH: "\u26a0\ufe0f",
            RiskLevel.MEDIUM: "\U0001f7e1",
            RiskLevel.LOW: "\U0001f7e2",
            RiskLevel.NONE: "\u2705",
        }

        emoji = risk_emoji.get(request.risk_level, "")
        args_summary = "\n".join(
            f"  {k}: {v!r}" for k, v in request.tool_args.items()
        ) or "  (none)"

        return "\n".join(
            [
                f"{emoji} AIR Trust — Consent Required",
                "",
                f"Tool: {request.tool_name}",
                f"Risk: {request.risk_level.value.upper()}",
                "",
                "Arguments:",
                args_summary,
                "",
                f"Approve? [y/N] (auto-rejects in {self.config.timeout_seconds:.0f}s)",
            ]
        )

    def _console_prompt(self, message: str) -> bool:
        """Prompt user via console. Returns True if approved."""
        try:
            print(message, file=sys.stderr)
            response = input("> ").strip().lower()
            return response in ("y", "yes", "approve")
        except (EOFError, KeyboardInterrupt):
            return False
