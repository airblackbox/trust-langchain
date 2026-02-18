"""
air-langchain-trust — Custom Exceptions

LangChain callbacks are observation-only — they can't return False
to block operations like CrewAI hooks can. Instead, we raise custom
exceptions to halt execution when trust checks fail.

These exceptions inherit from a common AirTrustError base so
users can catch them cleanly in try/except blocks.
"""

from __future__ import annotations


class AirTrustError(Exception):
    """Base exception for all AIR Trust Layer errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ConsentDeniedError(AirTrustError):
    """
    Raised when a tool call is blocked because the user denied consent.

    Attributes:
        tool_name: The tool that was blocked.
        risk_level: The risk classification that triggered the consent check.
    """

    def __init__(
        self,
        tool_name: str,
        risk_level: str,
        message: str | None = None,
    ) -> None:
        msg = message or f"Consent denied for tool '{tool_name}' (risk: {risk_level})"
        super().__init__(msg, {"tool_name": tool_name, "risk_level": risk_level})
        self.tool_name = tool_name
        self.risk_level = risk_level


class InjectionBlockedError(AirTrustError):
    """
    Raised when a prompt injection is detected and the score
    exceeds the configured block threshold.

    Attributes:
        score: The injection detection score (0-1).
        patterns: List of pattern names that matched.
    """

    def __init__(
        self,
        score: float,
        patterns: list[str],
        message: str | None = None,
    ) -> None:
        msg = message or (
            f"Prompt injection blocked (score: {score:.2f}, "
            f"patterns: {', '.join(patterns)})"
        )
        super().__init__(msg, {"score": score, "patterns": patterns})
        self.score = score
        self.patterns = patterns
