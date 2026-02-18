"""
air-crewai-trust — Injection Detector

Monitors inbound messages for prompt injection patterns.
Scores content against a library of known injection techniques
and optionally blocks messages that exceed the threshold.

Hooks into CrewAI's before_llm_call and message processing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .config import InjectionDetectionConfig


@dataclass
class PatternDef:
    """Definition of an injection detection pattern."""

    name: str
    regex: re.Pattern[str]
    weight: float
    min_sensitivity: Literal["low", "medium", "high"]


@dataclass
class InjectionResult:
    """Result of scanning content for injection patterns."""

    detected: bool
    score: float
    patterns: list[str] = field(default_factory=list)
    blocked: bool = False


# Library of prompt injection patterns.
# Weights reflect how suspicious the pattern is (0-1 scale).
INJECTION_PATTERNS: list[PatternDef] = [
    # Direct role override attempts
    PatternDef(
        name="role_override",
        regex=re.compile(
            r"(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts|rules|directives)",
            re.IGNORECASE,
        ),
        weight=0.9,
        min_sensitivity="low",
    ),
    PatternDef(
        name="new_identity",
        regex=re.compile(
            r"(?:you\s+are\s+now|act\s+as|pretend\s+(?:to\s+be|you're)|your\s+new\s+role\s+is)",
            re.IGNORECASE,
        ),
        weight=0.8,
        min_sensitivity="low",
    ),
    PatternDef(
        name="system_prompt_override",
        regex=re.compile(
            r"(?:system\s*prompt|system\s*message|system\s*instruction)\s*[:=]",
            re.IGNORECASE,
        ),
        weight=0.85,
        min_sensitivity="low",
    ),
    # Delimiter-based injection
    PatternDef(
        name="delimiter_injection",
        regex=re.compile(
            r"(?:---+|===+|###)\s*(?:system|admin|developer|root)\s*(?:---+|===+|###)",
            re.IGNORECASE,
        ),
        weight=0.7,
        min_sensitivity="medium",
    ),
    PatternDef(
        name="xml_tag_injection",
        regex=re.compile(
            r"</?(?:system|instruction|admin|prompt|override|command)>",
            re.IGNORECASE,
        ),
        weight=0.6,
        min_sensitivity="medium",
    ),
    # Privilege escalation
    PatternDef(
        name="privilege_escalation",
        regex=re.compile(
            r"(?:admin\s+mode|developer\s+mode|debug\s+mode|god\s+mode|sudo|root\s+access|unrestricted)",
            re.IGNORECASE,
        ),
        weight=0.75,
        min_sensitivity="low",
    ),
    PatternDef(
        name="safety_bypass",
        regex=re.compile(
            r"(?:bypass|disable|turn\s+off|remove)\s+(?:safety|filter|guard|restriction|limit|protection|moderation)",
            re.IGNORECASE,
        ),
        weight=0.85,
        min_sensitivity="low",
    ),
    # Output manipulation
    PatternDef(
        name="output_manipulation",
        regex=re.compile(
            r"(?:do\s+not|don'?t|never)\s+(?:mention|reveal|disclose|show|tell|output)\s+(?:this|that|the|your)",
            re.IGNORECASE,
        ),
        weight=0.5,
        min_sensitivity="medium",
    ),
    PatternDef(
        name="encoding_evasion",
        regex=re.compile(
            r"(?:base64|rot13|hex|encode|decode|obfuscate|translate\s+to\s+(?:morse|binary|ascii))",
            re.IGNORECASE,
        ),
        weight=0.4,
        min_sensitivity="high",
    ),
    # Indirect injection (from external content)
    PatternDef(
        name="hidden_instruction",
        regex=re.compile(
            r"(?:if\s+you\s+(?:are|'re)\s+an?\s+(?:ai|llm|language\s+model|assistant|chatbot))",
            re.IGNORECASE,
        ),
        weight=0.7,
        min_sensitivity="medium",
    ),
    PatternDef(
        name="urgent_override",
        regex=re.compile(
            r"(?:IMPORTANT|URGENT|CRITICAL|EMERGENCY)\s*[:!]\s*(?:ignore|override|new\s+instruction)",
            re.IGNORECASE,
        ),
        weight=0.8,
        min_sensitivity="low",
    ),
    # Tool/function abuse
    PatternDef(
        name="tool_abuse",
        regex=re.compile(
            r"(?:call|execute|run|invoke)\s+(?:the\s+)?(?:function|tool|command|api)\s+(?:with|using)",
            re.IGNORECASE,
        ),
        weight=0.35,
        min_sensitivity="high",
    ),
    PatternDef(
        name="data_exfil",
        regex=re.compile(
            r"(?:send|transmit|forward|email|post)\s+(?:all|the|my|this)\s+(?:data|information|content|conversation|history|context)",
            re.IGNORECASE,
        ),
        weight=0.65,
        min_sensitivity="medium",
    ),
    # Jailbreak patterns
    PatternDef(
        name="dan_jailbreak",
        regex=re.compile(
            r"(?:DAN|do\s+anything\s+now|jailbreak|uncensored\s+mode)",
            re.IGNORECASE,
        ),
        weight=0.9,
        min_sensitivity="low",
    ),
    PatternDef(
        name="hypothetical_bypass",
        regex=re.compile(
            r"(?:hypothetically|in\s+theory|for\s+educational\s+purposes|for\s+research)\s+(?:how\s+would|could\s+you|can\s+you)\s+(?:bypass|hack|break|exploit)",
            re.IGNORECASE,
        ),
        weight=0.6,
        min_sensitivity="medium",
    ),
]

SENSITIVITY_ORDER: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


class InjectionDetector:
    """
    Scans content for prompt injection patterns.

    Filters patterns by sensitivity level and scores content
    against matched patterns. Can optionally block content
    that exceeds the configured threshold.
    """

    def __init__(self, config: InjectionDetectionConfig) -> None:
        self.config = config

        # Filter patterns by sensitivity level
        sensitivity_level = SENSITIVITY_ORDER.get(config.sensitivity, 2)
        self._active_patterns = [
            p
            for p in INJECTION_PATTERNS
            if SENSITIVITY_ORDER.get(p.min_sensitivity, 2) <= sensitivity_level
        ]

    def scan(self, content: str) -> InjectionResult:
        """
        Scan content for injection patterns.
        Returns detection result with score and matched patterns.
        """
        if not content or not content.strip():
            return InjectionResult(detected=False, score=0.0, patterns=[], blocked=False)

        matched_patterns: list[str] = []
        total_weight = 0.0

        for pattern in self._active_patterns:
            if pattern.regex.search(content):
                matched_patterns.append(pattern.name)
                total_weight += pattern.weight

        # Normalize score to 0-1 range (cap at 1.0)
        score = min(total_weight, 1.0)
        detected = len(matched_patterns) > 0
        blocked = (
            self.config.block_threshold > 0 and score >= self.config.block_threshold
        )

        return InjectionResult(
            detected=detected,
            score=score,
            patterns=matched_patterns,
            blocked=blocked,
        )

    def get_active_patterns(self) -> list[str]:
        """Get the list of active pattern names (for debugging/transparency)."""
        return [p.name for p in self._active_patterns]
