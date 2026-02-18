"""
air-crewai-trust — Data Vault

Detects sensitive data (API keys, PII, credentials) in tool
arguments and LLM context, replaces them with opaque tokens.
Original values are stored locally and optionally forwarded
to the AIR vault for centralized management.

Token format: [AIR:vault:category:tokenId]
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import VaultConfig


class TokenizationPattern:
    """A regex pattern for detecting sensitive data."""

    def __init__(self, name: str, category: str, regex: re.Pattern[str]) -> None:
        self.name = name
        self.category = category
        self.regex = regex


class VaultToken:
    """A stored token mapping."""

    def __init__(
        self,
        token_id: str,
        category: str,
        created_at: str,
        expires_at: str,
        original_value: str,
    ) -> None:
        self.token_id = token_id
        self.category = category
        self.created_at = created_at
        self.expires_at = expires_at
        self.original_value = original_value


# Built-in patterns for common sensitive data
BUILTIN_PATTERNS: list[TokenizationPattern] = [
    TokenizationPattern(
        "OpenAI API Key", "api_key", re.compile(r"sk-[A-Za-z0-9]{20,}")
    ),
    TokenizationPattern(
        "Anthropic API Key", "api_key", re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}")
    ),
    TokenizationPattern(
        "AWS Access Key", "api_key", re.compile(r"AKIA[0-9A-Z]{16}")
    ),
    TokenizationPattern(
        "GitHub Token", "api_key", re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}")
    ),
    TokenizationPattern(
        "Stripe Key", "api_key", re.compile(r"sk_(?:live|test)_[A-Za-z0-9]{24,}")
    ),
    TokenizationPattern(
        "Bearer Token", "credential", re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*")
    ),
    TokenizationPattern(
        "Private Key Block",
        "credential",
        re.compile(
            r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END (?:RSA |EC )?PRIVATE KEY-----"
        ),
    ),
    TokenizationPattern(
        "Connection String",
        "credential",
        re.compile(r"(?:mongodb|postgres|mysql|redis)://[^\s\"']+"),
    ),
    TokenizationPattern(
        "Email Address",
        "pii",
        re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    ),
    TokenizationPattern(
        "Phone Number",
        "pii",
        re.compile(r"(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    ),
    TokenizationPattern(
        "SSN", "pii", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    ),
    TokenizationPattern(
        "Credit Card",
        "pii",
        re.compile(
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"
        ),
    ),
    TokenizationPattern(
        "Generic Secret Assignment",
        "credential",
        re.compile(
            r"(?:password|secret|token|api_key|apikey)\s*[:=]\s*[\"']?[A-Za-z0-9\-._~+/]{8,}[\"']?",
            re.IGNORECASE,
        ),
    ),
]


class DataVault:
    """
    Detects and tokenizes sensitive data.

    Scans text for API keys, credentials, and PII using regex patterns,
    replaces matches with opaque tokens, and stores original values
    for later detokenization.
    """

    def __init__(
        self,
        config: VaultConfig,
        gateway_url: str | None = None,
        gateway_key: str | None = None,
    ) -> None:
        self.config = config
        self._gateway_url = gateway_url
        self._gateway_key = gateway_key
        self._tokens: dict[str, VaultToken] = {}

        # Filter built-in patterns by configured categories
        self._patterns: list[TokenizationPattern] = []
        for p in BUILTIN_PATTERNS:
            if not config.categories or p.category in config.categories:
                self._patterns.append(p)

        # Add custom patterns
        for cp in config.custom_patterns:
            self._patterns.append(
                TokenizationPattern(
                    name=cp.get("name", "custom"),
                    category=cp.get("category", "custom"),
                    regex=re.compile(cp["regex"]),
                )
            )

    def tokenize(self, text: str) -> dict[str, Any]:
        """
        Scan text for sensitive data and replace with vault tokens.
        Returns dict with 'result', 'tokenized' (bool), and 'count'.
        """
        result = text
        count = 0

        for pattern in self._patterns:
            def replacer(match: re.Match[str], pat: TokenizationPattern = pattern) -> str:
                nonlocal count
                token_id = uuid.uuid4().hex[:8]
                full_token = f"[AIR:vault:{pat.category}:{token_id}]"

                now = datetime.now(timezone.utc)
                vault_token = VaultToken(
                    token_id=token_id,
                    category=pat.category,
                    created_at=now.isoformat(),
                    expires_at=(
                        now + timedelta(seconds=self.config.ttl_seconds)
                    ).isoformat(),
                    original_value=match.group(0),
                )
                self._tokens[token_id] = vault_token

                # Non-blocking forward to gateway
                if self.config.forward_to_gateway and self._gateway_url:
                    self._forward_token(vault_token)

                count += 1
                return full_token

            result = pattern.regex.sub(replacer, result)

        return {"result": result, "tokenized": count > 0, "count": count}

    def detokenize(self, text: str) -> str:
        """Replace vault tokens back with original values."""

        def replacer(match: re.Match[str]) -> str:
            token_id = match.group(2)
            token = self._tokens.get(token_id)
            if token:
                return token.original_value
            return match.group(0)  # Leave as-is if not found

        return re.sub(
            r"\[AIR:vault:([^:]+):([^\]]+)\]", replacer, text
        )

    def stats(self) -> dict[str, Any]:
        """Vault statistics."""
        by_category: dict[str, int] = {}
        for token in self._tokens.values():
            by_category[token.category] = by_category.get(token.category, 0) + 1
        return {"total_tokens": len(self._tokens), "by_category": by_category}

    def cleanup(self) -> int:
        """Remove expired tokens. Returns count of removed tokens."""
        now = datetime.now(timezone.utc)
        expired = [
            tid
            for tid, token in self._tokens.items()
            if datetime.fromisoformat(token.expires_at) < now
        ]
        for tid in expired:
            del self._tokens[tid]
        return len(expired)

    def _forward_token(self, token: VaultToken) -> None:
        """Best-effort forward to gateway."""
        if not self._gateway_url:
            return
        try:
            import urllib.request

            url = f"{self._gateway_url}/v1/vault/store"
            data = json.dumps(
                {
                    "token_id": token.token_id,
                    "category": token.category,
                    "created_at": token.created_at,
                    "expires_at": token.expires_at,
                }
            ).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    **(
                        {"Authorization": f"Bearer {self._gateway_key}"}
                        if self._gateway_key
                        else {}
                    ),
                },
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
