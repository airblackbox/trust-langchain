"""
air-langchain-trust — LangChain Callback Handler

Wires the trust layer components into LangChain's callback system:
  - on_tool_start  → ConsentGate + DataVault + AuditLedger
  - on_tool_end    → AuditLedger
  - on_tool_error  → AuditLedger
  - on_llm_start   → InjectionDetector + DataVault + AuditLedger
  - on_llm_end     → AuditLedger
  - on_chain_start → AuditLedger
  - on_chain_end   → AuditLedger

Unlike CrewAI hooks (which return False to block), LangChain callbacks
are observation-only. We raise AirTrustError subclasses to halt execution.

Usage:
    from air_langchain_trust import AirTrustCallbackHandler

    handler = AirTrustCallbackHandler()
    result = chain.invoke(input, config={"callbacks": [handler]})
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from .config import AirTrustConfig
from .audit_ledger import AuditLedger
from .consent_gate import ConsentGate
from .data_vault import DataVault
from .errors import ConsentDeniedError, InjectionBlockedError
from .injection_detector import InjectionDetector

logger = logging.getLogger("air_langchain_trust")

# We avoid importing langchain_core at module level so the package
# can be installed and tested without langchain_core present.
# The import happens at class definition time via a factory.

try:
    from langchain_core.callbacks.base import BaseCallbackHandler

    _HAS_LANGCHAIN = True
except ImportError:
    # Provide a fallback base class for environments without langchain
    BaseCallbackHandler = object  # type: ignore[misc,assignment]
    _HAS_LANGCHAIN = False


class AirTrustCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """
    LangChain callback handler that integrates all AIR trust components.

    Monitors tool calls, LLM invocations, and chain execution.
    Raises exceptions to block operations that fail trust checks:
      - ConsentDeniedError: when user denies consent for a risky tool
      - InjectionBlockedError: when prompt injection is detected

    Args:
        config: Optional trust configuration. Uses sensible defaults if omitted.
    """

    def __init__(self, config: AirTrustConfig | None = None) -> None:
        if _HAS_LANGCHAIN:
            super().__init__()

        self.config = config or AirTrustConfig()

        # Initialize trust components
        self.ledger = AuditLedger(
            self.config.audit_ledger,
            self.config.gateway_url,
            self.config.gateway_key,
        )
        self.consent_gate = ConsentGate(self.config.consent_gate, self.ledger)
        self.vault = DataVault(
            self.config.vault,
            self.config.gateway_url,
            self.config.gateway_key,
        )
        self.injection_detector = InjectionDetector(self.config.injection_detection)

    # ─── Tool Callbacks ───────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a tool starts. May raise ConsentDeniedError."""
        if not self.config.enabled:
            return

        tool_name = serialized.get("name", "unknown")

        # 1. Tokenize sensitive data in tool input
        data_tokenized = False
        if self.config.vault.enabled and input_str:
            vault_result = self.vault.tokenize(input_str)
            if vault_result["tokenized"]:
                data_tokenized = True

        # 2. Check consent gate — raises ConsentDeniedError if blocked
        if self.config.consent_gate.enabled:
            consent_result = self.consent_gate.intercept(
                tool_name,
                inputs or {"input": input_str},
            )
            if consent_result.get("blocked"):
                risk = self.consent_gate.classify_risk(tool_name)
                raise ConsentDeniedError(
                    tool_name=tool_name,
                    risk_level=risk.value,
                )

        # 3. Log to audit ledger
        if self.config.audit_ledger.enabled:
            self.ledger.append(
                action="tool_call",
                tool_name=tool_name,
                risk_level=self.consent_gate.classify_risk(tool_name).value,
                consent_required=self.consent_gate.requires_consent(tool_name),
                consent_granted=True,
                data_tokenized=data_tokenized,
                injection_detected=False,
                metadata={
                    "run_id": str(run_id),
                    **({"parent_run_id": str(parent_run_id)} if parent_run_id else {}),
                },
            )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a tool finishes successfully."""
        if not self.config.enabled or not self.config.audit_ledger.enabled:
            return

        self.ledger.append(
            action="tool_result",
            risk_level="none",
            consent_required=False,
            data_tokenized=False,
            injection_detected=False,
            metadata={
                "run_id": str(run_id),
                "output_length": len(str(output)) if output else 0,
            },
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a tool raises an error."""
        if not self.config.enabled or not self.config.audit_ledger.enabled:
            return

        self.ledger.append(
            action="tool_error",
            risk_level="high",
            consent_required=False,
            data_tokenized=False,
            injection_detected=False,
            metadata={
                "run_id": str(run_id),
                "error_type": type(error).__name__,
                "error_message": str(error)[:200],
            },
        )

    # ─── LLM Callbacks ────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM starts. May raise InjectionBlockedError."""
        if not self.config.enabled:
            return

        full_content = "\n".join(prompts)
        data_tokenized = False
        injection_detected = False

        # 1. Tokenize sensitive data in prompts
        if self.config.vault.enabled and full_content.strip():
            vault_result = self.vault.tokenize(full_content)
            if vault_result["tokenized"]:
                data_tokenized = True

        # 2. Check for prompt injection
        if self.config.injection_detection.enabled and full_content.strip():
            scan_result = self.injection_detector.scan(full_content)

            if scan_result.detected:
                injection_detected = True

                # Log the detection
                if (
                    self.config.injection_detection.log_detections
                    and self.config.audit_ledger.enabled
                ):
                    risk = (
                        "critical"
                        if scan_result.score >= 0.8
                        else ("high" if scan_result.score >= 0.5 else "medium")
                    )
                    self.ledger.append(
                        action="injection_detected",
                        risk_level=risk,
                        consent_required=False,
                        data_tokenized=data_tokenized,
                        injection_detected=True,
                        metadata={
                            "run_id": str(run_id),
                            "score": scan_result.score,
                            "patterns": scan_result.patterns,
                            "blocked": scan_result.blocked,
                            "source": "llm_input",
                        },
                    )

                # Block if threshold exceeded
                if scan_result.blocked:
                    raise InjectionBlockedError(
                        score=scan_result.score,
                        patterns=scan_result.patterns,
                    )

        # 3. Log the LLM call
        if self.config.audit_ledger.enabled:
            model_name = serialized.get("kwargs", {}).get("model_name", "unknown")
            if model_name == "unknown":
                model_name = serialized.get("id", ["unknown"])[-1]

            self.ledger.append(
                action="llm_call",
                risk_level="none",
                consent_required=False,
                data_tokenized=data_tokenized,
                injection_detected=injection_detected,
                metadata={
                    "run_id": str(run_id),
                    "model": str(model_name),
                    "prompt_count": len(prompts),
                    "total_chars": len(full_content),
                },
            )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM responds."""
        if not self.config.enabled or not self.config.audit_ledger.enabled:
            return

        # Extract content length from LLMResult
        content_length = 0
        if hasattr(response, "generations"):
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "text"):
                        content_length += len(gen.text)

        self.ledger.append(
            action="llm_output",
            risk_level="none",
            consent_required=False,
            data_tokenized=False,
            injection_detected=False,
            metadata={
                "run_id": str(run_id),
                "content_length": content_length,
            },
        )

    # ─── Chain Callbacks ──────────────────────────────────────

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a chain starts execution."""
        if not self.config.enabled or not self.config.audit_ledger.enabled:
            return

        chain_name = serialized.get("id", ["unknown"])[-1]

        self.ledger.append(
            action="chain_start",
            risk_level="none",
            consent_required=False,
            data_tokenized=False,
            injection_detected=False,
            metadata={
                "run_id": str(run_id),
                "chain_name": chain_name,
                **({"parent_run_id": str(parent_run_id)} if parent_run_id else {}),
            },
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when a chain finishes execution."""
        if not self.config.enabled or not self.config.audit_ledger.enabled:
            return

        self.ledger.append(
            action="chain_end",
            risk_level="none",
            consent_required=False,
            data_tokenized=False,
            injection_detected=False,
            metadata={
                "run_id": str(run_id),
            },
        )

    # ─── Public API ───────────────────────────────────────────

    def get_audit_stats(self) -> dict:
        """Get audit chain statistics."""
        return self.ledger.stats()

    def verify_chain(self) -> dict:
        """Verify audit chain integrity."""
        return self.ledger.verify().to_dict()

    def export_audit(self) -> list[dict]:
        """Export all audit entries."""
        return self.ledger.export()

    def get_vault_stats(self) -> dict:
        """Get data vault statistics."""
        return self.vault.stats()
