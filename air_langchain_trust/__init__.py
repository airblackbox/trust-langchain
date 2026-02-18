"""
air-langchain-trust — AIR Trust Layer for LangChain / LangGraph

Drop-in security, audit, and compliance layer for LangChain agents.
Adds tamper-proof audit trails, sensitive data tokenization,
consent gates for destructive tools, and prompt injection detection.

Usage:
    from langchain_openai import ChatOpenAI
    from air_langchain_trust import AirTrustCallbackHandler

    handler = AirTrustCallbackHandler()
    llm = ChatOpenAI(model="gpt-4")
    result = llm.invoke("Hello", config={"callbacks": [handler]})

    # Check what happened
    print(handler.get_audit_stats())
    print(handler.verify_chain())
"""

from __future__ import annotations

from .config import (
    AirTrustConfig,
    AuditLedgerConfig,
    ConsentGateConfig,
    InjectionDetectionConfig,
    RiskLevel,
    RISK_ORDER,
    VaultConfig,
)
from .errors import AirTrustError, ConsentDeniedError, InjectionBlockedError
from .handler import AirTrustCallbackHandler

__version__ = "0.1.0"
__all__ = [
    "AirTrustCallbackHandler",
    "AirTrustConfig",
    "AirTrustError",
    "AuditLedgerConfig",
    "ConsentDeniedError",
    "ConsentGateConfig",
    "InjectionBlockedError",
    "InjectionDetectionConfig",
    "RiskLevel",
    "RISK_ORDER",
    "VaultConfig",
]
