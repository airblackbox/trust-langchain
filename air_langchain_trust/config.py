"""
air-crewai-trust — Configuration

Pydantic models for all trust layer settings.
Sensible defaults match the TypeScript openclaw-air-trust plugin.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}


class ConsentGateConfig(BaseModel):
    enabled: bool = True
    always_require: list[str] = Field(
        default_factory=lambda: ["exec", "spawn", "shell", "deploy"]
    )
    never_require: list[str] = Field(
        default_factory=lambda: ["fs_read", "search", "query"]
    )
    timeout_seconds: float = 30.0
    risk_threshold: RiskLevel = RiskLevel.HIGH


class AuditLedgerConfig(BaseModel):
    enabled: bool = True
    local_path: str = Field(
        default_factory=lambda: str(
            Path.home() / ".air-trust" / "audit-ledger.json"
        )
    )
    forward_to_gateway: bool = False
    max_entries: int = 10_000


class VaultConfig(BaseModel):
    enabled: bool = True
    categories: list[str] = Field(
        default_factory=lambda: ["api_key", "credential", "pii"]
    )
    custom_patterns: list[dict] = Field(default_factory=list)
    forward_to_gateway: bool = False
    ttl_seconds: int = 86_400  # 24 hours


class InjectionDetectionConfig(BaseModel):
    enabled: bool = True
    sensitivity: Literal["low", "medium", "high"] = "medium"
    block_threshold: float = 0.8
    log_detections: bool = True


class AirTrustConfig(BaseModel):
    enabled: bool = True
    consent_gate: ConsentGateConfig = Field(default_factory=ConsentGateConfig)
    audit_ledger: AuditLedgerConfig = Field(default_factory=AuditLedgerConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
    injection_detection: InjectionDetectionConfig = Field(
        default_factory=InjectionDetectionConfig
    )
    gateway_url: str | None = None
    gateway_key: str | None = None