"""Shared test fixtures for air-crewai-trust."""

import os
import tempfile

import pytest

from air_langchain_trust.audit_ledger import AuditLedger
from air_langchain_trust.config import (
    AuditLedgerConfig,
    ConsentGateConfig,
    InjectionDetectionConfig,
    VaultConfig,
)
from air_langchain_trust.consent_gate import ConsentGate
from air_langchain_trust.data_vault import DataVault
from air_langchain_trust.injection_detector import InjectionDetector


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def ledger_config(tmp_dir):
    """Audit ledger config pointing to temp directory."""
    return AuditLedgerConfig(
        enabled=True,
        local_path=os.path.join(tmp_dir, "audit-ledger.json"),
        forward_to_gateway=False,
        max_entries=10_000,
    )


@pytest.fixture
def ledger(ledger_config):
    """Fresh AuditLedger instance."""
    return AuditLedger(ledger_config)


@pytest.fixture
def vault_config():
    """Default VaultConfig."""
    return VaultConfig()


@pytest.fixture
def vault(vault_config):
    """Fresh DataVault instance."""
    return DataVault(vault_config)


@pytest.fixture
def consent_config():
    """Default ConsentGateConfig."""
    return ConsentGateConfig()


@pytest.fixture
def consent_gate(consent_config, ledger):
    """Fresh ConsentGate instance."""
    return ConsentGate(consent_config, ledger)


@pytest.fixture
def injection_config():
    """Default InjectionDetectionConfig."""
    return InjectionDetectionConfig()


@pytest.fixture
def detector(injection_config):
    """Fresh InjectionDetector instance."""
    return InjectionDetector(injection_config)
