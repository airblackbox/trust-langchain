"""Tests for the AuditLedger — tamper-evident HMAC-SHA256 chain."""

import os

from air_langchain_trust.audit_ledger import GENESIS_HASH, AuditLedger
from air_langchain_trust.config import AuditLedgerConfig


class TestAuditLedgerAppend:
    def test_append_creates_entry(self, ledger):
        entry = ledger.append(action="test_action", risk_level="low")
        assert entry.action == "test_action"
        assert entry.risk_level == "low"
        assert entry.sequence == 1

    def test_append_increments_sequence(self, ledger):
        e1 = ledger.append(action="first", risk_level="low")
        e2 = ledger.append(action="second", risk_level="low")
        e3 = ledger.append(action="third", risk_level="low")
        assert e1.sequence == 1
        assert e2.sequence == 2
        assert e3.sequence == 3

    def test_append_sets_prev_hash(self, ledger):
        e1 = ledger.append(action="first", risk_level="low")
        e2 = ledger.append(action="second", risk_level="low")
        assert e1.prev_hash == GENESIS_HASH
        assert e2.prev_hash == e1.hash

    def test_append_with_metadata(self, ledger):
        entry = ledger.append(
            action="tool_call",
            tool_name="exec",
            risk_level="critical",
            consent_required=True,
            consent_granted=True,
            data_tokenized=False,
            injection_detected=False,
            metadata={"session_id": "abc-123"},
        )
        assert entry.tool_name == "exec"
        assert entry.consent_required is True
        assert entry.metadata["session_id"] == "abc-123"


class TestAuditLedgerVerify:
    def test_empty_chain_is_valid(self, ledger):
        result = ledger.verify()
        assert result.valid is True
        assert result.total_entries == 0

    def test_valid_chain_passes(self, ledger):
        for i in range(5):
            ledger.append(action=f"action_{i}", risk_level="low")
        result = ledger.verify()
        assert result.valid is True
        assert result.total_entries == 5

    def test_tampered_entry_breaks_chain(self, ledger):
        for i in range(5):
            ledger.append(action=f"action_{i}", risk_level="low")

        # Tamper with the 3rd entry's action
        ledger._entries[2].action = "tampered!"

        result = ledger.verify()
        assert result.valid is False
        assert result.broken_at_sequence == ledger._entries[2].sequence
        assert "hash mismatch" in result.reason.lower()

    def test_tampered_signature_breaks_chain(self, ledger):
        for i in range(3):
            ledger.append(action=f"action_{i}", risk_level="low")

        # Tamper with signature
        ledger._entries[1].signature = "deadbeef" * 8

        result = ledger.verify()
        assert result.valid is False
        assert result.broken_at_sequence == ledger._entries[1].sequence

    def test_broken_prev_hash_detected(self, ledger):
        for i in range(3):
            ledger.append(action=f"action_{i}", risk_level="low")

        # Break the chain linkage
        ledger._entries[1].prev_hash = "0" * 64

        result = ledger.verify()
        assert result.valid is False
        assert "prev_hash mismatch" in result.reason


class TestAuditLedgerPersistence:
    def test_saves_and_loads(self, tmp_dir):
        config = AuditLedgerConfig(
            local_path=os.path.join(tmp_dir, "ledger.json")
        )
        ledger1 = AuditLedger(config)
        ledger1.append(action="persisted", risk_level="medium")
        ledger1.append(action="also_persisted", risk_level="low")

        # Create a new ledger that loads from the same file
        ledger2 = AuditLedger(config)
        assert len(ledger2._entries) == 2
        assert ledger2._entries[0].action == "persisted"
        assert ledger2._entries[1].action == "also_persisted"

    def test_loaded_chain_still_valid(self, tmp_dir):
        config = AuditLedgerConfig(
            local_path=os.path.join(tmp_dir, "ledger.json")
        )
        ledger1 = AuditLedger(config)
        for i in range(5):
            ledger1.append(action=f"action_{i}", risk_level="low")

        ledger2 = AuditLedger(config)
        result = ledger2.verify()
        assert result.valid is True
        assert result.total_entries == 5


class TestAuditLedgerStats:
    def test_stats_returns_correct_counts(self, ledger):
        ledger.append(action="a", risk_level="low")
        ledger.append(action="b", risk_level="high")

        stats = ledger.stats()
        assert stats["total_entries"] == 2
        assert stats["chain_valid"] is True
        assert "earliest" in stats
        assert "latest" in stats

    def test_export_returns_all_entries(self, ledger):
        for i in range(3):
            ledger.append(action=f"action_{i}", risk_level="low")

        exported = ledger.export()
        assert len(exported) == 3
        assert all(isinstance(e, dict) for e in exported)
        assert exported[0]["action"] == "action_0"

    def test_get_recent(self, ledger):
        for i in range(10):
            ledger.append(action=f"action_{i}", risk_level="low")

        recent = ledger.get_recent(3)
        assert len(recent) == 3
        assert recent[0].action == "action_7"
        assert recent[2].action == "action_9"