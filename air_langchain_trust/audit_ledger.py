"""
air-crewai-trust — Audit Ledger

Tamper-evident action log using HMAC-SHA256 chaining.
Each entry includes the hash of the previous entry, creating
a blockchain-style chain. Modifying any entry breaks the chain.

Supports local JSON persistence and optional forwarding to
the AIR Blackbox gateway.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AuditLedgerConfig

GENESIS_HASH = "0" * 64


class AuditEntry:
    """A single signed entry in the audit chain."""

    __slots__ = (
        "id",
        "sequence",
        "hash",
        "prev_hash",
        "signature",
        "timestamp",
        "action",
        "tool_name",
        "risk_level",
        "consent_required",
        "consent_granted",
        "data_tokenized",
        "injection_detected",
        "metadata",
    )

    def __init__(
        self,
        *,
        id: str,
        sequence: int,
        hash: str,
        prev_hash: str,
        signature: str,
        timestamp: str,
        action: str,
        tool_name: str | None = None,
        risk_level: str = "none",
        consent_required: bool = False,
        consent_granted: bool | None = None,
        data_tokenized: bool = False,
        injection_detected: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.sequence = sequence
        self.hash = hash
        self.prev_hash = prev_hash
        self.signature = signature
        self.timestamp = timestamp
        self.action = action
        self.tool_name = tool_name
        self.risk_level = risk_level
        self.consent_required = consent_required
        self.consent_granted = consent_granted
        self.data_tokenized = data_tokenized
        self.injection_detected = injection_detected
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "hash": self.hash,
            "prev_hash": self.prev_hash,
            "signature": self.signature,
            "timestamp": self.timestamp,
            "action": self.action,
            "tool_name": self.tool_name,
            "risk_level": self.risk_level,
            "consent_required": self.consent_required,
            "consent_granted": self.consent_granted,
            "data_tokenized": self.data_tokenized,
            "injection_detected": self.injection_detected,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        return cls(**data)


class ChainVerification:
    """Result of verifying chain integrity."""

    def __init__(
        self,
        valid: bool,
        total_entries: int,
        broken_at_sequence: int | None = None,
        broken_at_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.valid = valid
        self.total_entries = total_entries
        self.broken_at_sequence = broken_at_sequence
        self.broken_at_id = broken_at_id
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "valid": self.valid,
            "total_entries": self.total_entries,
        }
        if self.broken_at_sequence is not None:
            d["broken_at_sequence"] = self.broken_at_sequence
            d["broken_at_id"] = self.broken_at_id
            d["reason"] = self.reason
        return d


class AuditLedger:
    """
    Tamper-evident audit chain using HMAC-SHA256.

    Each entry's signature incorporates the previous entry's hash,
    creating an unbreakable chain. Any modification to any entry
    invalidates all subsequent signatures.
    """

    def __init__(
        self,
        config: AuditLedgerConfig,
        gateway_url: str | None = None,
        gateway_key: str | None = None,
    ) -> None:
        self.config = config
        self._gateway_url = gateway_url
        self._gateway_key = gateway_key
        self._entries: list[AuditEntry] = []
        self._last_hash: str = GENESIS_HASH
        self._sequence: int = 0

        # Load or generate HMAC key
        key_path = config.local_path.replace(".json", "") + ".key"
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                self._secret = bytes.fromhex(f.read().strip())
        else:
            self._secret = os.urandom(32)
            Path(key_path).parent.mkdir(parents=True, exist_ok=True)
            with open(key_path, "w") as f:
                f.write(self._secret.hex())
            os.chmod(key_path, 0o600)

        # Load existing chain
        self._load_chain()

    def append(
        self,
        *,
        action: str,
        tool_name: str | None = None,
        risk_level: str = "none",
        consent_required: bool = False,
        consent_granted: bool | None = None,
        data_tokenized: bool = False,
        injection_detected: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Append an action to the audit chain. Returns the signed entry."""
        self._sequence += 1

        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Compute content hash
        content_for_hash = json.dumps(
            {
                "id": entry_id,
                "sequence": self._sequence,
                "timestamp": timestamp,
                "action": action,
                "tool_name": tool_name,
                "risk_level": risk_level,
                "consent_required": consent_required,
                "consent_granted": consent_granted,
                "data_tokenized": data_tokenized,
                "injection_detected": injection_detected,
                "metadata": metadata or {},
            },
            sort_keys=True,
        )
        record_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()

        # HMAC signature chains this entry to the previous one
        sig_payload = f"{self._sequence}|{entry_id}|{record_hash}|{self._last_hash}"
        signature = hmac.new(
            self._secret, sig_payload.encode(), hashlib.sha256
        ).hexdigest()

        entry = AuditEntry(
            id=entry_id,
            sequence=self._sequence,
            hash=record_hash,
            prev_hash=self._last_hash,
            signature=signature,
            timestamp=timestamp,
            action=action,
            tool_name=tool_name,
            risk_level=risk_level,
            consent_required=consent_required,
            consent_granted=consent_granted,
            data_tokenized=data_tokenized,
            injection_detected=injection_detected,
            metadata=metadata or {},
        )

        self._last_hash = record_hash
        self._entries.append(entry)

        # Trim if over max
        if self.config.max_entries > 0 and len(self._entries) > self.config.max_entries:
            self._entries = self._entries[-self.config.max_entries :]

        # Persist locally
        self._save_chain()

        # Non-blocking forward to gateway
        if self.config.forward_to_gateway and self._gateway_url:
            self._forward_entry(entry)

        return entry

    def verify(self) -> ChainVerification:
        """Verify the integrity of the entire chain."""
        if not self._entries:
            return ChainVerification(valid=True, total_entries=0)

        expected_prev_hash = GENESIS_HASH

        for entry in self._entries:
            # Check prevHash linkage
            if entry.prev_hash != expected_prev_hash:
                return ChainVerification(
                    valid=False,
                    total_entries=len(self._entries),
                    broken_at_sequence=entry.sequence,
                    broken_at_id=entry.id,
                    reason=f"prev_hash mismatch at sequence {entry.sequence}",
                )

            # Recompute content hash
            content_for_hash = json.dumps(
                {
                    "id": entry.id,
                    "sequence": entry.sequence,
                    "timestamp": entry.timestamp,
                    "action": entry.action,
                    "tool_name": entry.tool_name,
                    "risk_level": entry.risk_level,
                    "consent_required": entry.consent_required,
                    "consent_granted": entry.consent_granted,
                    "data_tokenized": entry.data_tokenized,
                    "injection_detected": entry.injection_detected,
                    "metadata": entry.metadata,
                },
                sort_keys=True,
            )
            computed_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()

            if entry.hash != computed_hash:
                return ChainVerification(
                    valid=False,
                    total_entries=len(self._entries),
                    broken_at_sequence=entry.sequence,
                    broken_at_id=entry.id,
                    reason=f"Content hash mismatch at sequence {entry.sequence}",
                )

            # Verify HMAC signature
            sig_payload = (
                f"{entry.sequence}|{entry.id}|{entry.hash}|{entry.prev_hash}"
            )
            expected_sig = hmac.new(
                self._secret, sig_payload.encode(), hashlib.sha256
            ).hexdigest()

            if entry.signature != expected_sig:
                return ChainVerification(
                    valid=False,
                    total_entries=len(self._entries),
                    broken_at_sequence=entry.sequence,
                    broken_at_id=entry.id,
                    reason=f"Signature mismatch at sequence {entry.sequence}",
                )

            expected_prev_hash = entry.hash

        return ChainVerification(valid=True, total_entries=len(self._entries))

    def get_recent(self, n: int = 50) -> list[AuditEntry]:
        """Get the N most recent entries."""
        return self._entries[-n:]

    def export(self) -> list[dict[str, Any]]:
        """Export all entries as dicts."""
        return [e.to_dict() for e in self._entries]

    def stats(self) -> dict[str, Any]:
        """Chain statistics."""
        verification = self.verify()
        result: dict[str, Any] = {
            "total_entries": len(self._entries),
            "chain_valid": verification.valid,
        }
        if self._entries:
            result["earliest"] = self._entries[0].timestamp
            result["latest"] = self._entries[-1].timestamp
        return result

    # --- Private ---

    def _load_chain(self) -> None:
        if os.path.exists(self.config.local_path):
            try:
                with open(self.config.local_path, "r") as f:
                    data = json.load(f)
                self._entries = [
                    AuditEntry.from_dict(e) for e in data.get("entries", [])
                ]
                self._sequence = data.get("sequence", 0)
                self._last_hash = data.get("last_hash", GENESIS_HASH)
            except (json.JSONDecodeError, KeyError):
                self._entries = []
                self._sequence = 0
                self._last_hash = GENESIS_HASH

    def _save_chain(self) -> None:
        Path(self.config.local_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [e.to_dict() for e in self._entries],
            "sequence": self._sequence,
            "last_hash": self._last_hash,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.config.local_path, "w") as f:
            json.dump(data, f, indent=2)

    def _forward_entry(self, entry: AuditEntry) -> None:
        """Best-effort forward to gateway (non-blocking)."""
        if not self._gateway_url:
            return
        try:
            import urllib.request

            url = f"{self._gateway_url}/v1/audit"
            req = urllib.request.Request(
                url,
                data=json.dumps(entry.to_dict()).encode(),
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
            pass  # Silent fail — gateway forwarding is best-effort
