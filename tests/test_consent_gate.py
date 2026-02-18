"""Tests for the ConsentGate — risk classification and approval."""


from air_langchain_trust.config import ConsentGateConfig, RiskLevel
from air_langchain_trust.consent_gate import ConsentGate


class TestClassifyRisk:
    def test_exact_match_critical(self, consent_gate):
        assert consent_gate.classify_risk("exec") == RiskLevel.CRITICAL
        assert consent_gate.classify_risk("spawn") == RiskLevel.CRITICAL
        assert consent_gate.classify_risk("shell") == RiskLevel.CRITICAL

    def test_exact_match_high(self, consent_gate):
        assert consent_gate.classify_risk("fs_write") == RiskLevel.HIGH
        assert consent_gate.classify_risk("deploy") == RiskLevel.HIGH
        assert consent_gate.classify_risk("git_push") == RiskLevel.HIGH

    def test_exact_match_medium(self, consent_gate):
        assert consent_gate.classify_risk("send_email") == RiskLevel.MEDIUM
        assert consent_gate.classify_risk("http_request") == RiskLevel.MEDIUM

    def test_exact_match_low(self, consent_gate):
        assert consent_gate.classify_risk("fs_read") == RiskLevel.LOW
        assert consent_gate.classify_risk("search") == RiskLevel.LOW
        assert consent_gate.classify_risk("query") == RiskLevel.LOW

    def test_partial_match(self, consent_gate):
        assert consent_gate.classify_risk("my_exec_tool") == RiskLevel.CRITICAL
        assert consent_gate.classify_risk("custom_deploy_v2") == RiskLevel.HIGH

    def test_unknown_tool_defaults_to_low(self, consent_gate):
        assert consent_gate.classify_risk("unknown_tool") == RiskLevel.LOW
        assert consent_gate.classify_risk("foobar") == RiskLevel.LOW


class TestRequiresConsent:
    def test_always_require_list(self, consent_gate):
        assert consent_gate.requires_consent("exec") is True
        assert consent_gate.requires_consent("deploy") is True

    def test_never_require_list(self, consent_gate):
        assert consent_gate.requires_consent("fs_read") is False
        assert consent_gate.requires_consent("search") is False
        assert consent_gate.requires_consent("query") is False

    def test_risk_threshold(self, consent_gate):
        # Default threshold is HIGH
        assert consent_gate.requires_consent("git_push") is True  # high risk
        assert consent_gate.requires_consent("http_request") is False  # medium risk

    def test_custom_threshold(self, ledger):
        config = ConsentGateConfig(risk_threshold=RiskLevel.MEDIUM)
        gate = ConsentGate(config, ledger)
        assert gate.requires_consent("http_request") is True  # medium >= medium
        assert gate.requires_consent("unknown_tool") is False  # low < medium


class TestIntercept:
    def test_low_risk_not_blocked(self, consent_gate):
        result = consent_gate.intercept("unknown_tool", {"arg": "value"})
        assert result["blocked"] is False

    def test_high_risk_blocked_when_rejected(self, consent_gate):
        result = consent_gate.intercept(
            "exec",
            {"cmd": "rm -rf /"},
            prompt_fn=lambda msg: False,  # Reject
        )
        assert result["blocked"] is True

    def test_high_risk_allowed_when_approved(self, consent_gate):
        result = consent_gate.intercept(
            "exec",
            {"cmd": "echo hello"},
            prompt_fn=lambda msg: True,  # Approve
        )
        assert result["blocked"] is False

    def test_never_require_bypasses_consent(self, consent_gate):
        result = consent_gate.intercept("fs_read", {"path": "/tmp"})
        assert result["blocked"] is False


class TestFormatConsentMessage:
    def test_message_contains_tool_name(self, consent_gate):
        from air_langchain_trust.consent_gate import ConsentRequest

        request = ConsentRequest(
            id="test-id",
            tool_name="exec",
            tool_args={"cmd": "ls"},
            risk_level=RiskLevel.CRITICAL,
            reason="test",
        )
        message = consent_gate.format_consent_message(request)
        assert "exec" in message
        assert "CRITICAL" in message
        assert "cmd" in message