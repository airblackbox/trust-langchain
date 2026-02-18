"""Tests for the DataVault — sensitive data tokenization."""

from datetime import datetime, timedelta, timezone

from air_langchain_trust.config import VaultConfig
from air_langchain_trust.data_vault import DataVault


class TestTokenize:
    def test_tokenize_openai_key(self, vault):
        text = "Use this: sk-abc123def456ghi789jkl012mno"
        result = vault.tokenize(text)
        assert result["tokenized"] is True
        assert result["count"] >= 1
        assert "sk-abc123" not in result["result"]
        assert "[AIR:vault:" in result["result"]

    def test_tokenize_aws_key(self, vault):
        text = "AWS key: AKIAIOSFODNN7EXAMPLE"
        result = vault.tokenize(text)
        assert result["tokenized"] is True
        assert "AKIAIOSFODNN7EXAMPLE" not in result["result"]

    def test_tokenize_email(self, vault):
        text = "Contact me at user@example.com please"
        result = vault.tokenize(text)
        assert result["tokenized"] is True
        assert "user@example.com" not in result["result"]
        assert "[AIR:vault:pii:" in result["result"]

    def test_tokenize_ssn(self, vault):
        text = "SSN: 123-45-6789"
        result = vault.tokenize(text)
        assert result["tokenized"] is True
        assert "123-45-6789" not in result["result"]

    def test_tokenize_connection_string(self, vault):
        text = "DB: postgres://user:pass@localhost:5432/mydb"
        result = vault.tokenize(text)
        assert result["tokenized"] is True
        assert "postgres://" not in result["result"]

    def test_no_false_positives_on_benign_text(self, vault):
        text = "The quick brown fox jumps over the lazy dog"
        result = vault.tokenize(text)
        assert result["tokenized"] is False
        assert result["count"] == 0
        assert result["result"] == text

    def test_multiple_sensitive_values(self, vault):
        text = "Key: sk-abc123def456ghi789jkl012mno, Email: test@example.com"
        result = vault.tokenize(text)
        assert result["tokenized"] is True
        assert result["count"] >= 2


class TestDetokenize:
    def test_roundtrip_email(self, vault):
        original = "Email: user@example.com"
        tokenized = vault.tokenize(original)
        restored = vault.detokenize(tokenized["result"])
        assert "user@example.com" in restored

    def test_unknown_token_left_as_is(self, vault):
        text = "Token: [AIR:vault:api_key:unknown123]"
        result = vault.detokenize(text)
        assert result == text  # Unchanged


class TestVaultStats:
    def test_stats_empty(self, vault):
        stats = vault.stats()
        assert stats["total_tokens"] == 0
        assert stats["by_category"] == {}

    def test_stats_after_tokenization(self, vault):
        vault.tokenize("Email: user@example.com, SSN: 123-45-6789")
        stats = vault.stats()
        assert stats["total_tokens"] >= 2
        assert "pii" in stats["by_category"]


class TestVaultCleanup:
    def test_cleanup_expired(self, vault):
        # Tokenize something
        vault.tokenize("test@example.com")
        assert vault.stats()["total_tokens"] >= 1

        # Manually expire all tokens
        for token in vault._tokens.values():
            token.expires_at = (
                datetime.now(timezone.utc) - timedelta(hours=1)
            ).isoformat()

        removed = vault.cleanup()
        assert removed >= 1
        assert vault.stats()["total_tokens"] == 0

    def test_cleanup_keeps_valid(self, vault):
        vault.tokenize("test@example.com")
        removed = vault.cleanup()
        assert removed == 0
        assert vault.stats()["total_tokens"] >= 1


class TestCustomPatterns:
    def test_custom_pattern(self):
        config = VaultConfig(
            custom_patterns=[
                {
                    "name": "Custom Token",
                    "category": "custom",
                    "regex": r"CUSTOM-[A-Z]{4}-\d{4}",
                }
            ]
        )
        vault = DataVault(config)
        result = vault.tokenize("My token: CUSTOM-ABCD-1234")
        assert result["tokenized"] is True
        assert "CUSTOM-ABCD-1234" not in result["result"]