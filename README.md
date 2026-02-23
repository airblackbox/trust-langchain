# air-langchain-trust

**AIR Trust Layer for LangChain / LangGraph** — Drop-in security, audit, and compliance for your AI agents.

Part of the [AIR Blackbox](https://airblackbox.com) ecosystem. Adds tamper-proof audit trails, sensitive data tokenization, consent gates for destructive tools, and prompt injection detection to any LangChain or LangGraph project.

## Quick Start

```bash
pip install air-langchain-trust
```

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from air_langchain_trust import AirTrustCallbackHandler

# Create the trust handler
handler = AirTrustCallbackHandler()

# Use it with any LangChain component via config
llm = ChatOpenAI(model="gpt-4")
result = llm.invoke(
    "What is the capital of France?",
    config={"callbacks": [handler]}
)

# Or with agents — callbacks propagate to all child components
result = agent_executor.invoke(
    {"input": "Search for AI safety papers"},
    config={"callbacks": [handler]}
)

# Check what happened
print(handler.get_audit_stats())
print(handler.verify_chain())
```

## What It Does

### Tamper-Proof Audit Trail
Every tool call, LLM invocation, and chain execution is logged to an HMAC-SHA256 signed chain. Each entry references the previous entry's hash — modify any record and the chain breaks.

### Sensitive Data Tokenization
API keys, credentials, PII (emails, SSNs, phone numbers, credit cards) are automatically detected in tool inputs and LLM prompts, logged as tokenized versions. **14 built-in patterns** covering API keys, credentials, and PII.

### Consent Gate
Destructive tools are blocked until the user explicitly approves them. Unlike CrewAI (where hooks return False), LangChain callbacks block by **raising exceptions**:

```python
from air_langchain_trust import ConsentDeniedError, InjectionBlockedError

try:
    result = agent.invoke(input, config={"callbacks": [handler]})
except ConsentDeniedError as e:
    print(f"Tool '{e.tool_name}' blocked (risk: {e.risk_level})")
except InjectionBlockedError as e:
    print(f"Injection detected (score: {e.score}, patterns: {e.patterns})")
```

| Risk Level | Tools | Action |
|-----------|-------|--------|
| **Critical** | exec, spawn, shell | Always requires consent |
| **High** | fs_write, deploy, git_push | Requires consent (default) |
| **Medium** | send_email, http_request | Configurable |
| **Low** | fs_read, search, query | Auto-approved |

### Prompt Injection Detection
15+ weighted patterns detect prompt injection attempts including role overrides, jailbreaks, delimiter injection, privilege escalation, and data exfiltration.

## Configuration

```python
from air_langchain_trust import AirTrustCallbackHandler, AirTrustConfig

config = AirTrustConfig(
    consent_gate={
        "enabled": True,
        "always_require": ["exec", "spawn", "shell", "deploy"],
        "risk_threshold": "high",
    },
    vault={
        "enabled": True,
        "categories": ["api_key", "credential", "pii"],
    },
    injection_detection={
        "enabled": True,
        "sensitivity": "medium",
        "block_threshold": 0.8,
    },
    audit_ledger={
        "enabled": True,
        "max_entries": 10000,
    },
    # Optional: forward to AIR Blackbox gateway
    gateway_url="https://your-gateway.example.com",
    gateway_key="your-api-key",
)

handler = AirTrustCallbackHandler(config)
```

## LangChain Callback Mapping

| LangChain Callback | Trust Components |
|-------------------|-----------------|
| `on_tool_start` | ConsentGate → DataVault → AuditLedger |
| `on_tool_end` | AuditLedger |
| `on_tool_error` | AuditLedger |
| `on_llm_start` | InjectionDetector → DataVault → AuditLedger |
| `on_llm_end` | AuditLedger |
| `on_chain_start` | AuditLedger |
| `on_chain_end` | AuditLedger |

## Works with LangGraph Too

```python
from langgraph.graph import StateGraph
from air_langchain_trust import AirTrustCallbackHandler

handler = AirTrustCallbackHandler()
graph = StateGraph(...)
app = graph.compile()

# Callbacks propagate through the entire graph
result = app.invoke(input, config={"callbacks": [handler]})
```

## API Reference

```python
from air_langchain_trust import AirTrustCallbackHandler

handler = AirTrustCallbackHandler(config=None)

# Inspection methods
handler.get_audit_stats()   # → {"total_entries": 42, "chain_valid": True, ...}
handler.verify_chain()      # → {"valid": True, "total_entries": 42}
handler.export_audit()      # → [{"id": "...", "action": "tool_call", ...}, ...]
handler.get_vault_stats()   # → {"total_tokens": 5, "by_category": {"api_key": 3}}
```

## AIR Blackbox Ecosystem

| Repository | Purpose |
|-----------|---------|
| [gateway](https://github.com/airblackbox/gateway) | Go proxy gateway |
| [python-sdk](https://github.com/airblackbox/python-sdk) | Python SDK |
| [trust-openclaw](https://github.com/airblackbox/trust-openclaw) | TypeScript trust layer for OpenClaw |
| [trust-crewai](https://github.com/airblackbox/trust-crewai) | Python trust layer for CrewAI |
| **trust-langchain** | **Python trust layer for LangChain** (this repo) |

## Development

```bash
git clone https://github.com/airblackbox/trust-langchain.git
cd trust-langchain
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
