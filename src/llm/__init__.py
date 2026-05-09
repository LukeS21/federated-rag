"""Unified LLM provider — local Ollama and optional DeepSeek API (Phase 5).

Supports two backends selectable via ``LLM_PROVIDER`` env var:
  - ``ollama``  — local Ollama instance(s), dual-instance for air-gap (DEFAULT)
  - ``deepseek`` — DeepSeek cloud API (opt-in; WARNING: sends data to cloud)

For air-gap deployments, two Ollama hosts are configured:
  - ``OLLAMA_PUBLIC_HOST`` — internet-accessible Ollama (public corpus)
  - ``OLLAMA_SECURE_HOST`` — air-gapped Ollama (secure corpus)

⚠️  SECURITY: Secure-scope queries are NEVER routed to DeepSeek.
    ``get_chat_model_for_scope`` raises ``RuntimeError`` if a secure query
    would be sent to the cloud.  Public-scope DeepSeek usage logs a warning.

Use :func:`get_chat_model_for_scope` to route LLM calls to the correct
Ollama instance based on ``query_scope``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from langchain_openai import ChatOpenAI

from src.security.audit_log import get_audit_logger
from src.unicode_map import sanitize_api_key

logger = logging.getLogger(__name__)

# Provider configuration (initialized once, cached)
_config_lock = threading.Lock()

_provider: str | None = None
_base_url: str | None = None
_api_key: str | None = None
_public_base_url: str | None = None
_secure_base_url: str | None = None
_small_model: str | None = None
_large_model: str | None = None
_alt_model: str | None = None


def _init_provider_config() -> None:
    """Lazy-init provider configuration from environment."""
    global _provider, _base_url, _api_key
    global _public_base_url, _secure_base_url
    global _small_model, _large_model, _alt_model

    if _provider is not None:
        return
    with _config_lock:
        if _provider is not None:
            return

        _provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

        if _provider == "deepseek":
            logger.warning(
                "LLM_PROVIDER=deepseek — queries will be sent to DeepSeek cloud API. "
                "This is a privacy risk. Secure-scope queries are blocked."
            )
            _base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            _api_key = sanitize_api_key(os.getenv("DEEPSEEK_API_KEY")) or ""
            _small_model = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
            _large_model = os.getenv("DEEPSEEK_REASONING_MODEL", "deepseek-v4-pro")
        else:
            if _provider not in ("ollama", ""):
                logger.warning("Unknown LLM_PROVIDER=%r — falling back to ollama", _provider)
                _provider = "ollama"
            _public_base_url = (
                os.getenv("OLLAMA_PUBLIC_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
                + "/v1"
            )
            _secure_base_url = (
                os.getenv("OLLAMA_SECURE_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
                + "/v1"
            )
            _base_url = _public_base_url
            _api_key = "ollama"
            _small_model = os.getenv("OLLAMA_SMALL_MODEL", "qwen3.6:35b-a3b")
            _large_model = os.getenv("OLLAMA_LARGE_MODEL", "qwen3.6:35b-a3b")
            _alt_model = os.getenv("OLLAMA_ALT_MODEL", "medgemma:4b")

        logger.info("LLM provider: %s (public=%s, secure=%s, small=%s, large=%s, alt=%s)",
                     _provider, _public_base_url, _secure_base_url, _small_model, _large_model, _alt_model)


def get_provider() -> str:
    _init_provider_config()
    return _provider or "deepseek"


def get_base_url() -> str:
    _init_provider_config()
    return _base_url or ""


def get_api_key() -> str:
    _init_provider_config()
    return _api_key or ""


def resolve_model(model: str | None) -> str:
    """Resolve a model name based on tier keywords.

    ``"chat"`` or ``"small"`` → small/fast tier model.
    ``"pro"`` or ``"large"`` → reasoning tier model.
    Otherwise, return *model* unchanged.

    When *model* is None, returns the configured large-tier model
    (Ollama: OLLAMA_LARGE_MODEL; DeepSeek: DEEPSEEK_REASONING_MODEL).
    """
    _init_provider_config()
    if model is None:
        return _large_model or "qwen3.6:35b-a3b"

    lower = model.lower()
    if "chat" in lower or "small" in lower:
        return _small_model or model
    if "alt" in lower:
        return _alt_model or model
    if "pro" in lower or "large" in lower:
        return _large_model or model
    return model


def get_chat_model(
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int = 2,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a configured LangChain ChatOpenAI instance.

    *max_tokens* defaults to ``LLM_MAX_TOKENS`` env var (4096).  Lower
    values reduce generation time for small outputs (synthesis, critique).

    *timeout* defaults to the ``LLM_TIMEOUT`` env var (seconds), or 300
    if unset.  Local Ollama may need higher values when requests queue.
    """
    if max_tokens is None:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    if timeout is None:
        timeout = int(os.getenv("LLM_TIMEOUT", "300"))
    resolved_model = resolve_model(model)

    if get_provider() == "ollama":
        return ChatOpenAI(
            model=resolved_model,
            temperature=temperature,
            api_key="ollama",
            base_url=get_base_url(),
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    # DeepSeek provider — log privacy warning
    logger.warning(
        "LLM call via DeepSeek API (model=%s). Data is sent to cloud. "
        "Prefer LLM_PROVIDER=ollama for air-gap security.",
        resolved_model,
    )
    return ChatOpenAI(
        model=resolved_model,
        temperature=temperature,
        api_key=get_api_key(),
        base_url=get_base_url(),
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        default_headers={
            "User-Agent": "federated-rag",
            "Accept": "application/json",
        },
    )


def get_chat_model_for_scope(
    query_scope: str = "public",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int = 2,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance routed to the correct Ollama host.

    *max_tokens* defaults to ``LLM_MAX_TOKENS`` env var (4096).
    *timeout* defaults to the ``LLM_TIMEOUT`` env var (seconds), or 300.
    """
    if max_tokens is None:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    if timeout is None:
        timeout = int(os.getenv("LLM_TIMEOUT", "300"))
    resolved_model = resolve_model(model)

    if get_provider() == "ollama":
        _init_provider_config()
        if query_scope == "secure":
            base_url = _secure_base_url or "http://ollama-secure:11434/v1"
        else:
            base_url = _public_base_url or "http://ollama-public:11434/v1"

        return ChatOpenAI(
            model=resolved_model,
            temperature=temperature,
            api_key="ollama",
            base_url=base_url,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    # DeepSeek provider — block secure scope from cloud routing
    if query_scope == "secure":
        raise RuntimeError(
            "SECURITY VIOLATION: query_scope='secure' cannot use DeepSeek API. "
            "Set LLM_PROVIDER=ollama for secure-corpus queries. "
            "Secure data must never leave the air-gap."
        )

    if query_scope in ("public", "both"):
        logger.warning(
            "LLM_PROVIDER=deepseek with query_scope=%r — data will be sent to cloud API. "
            "Ensure no proprietary/secure data is in the prompt.",
            query_scope,
        )

    return ChatOpenAI(
        model=resolved_model,
        temperature=temperature,
        api_key=get_api_key(),
        base_url=get_base_url(),
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        default_headers={
            "User-Agent": "federated-rag",
            "Accept": "application/json",
        },
    )


# ── Instrumented invoke wrapper ────────────────────────────────────────────
# Agents can optionally wrap their llm.invoke() to get audit logging.
# Usage: response = audited_invoke(llm, messages, operation="drafter",
#                                  model=..., cached=False)


def audited_invoke(
    llm: ChatOpenAI,
    messages: list,
    operation: str = "llm_call",
    model: str | None = None,
    cached: bool = False,
) -> Any:
    """Invoke *llm* with automatic audit logging of latency and token counts."""
    prompt_chars = sum(len(str(m.content)) for m in messages)
    t0 = time.monotonic()
    try:
        response = llm.invoke(messages)
    except Exception:
        # Log failure even on exception
        latency_ms = (time.monotonic() - t0) * 1000
        try:
            audit = get_audit_logger()
            audit.log_llm_call(
                model=model or getattr(llm, "model_name", "unknown"),
                provider=get_provider(),
                operation=operation,
                prompt_chars=prompt_chars,
                response_chars=0,
                latency_ms=latency_ms,
                cached=cached,
            )
        except Exception:
            pass
        raise

    latency_ms = (time.monotonic() - t0) * 1000
    response_chars = len(str(response.content or ""))

    try:
        audit = get_audit_logger()
        audit.log_llm_call(
            model=model or getattr(llm, "model_name", "unknown"),
            provider=get_provider(),
            operation=operation,
            prompt_chars=prompt_chars,
            response_chars=response_chars,
            latency_ms=latency_ms,
            cached=cached,
        )
    except Exception:
        logger.debug("Audit logging failed (non-fatal)", exc_info=True)

    return response
