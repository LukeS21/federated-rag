"""Arbiter – evidence-anchored revision of the draft."""

import os
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.cache.llm_cache import get_cache
from src.unicode_map import sanitize_api_key, scrub_unicode


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class Arbiter:
    """Revises the draft only where the Critic identified genuine gaps.

    Model: Qwen3.6 35B‑A3B (different prompt than Drafter).
    """

    def __init__(
        self,
        model_name: str = "qwen3.6:35b",
        num_ctx: int = 8192,
        client_kwargs: dict | None = None,
        callback=None,
        model: str | None = None,
    ) -> None:
        if client_kwargs is None:
            client_kwargs = {}
        self._model = model or "deepseek-v4-pro"
        self.llm = ChatOpenAI(
            model=self._model,
            temperature=0.0,
            api_key=sanitize_api_key(os.getenv("DEEPSEEK_API_KEY")),
            base_url="https://api.deepseek.com/v1",
            max_tokens=8192,
            timeout=120,
            default_headers={
                "User-Agent": "federated-rag",
                "Accept": "application/json",
            },
        )
        self.callback = callback

    def revise(
        self,
        draft: str,
        critique: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """Produce a revised synthesis paragraph (README §5.2)."""
        system_prompt = (
            "You are a biomedical synthesis arbiter. You receive a draft, a Socratic critique, "
            "and the original evidence. Revise the draft to address the critique.\n"
            "- For each critique, either cite specific evidence that supports the claim or modify/remove the claim.\n"
            "- Do not alter claims that were not critiqued.\n"
            "- Output the complete revised paragraph. Plain ASCII only."
        )

        chunks = [
            {**ch, "text": scrub_unicode(ch["text"])} for ch in chunks
        ]
        original_chunks = "\n\n".join(f"[Chunk {i}] {ch.get('text', '')}" for i, ch in enumerate(chunks))

        user_prompt = (
            f"Draft: {draft}\n"
            f"Critique: {critique}\n"
            f"Evidence: {original_chunks}\n"
            "Revise the draft, addressing each critique."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        cache = get_cache()
        cached = cache.get(system_prompt, user_prompt, model=self._model)
        if cached is not None:
            return scrub_unicode(cached)

        config = {}
        if self.callback:
            config["callbacks"] = [self.callback]
        response = self.llm.invoke(messages, config=config)
        raw = _strip_thinking((response.content or "").strip())
        result = scrub_unicode(raw)
        cache.set(system_prompt, user_prompt, result, model=self._model)
        return result

