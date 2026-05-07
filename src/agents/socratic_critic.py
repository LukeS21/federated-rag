"""Socratic Critic – evidence-grounded question generation.

Uses Gemma 4 26B A4B to resist peer-pressure convergence.
"""

import json
import os
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.unicode_map import sanitize_api_key, scrub_unicode


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class SocraticCritic:
    """Identifies unsupported claims without proposing alternative text.

    Model: Gemma 4 26B A4B (``gemma4:26b-a4b``).
    """

    def __init__(
        self,
        model_name: str = "gemma4:26b",
        num_ctx: int = 8192,
        client_kwargs: dict | None = None,
        callback=None,
    ) -> None:
        if client_kwargs is None:
            client_kwargs = {}
        self.llm = ChatOpenAI(
            model="deepseek-v4-pro",
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

    def critique(
        self,
        draft: str,
        chunks: List[Dict[str, Any]],
        entities: Dict[str, Any],
    ) -> str:
        """Return a list of critiques or ``NO_CRITIQUE`` (README §5.2)."""
        system_prompt = (
            "You are a Socratic critic. Your job is to identify claims in the draft that lack "
            "sufficient evidence or overstate what the evidence supports.\n"
            "- For each questionable claim, state what the evidence actually says.\n"
            "- Ask a specific question about an unsupported assertion.\n"
            "- NEVER propose alternative text or \"correct\" the draft.\n"
            "- If the draft is fully supported, state: \"NO_CRITIQUE: All claims are evidence-grounded.\"\n"
            "Output plain ASCII only."
        )

        chunks = [
            {**ch, "text": scrub_unicode(ch["text"])} for ch in chunks
        ]
        original_chunks = "\n\n".join(f"[Chunk {i}] {ch.get('text', '')}" for i, ch in enumerate(chunks))
        extracted_entities = json.dumps(entities, indent=2, ensure_ascii=False)

        user_prompt = (
            f"Draft: {draft}\n"
            f"Evidence: {original_chunks}\n"
            f"Entities: {extracted_entities}\n"
            "Identify unsupported claims and state the gap."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        config = {}
        if self.callback:
            config["callbacks"] = [self.callback]
        response = self.llm.invoke(messages, config=config)
        raw = _strip_thinking((response.content or "").strip())
        return scrub_unicode(raw)

