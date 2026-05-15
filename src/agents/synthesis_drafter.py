"""Synthesis Drafter – first-pass literature review synthesis."""

import json
import logging
import re
import time
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.cache.llm_cache import get_cache
from src.llm import get_chat_model
from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _entities_to_line_tagged(entities: Dict[str, Any]) -> str:
    """Format entity dict as compact line‑tagged text for the Drafter prompt.

    Saves ~25-30 % tokens vs ``json.dumps(indent=2)`` by eliminating
    repeated field names, quotes, commas, and braces.
    """
    if not entities:
        return "(none)"

    lines: List[str] = []
    for category, entity_list in sorted(entities.items()):
        if not isinstance(entity_list, list) or not entity_list:
            continue
        lines.append(f"## {category}")
        for ent in entity_list:
            if not isinstance(ent, dict):
                continue
            for key, value in ent.items():
                key_label = key.upper().replace("_", " ")
                val = str(value).strip()
                if val:
                    lines.append(f"  {key_label}: {val}")
            lines.append("")
    return "\n".join(lines)


class SynthesisDrafter:
    """Writes an initial synthesis paragraph with inline citations.

    Uses Qwen3.6 35B‑A3B for its strong agentic tool‑use score.
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
        self.llm = get_chat_model(
            model=self._model,
            temperature=0.0,
        )
        self.callback = callback

    def draft(
        self,
        query: str,
        entities: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        citations: List[str],
        kg_context: Dict[str, Any] | str,
    ) -> str:
        """Produce a draft synthesis paragraph (README §5.2)."""
        system_prompt = (
            "You are a biomedical literature synthesis drafter. Produce evidence-backed "
            "claims with inline citation keys from the provided Available Citations list. "
            "Use ONLY the exact citation keys provided — never invent new ones. "
            "Be as concise as possible — "
            "prefer dense factual claims over narrative prose. Preserve ALL key findings, "
            "contradictions, and quantitative data from the evidence. Every claim must "
            "be traceable to a provided evidence chunk. Use knowledge graph insights "
            "to identify cross-cutting relationships. Output plain ASCII only.\n"
            "Format: one claim per line. No preamble, no transitions, no repetition."
        )

        entities_json = json.dumps(entities, indent=2, ensure_ascii=False)
        chunks = [
            {**ch, "text": scrub_unicode(ch["text"])} for ch in chunks
        ]
        chunk_texts = "\n\n".join(f"[Chunk {i}] {ch.get('text', '')}" for i, ch in enumerate(chunks))
        cite_keys = ", ".join(citations) if citations else "none provided"
        kg_text = kg_context if isinstance(kg_context, str) else json.dumps(kg_context or {}, indent=2, ensure_ascii=False)

        # Use line‑tagged format for entities (saves ~25-30 % tokens vs JSON)
        entities_text = _entities_to_line_tagged(entities)

        user_prompt = (
            f"Query: {query}\n"
            f"Extracted Entities:\n{entities_text}\n"
            f"Evidence Summaries: {chunk_texts}\n"
            f"Available Citations: {cite_keys}\n"
            f"Knowledge Graph Context: {kg_text}\n"
            "Produce evidence-backed claims from this information. "
            "One claim per line. No preamble or transitions."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        cache = get_cache()
        cached = cache.get(system_prompt, user_prompt, model=self._model)
        if cached is not None:
            logger.info("Drafter cache hit (model=%s, %d chars)", self._model, len(cached))
            return scrub_unicode(cached)

        config = {}
        if self.callback:
            config["callbacks"] = [self.callback]

        prompt_chars = len(system_prompt) + len(user_prompt)
        logger.info("Drafter invoke start (model=%s, prompt=%d chars)", self._model, prompt_chars)
        t0 = time.time()
        response = self.llm.invoke(messages, config=config)
        elapsed = time.time() - t0
        raw = _strip_thinking((response.content or "").strip())
        result = scrub_unicode(raw)
        logger.info("Drafter invoke done (model=%s, %d chars output, %.1fs)",
                     self._model, len(result), elapsed)
        cache.set(system_prompt, user_prompt, result, model=self._model)
        return result

