"""Chunk Summarizer – condenses retrieved chunks into a compact evidence abstract.

Runs once after retrieval, before downstream agents, to cut token usage ~5x.
"""

import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.cache.llm_cache import get_cache
from src.unicode_map import sanitize_api_key, scrub_unicode


class Summarizer:
    """Produces a condensed evidence abstract from retrieved document chunks.

    Downstream agents (category discovery, drafter, critic, arbiter) consume
    this summary instead of raw chunk text, reducing per-agent token usage.
    Extraction still uses raw chunks for evidence-grounding quotes.
    """

    def __init__(
        self,
        model_name: str = "deepseek-v4-pro",
        num_ctx: int = 16384,
        client_kwargs: dict | None = None,
        callback=None,
    ) -> None:
        if client_kwargs is None:
            client_kwargs = {}
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.0,
            api_key=sanitize_api_key(os.getenv("DEEPSEEK_API_KEY")),
            base_url="https://api.deepseek.com/v1",
            max_tokens=512,
            timeout=120,
            default_headers={
                "User-Agent": "federated-rag",
                "Accept": "application/json",
            },
        )
        self.callback = callback

    def summarize(self, chunks: List[Dict[str, Any]], query: str) -> str:
        """Condense chunks into a ~500-word evidence abstract tailored to the query."""

        scrubbed = [{**ch, "text": scrub_unicode(ch["text"])} for ch in chunks]
        chunk_texts = "\n\n".join(
            f"[Chunk {i}] {ch.get('text', '')}" for i, ch in enumerate(scrubbed)
        )

        system_prompt = (
            "You are a biomedical evidence summarizer. Given document chunks and a "
            "research query, produce a concise evidence abstract. Include:\n"
            "- Key findings relevant to the query\n"
            "- Specific quantitative data (percentages, counts, p-values)\n"
            "- Methodological details that matter for interpretation\n"
            "- Any contradictions or differing results across sources\n"
            "Output plain text only, no markdown, no commentary. Keep it under 500 words."
        )

        user_prompt = (
            f"Research Query: {query}\n\n"
            f"Document Chunks:\n{chunk_texts}\n\n"
            "Produce a concise evidence abstract summarizing the information "
            "relevant to the query above."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Check cache first (LLM responses at temperature=0 are deterministic)
        cache = get_cache()
        cached = cache.get(system_prompt, user_prompt, model="deepseek-chat")
        if cached is not None:
            return scrub_unicode(cached)

        config = {}
        if self.callback:
            config["callbacks"] = [self.callback]
        response = self.llm.invoke(messages, config=config)
        result = scrub_unicode((response.content or "").strip())
        cache.set(system_prompt, user_prompt, result, model="deepseek-chat")
        return result
