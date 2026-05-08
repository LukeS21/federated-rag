"""Query Decomposition Agent — breaks complex research questions into
theme-focused sub-queries for Survey Mode.

This is the entry point for Survey Mode. It takes a broad biomedical
research question and identifies distinct thematic sub-questions that
can each be used for focused retrieval and per-theme synthesis.
"""

import json
import logging
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.cache.llm_cache import get_cache
from src.unicode_map import sanitize_api_key, scrub_unicode

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """Breaks a complex research question into theme-discovery sub-queries.

    Uses DeepSeek Chat by default (cheaper, adequate for classification tasks).
    Each sub-query targets a specific thematic dimension of the original question
    and can be used independently for retrieval and synthesis.
    """

    def __init__(self, model: str = "deepseek-chat") -> None:
        self.model = model
        self._llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=sanitize_api_key(os.getenv("DEEPSEEK_API_KEY")),
            base_url="https://api.deepseek.com/v1",
            max_tokens=2048,
            timeout=120,
            default_headers={
                "User-Agent": "federated-rag",
                "Accept": "application/json",
            },
        )

    def decompose(self, query: str) -> Dict[str, Any]:
        """Decompose a broad research question into thematic sub-queries.

        Args:
            query: A broad biomedical research question (e.g.,
                   "Map the current understanding of biomaterial surface
                   modifications and immune response in obese models").

        Returns:
            A dictionary with keys:
            - ``original_query``: the input query
            - ``themes``: list of objects, each with:
                - ``theme`` (str): short label for the theme
                - ``sub_query`` (str): focused query for retrieval
                - ``rationale`` (str): why this theme matters for the query
            - ``cross_cutting_themes``: list of strings identifying themes
              that span multiple sub-queries
        """
        system_prompt = (
            "You are a biomedical research analyst specializing in literature survey design. "
            "Given a broad research question, decompose it into distinct thematic sub-questions "
            "that each target a specific aspect of the question.\n\n"
            "Rules:\n"
            "- Identify 3-8 distinct themes. Each theme should be independently searchable.\n"
            "- Each sub-query should be a complete, self-contained question suitable for "
            "retrieval from a biomedical corpus.\n"
            "- Include cross-cutting themes that span multiple sub-queries (e.g., shared "
            "methodologies, common model systems).\n"
            "- Prefer specific over vague (e.g., 'role of IL-6 in macrophage polarization "
            "at titanium implants' over 'immune response').\n\n"
            "Output a JSON object with exactly three keys:\n"
            '  - "themes": list of objects, each with "theme", "sub_query", "rationale"\n'
            '  - "cross_cutting_themes": list of strings\n'
            "Use ONLY plain ASCII. Do not include any text outside the JSON object."
        )

        user_prompt = (
            f"Research Question: {query}\n\n"
            "Decompose this question into focused thematic sub-queries for a systematic "
            "literature survey. Output ONLY the JSON object."
        )

        cache_key_model = self.model
        cache = get_cache()
        cached = cache.get(system_prompt, user_prompt, model=cache_key_model)
        if cached is not None:
            return json.loads(scrub_unicode(cached))

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self._llm.invoke(messages)
        raw = scrub_unicode((response.content or "").strip())
        cache.set(system_prompt, user_prompt, raw, model=cache_key_model)

        result = self._parse_json(raw)
        result["original_query"] = query
        return result

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM output with markdown-fence handling."""
        text = raw_text.strip()
        if "```" in text:
            for segment in text.split("```"):
                seg = segment.strip()
                if seg.lower().startswith("json"):
                    seg = seg[4:].lstrip()
                if seg.startswith("{") or seg.startswith("["):
                    text = seg
                    break
        # Brace fallback
        l, r = text.find("{"), text.rfind("}")
        if l != -1 and r != -1 and r > l:
            text = text[l : r + 1]
        return json.loads(text)
