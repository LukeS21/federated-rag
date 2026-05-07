"""
Extraction Agent — schema-less, query-conditioned entity extraction.

Uses Qwen3.6 35B-A3B via Ollama to discover thematic categories and extract
evidence-grounded entities from retrieved document chunks.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class ExtractionAgent:
    """Handles category discovery and structured entity extraction.

    All LLM calls use the local ``qwen3.6:35b-a3b`` model and enforce plain
    ASCII output from the model; responses are scrubbed before JSON parsing.
    """

    def __init__(
        self,
        model_name: str = "qwen3.6:35b",
        temperature: float = 0.0,
        num_ctx: int = 16384,
        client_kwargs: dict | None = None,
        callback=None,
    ) -> None:
        if client_kwargs is None:
            client_kwargs = {}
        self._llm = ChatOpenAI(
            model="deepseek-v4-pro",
            temperature=temperature,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            max_tokens=8192,
            timeout=120,
        )
        self.callback = callback

    # ------------------------------------------------------------------
    #  Category Discovery (Pass 1)
    # ------------------------------------------------------------------
    def discover_categories(self, chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Read retrieved chunks and identify recurring themes, variables,
        and methods — driven by the user's query.

        Args:
            chunks: List of chunk dicts, each with ``text`` and ``metadata``.
            query: The original research question.

        Returns:
            A dictionary with keys ``discovered_categories``,
            ``key_variables``, and ``experimental_methods`` as described
            in the architecture (README §6.2).
        """
        chunk_summaries = self._format_chunks_for_prompt(chunks)

        system_prompt = (
            "You are a biomedical literature analyst. Given a research query and a set of "
            "retrieved document chunks, identify the thematic categories that are relevant "
            "to the query. "
            "Output a JSON object with exactly three keys:\n"
            '  - "discovered_categories": list of objects, each with "name", '
            '"description", and "examples_found" (list of strings).\n'
            '  - "key_variables": list of variables being studied (strings).\n'
            '  - "experimental_methods": list of methodologies mentioned (strings).\n'
            "Use ONLY plain ASCII. Do not include any text outside the JSON object."
        )

        user_prompt = (
            f"Research Query: {query}\n\n"
            f"Retrieved Chunks (format: [Chunk N] text):\n{chunk_summaries}\n\n"
            "Discover the categories, key variables, and experimental methods."
        )

        raw_output = self._call_llm(system_prompt, user_prompt)
        return self._parse_json_safely(raw_output, "category_discovery")

    # ------------------------------------------------------------------
    #  Entity Extraction (Pass 2 — LLM only; NER deferred)
    # ------------------------------------------------------------------
    def extract_entities(
        self,
        chunks: List[Dict[str, Any]],
        categories: Dict[str, Any],
        query: str,
    ) -> Dict[str, Any]:
        """Extract structured entities grouped by the discovered categories.

        The LLM normalises ambiguous terms and grounds every entity with an
        evidence phrase from the provided chunks.

        Args:
            chunks: Retrieved chunks (text and metadata).
            categories: Output of ``discover_categories()``.
            query: The original research question.

        Returns:
            A dictionary whose top-level keys are category names. Each value
            is a list of entity objects containing at least ``entity``,
            ``evidence``, and ``source``. Additional keys (e.g. ``conditions``,
            ``direction``, ``context``) are populated as discovered (README §6.3).
        """
        chunk_summaries = self._format_chunks_for_prompt(chunks)

        categories_str = json.dumps(categories, indent=2, ensure_ascii=False)

        system_prompt = (
            "You are a biomedical entity extraction specialist. "
            "The research query, document chunks, and discovered categories "
            "ARE PROVIDED in the user message below. Do NOT state that data is missing. "
            "Extract all specific entities that fall under each category. For each entity you MUST:\n"
            ' - include an "evidence" field quoting the exact sentence(s) from the chunks.\n'
            ' - include a "source" field with the chunk number (e.g. "Chunk 3") the evidence came from.\n'
            ' - normalise synonyms (e.g. "TNF-\u03b1" and "TNF-alpha" become "TNF-alpha").\n'
            "Output a JSON object whose keys are EXACTLY the category names provided. "
            "The value for each key must be a list of objects. Every object must contain the "
            'keys "entity" (string), "evidence" (string), and "source" (string). '
            "You may add fields like \"conditions\", \"direction\", or \"context\" when applicable. "
            "Use ONLY plain ASCII. Do not include any text outside the JSON object."
        )

        user_prompt = (
            f"Research Query: {query}\n\n"
            f"Discovered Categories:\n{categories_str}\n\n"
            f"Retrieved Chunks (format: [Chunk N] text):\n{chunk_summaries}\n\n"
            "Extract all entities grouped by the categories above. For every entity, provide "
            "the evidence phrase and the source chunk number."
        )

        raw_output = self._call_llm(system_prompt, user_prompt)
        return self._parse_json_safely(raw_output, "entity_extraction")

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _format_chunks_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Compress chunk list into a numbered text block."""
        lines = []
        for i, ch in enumerate(chunks):
            text = ch.get("text", "")
            clean = " ".join(text.split())
            lines.append(f"[Chunk {i}] {clean}")
        return "\n".join(lines)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to Ollama and return the full response text."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        config = {}
        if self.callback:
            config["callbacks"] = [self.callback]
        response = self._llm.invoke(messages, config=config)
        return (response.content or "").strip()

    def _parse_json_safely(self, raw_text: str, context: str) -> Dict[str, Any]:
        """Parse JSON after ASCII-scrubbing; up to two parse attempts (second uses looser extraction)."""
        ascii_text = scrub_unicode(raw_text)
        if "<think>" in raw_text:
            logger.warning("Thinking block detected in LLM output for %s", context)
        ascii_text = _strip_thinking(ascii_text)
        logger.debug("Stripped thinking block. Remaining text: %s", ascii_text[:200])
        candidate = self._isolate_json_text(ascii_text)

        for attempt in range(2):
            try:
                parsed: Any = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise ValueError("JSON root must be an object")
                return parsed
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == 0:
                    logger.warning(
                        "JSON parse failed (attempt 1) for %s: %s. Trying brace extraction.",
                        context,
                        e,
                    )
                    candidate = self._isolate_json_text(ascii_text, brace_fallback=True)
                else:
                    raise ValueError(
                        f"Failed to parse LLM output as JSON after scrubbing. Context: {context}\n"
                        f"Raw output:\n{raw_text}\n"
                        f"Scrubbed output:\n{ascii_text}"
                    ) from e
        raise ValueError(f"Unexpected parse path for context: {context}")

    def _isolate_json_text(self, ascii_text: str, brace_fallback: bool = False) -> str:
        """Remove markdown fences and optionally keep only the outermost {...} block."""
        t = ascii_text.strip()
        if "```" in t:
            chosen = None
            for segment in t.split("```"):
                seg = segment.strip()
                if not seg:
                    continue
                if seg.lower().startswith("json"):
                    seg = seg[4:].lstrip()
                if seg.startswith("{") or seg.startswith("["):
                    chosen = seg
                    break
            if chosen is not None:
                t = chosen

        if brace_fallback:
            l, r = t.find("{"), t.rfind("}")
            if l != -1 and r != -1 and r > l:
                t = t[l : r + 1]

        return t.strip()
