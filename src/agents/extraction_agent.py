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

from src.cache.llm_cache import get_cache
from src.unicode_map import sanitize_api_key, scrub_unicode

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
        # Scrub all chunk texts to plain ASCII before building the prompt
        scrubbed_chunks = [
            {**ch, "text": scrub_unicode(ch["text"])} for ch in chunks
        ]
        chunk_summaries = self._format_chunks_for_prompt(scrubbed_chunks)

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
            f"Retrieved Chunks (format: [Chunk N | source.pdf] text):\n{chunk_summaries}\n\n"
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
        ner_entities: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Extract structured entities grouped by the discovered categories.

        The LLM normalises ambiguous terms and grounds every entity with an
        evidence phrase from the provided chunks.

        Args:
            chunks: Retrieved chunks (text and metadata).
            categories: Output of ``discover_categories()``.
            query: The original research question.
            ner_entities: Optional deterministic NER entities from SciSpaCy
                to use as grounding hints.

        Returns:
            A dictionary whose top-level keys are category names. Each value
            is a list of entity objects containing at least ``entity``,
            ``evidence``, and ``source``. Additional keys (e.g. ``conditions``,
            ``direction``, ``context``) are populated as discovered (README §6.3).
        """
        # Scrub all chunk texts to plain ASCII before building the prompt
        scrubbed_chunks = [
            {**ch, "text": scrub_unicode(ch["text"])} for ch in chunks
        ]
        chunk_summaries = self._format_chunks_for_prompt(scrubbed_chunks)

        categories_str = json.dumps(categories, indent=2, ensure_ascii=False)

        # Build a stripped version of categories (no examples_found) to reduce prompt size.
        # The LLM generated these examples 30 seconds ago in category_discovery — it doesn't
        # need to re-read them. Keep names and descriptions only.
        stripped_categories = {
            k: [
                {sk: sv for sk, sv in c.items() if sk != "examples_found"}
                if isinstance(c, dict) else c
                for c in v
            ]
            if isinstance(v, list)
            else v
            for k, v in categories.items()
        }
        categories_str = json.dumps(stripped_categories, indent=2, ensure_ascii=False)

        ner_hint = ""
        if ner_entities:
            ner_lines = [f"  - {e['text']} ({e['label']}) [Chunk {e.get('source_chunk', '?')}]" for e in ner_entities[:30]]
            ner_hint = "Deterministic NER entities (use as hints; verify with evidence):\n" + "\n".join(ner_lines) + "\n\n"

        system_prompt = (
            "You are a biomedical entity extraction specialist. "
            "The research query, document chunks, and discovered categories "
            "ARE PROVIDED in the user message below. Do NOT state that data is missing. "
            "Extract all specific entities that fall under each category. For each entity you MUST:\n"
            ' - include an "evidence" field quoting the exact sentence(s) from the chunks.\n'
            ' - include a "source" field with the full chunk label including source PDF (e.g. "Chunk 3 | test.pdf").\n'
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
            f"{ner_hint}"
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
        """Compress chunk list into a numbered text block with PDF source labels."""
        lines = []
        for i, ch in enumerate(chunks):
            text = ch.get("text", "")
            clean = " ".join(text.split())
            src = (ch.get("metadata", {}) or {}).get("source", "?")
            lines.append(f"[Chunk {i} | {src}] {clean}")
        return "\n".join(lines)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return the full response text.

        Responses are cached by (system_prompt, user_prompt) hash with 24h TTL
        since temperature=0 makes outputs deterministic.
        """
        cache = get_cache()
        cached = cache.get(system_prompt, user_prompt)
        if cached is not None:
            return cached

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        config = {}
        if self.callback:
            config["callbacks"] = [self.callback]
        response = self._llm.invoke(messages, config=config)
        result = (response.content or "").strip()
        cache.set(system_prompt, user_prompt, result)
        return result

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
