"""
Extraction Agent — schema-less, query-conditioned entity extraction.

Uses Qwen3.6 35B-A3B via Ollama to discover thematic categories and extract
evidence-grounded entities from retrieved document chunks.
"""

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.cache.llm_cache import get_cache
from src.llm import get_chat_model
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
        model: str = "deepseek-v4-pro",
    ) -> None:
        if client_kwargs is None:
            client_kwargs = {}
        self.model = model
        self._llm = get_chat_model(
            model=model,
            temperature=temperature,
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
    #  Entity Extraction (Pass 2 — line‑tagged output)
    # ------------------------------------------------------------------
    def extract_entities(
        self,
        chunks: List[Dict[str, Any]],
        categories: Dict[str, Any],
        query: str,
        ner_entities: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Extract structured entities grouped by the discovered categories.

        The LLM outputs a **line‑tagged** format instead of JSON.  Each
        entity block starts with ``TYPE: <category>`` and is terminated by
        a blank line.  This eliminates JSON parse failures (no braces,
        commas, or quotes to break) and uses ~30 % fewer tokens than
        pretty‑printed JSON with repeated field names.

        Args:
            chunks: Retrieved chunks (text and metadata).
            categories: Output of ``discover_categories()``.
            query: The original research question.
            ner_entities: Optional deterministic NER entities from SciSpaCy
                to use as grounding hints.

        Returns:
            A dictionary whose top-level keys are category names.  Each
            value is a list of entity objects containing at least
            ``entity``, ``evidence``, and ``source``.
        """
        # Scrub all chunk texts to plain ASCII before building the prompt
        scrubbed_chunks = [
            {**ch, "text": scrub_unicode(ch["text"])} for ch in chunks
        ]
        chunk_summaries = self._format_chunks_for_prompt(scrubbed_chunks)

        # Convert categories to line‑tagged text (avoids JSON overhead in prompt)
        categories_text = self._categories_to_line_tagged(categories)

        ner_hint = ""
        if ner_entities:
            ner_lines = [
                f"  - {e['text']} ({e['label']}) [Chunk {e.get('source_chunk', '?')}]"
                for e in ner_entities[:30]
            ]
            ner_hint = (
                "Deterministic NER entities (use as hints; verify with evidence):\n"
                + "\n".join(ner_lines)
                + "\n\n"
            )

        system_prompt = (
            "CRITICAL FORMAT INSTRUCTION — YOU MUST FOLLOW THIS EXACTLY. "
            "Failure to use the correct format means your output will be DISCARDED.\n\n"
            "You are a biomedical entity extraction specialist. "
            "Given research chunks and discovered categories, extract every entity.\n\n"
            "--- CORRECT OUTPUT FORMAT ---\n"
            "Each entity block uses KEY: value lines, separated by one blank line:\n\n"
            "TYPE: material\n"
            "ENTITY: Ti-6Al-4V\n"
            "DIRECTION: unchanged\n"
            "EVIDENCE: Ti-6Al-4V alloy was used for all implantation experiments.\n"
            "SOURCE: Chunk 5 | paper_a.pdf\n\n"
            "TYPE: cytokine\n"
            "ENTITY: IL-6\n"
            "DIRECTION: elevated\n"
            "EVIDENCE: IL-6 levels were significantly elevated in the treated group.\n"
            "SOURCE: Chunk 12 | paper_b.pdf\n\n"
            "--- END OF FORMAT EXAMPLE ---\n\n"
            "FORMAT RULES (VIOLATIONS WILL BE REJECTED):\n"
            "1. Every entity block MUST start with TYPE: on its own line.\n"
            "2. Required fields: TYPE:, ENTITY:, EVIDENCE:, SOURCE:.\n"
            "3. Optional fields: DIRECTION:, CONTEXT:, CONDITIONS:.\n"
            "4. Separate entities with ONE blank line (empty line).\n"
            "5. NO markdown — NO **bold**, NO *italics*, NO bullet points, NO headers.\n"
            "6. NO JSON — NO braces, NO brackets, NO quotes.\n"
            "7. NO preamble — do NOT write 'Here are the entities' or 'Keywords:'.\n"
            "8. NO summary or conclusion at the end — just the entity blocks.\n"
            "9. QUOTE evidence VERBATIM from the chunks. Do not paraphrase.\n"
            "10. Plain ASCII only.\n\n"
            "WRONG FORMAT (DO NOT DO THIS):\n"
            '  **Keywords:**  *Materials:* Titanium, ...    <-- WRONG, NO markdown\n'
            '  {"category": "materials", "entities": [...]} <-- WRONG, NO JSON\n'
            '  - Titanium (evidence: ...)                    <-- WRONG, NO bullets\n'
        )

        user_prompt = (
            f"Research Query: {query}\n\n"
            f"Discovered Categories:\n{categories_text}\n\n"
            f"{ner_hint}"
            f"Retrieved Chunks (format: [Chunk N | source] text):\n{chunk_summaries}\n\n"
            "INSTRUCTIONS:\n"
            "1. Use ONLY the line‑tagged format shown in the system prompt.\n"
            "2. Start every entity with TYPE: on its own line.\n"
            "3. Always include ENTITY:, EVIDENCE:, and SOURCE: for every entity.\n"
            "4. Copy evidence VERBATIM from the chunks above — do not summarize.\n"
            "5. NO markdown, NO JSON, NO bullet points.\n"
            "6. Begin immediately with the first TYPE: line. No introduction."
        )

        raw_output = self._call_llm(system_prompt, user_prompt)
        result = self._parse_line_tagged(raw_output, context="entity_extraction")

        # Fallback: if line‑tagged produced nothing, try parsing markdown keyword lists
        if not result:
            result = self._parse_markdown_fallback(raw_output)
            if result:
                logger.info("Markdown fallback parser recovered %d categories from entity_extraction",
                             len(result))
            else:
                logger.warning(
                    "Both line‑tagged and markdown parsers failed for entity_extraction. "
                    "Raw (first 300 chars): %s",
                    raw_output[:300],
                )

        return result

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
        cached = cache.get(system_prompt, user_prompt, model=self.model)
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
        cache.set(system_prompt, user_prompt, result, model=self.model)
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

    # ------------------------------------------------------------------
    #  Line‑tagged format (Phase 10 — replaces JSON for entity extraction)
    # ------------------------------------------------------------------

    @staticmethod
    def _categories_to_line_tagged(categories: Dict[str, Any]) -> str:
        """Format discovered categories as line‑tagged text.

        Converts the Pass 1 JSON output into a compact text format for the
        Pass 2 LLM prompt — saves ~30 % tokens vs pretty‑printed JSON.
        """
        lines: List[str] = []
        discovered = categories.get("discovered_categories", [])
        if isinstance(discovered, list):
            for cat in discovered:
                if isinstance(cat, dict):
                    name = cat.get("name", "")
                    desc = cat.get("description", "")
                    if name:
                        lines.append(f"CATEGORY: {name}")
                    if desc:
                        lines.append(f"DESCRIPTION: {desc}")
                    lines.append("")

        key_vars = categories.get("key_variables", [])
        if isinstance(key_vars, list) and key_vars:
            lines.append("KEY_VARIABLES: " + ", ".join(str(v) for v in key_vars))
            lines.append("")

        methods = categories.get("experimental_methods", [])
        if isinstance(methods, list) and methods:
            lines.append("METHODS: " + ", ".join(str(m) for m in methods))
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _parse_line_tagged(text: str, context: str = "") -> Dict[str, Any]:
        """Parse line‑tagged entity blocks into the standard dict format.

        Each entity block is separated by a blank line and starts with
        ``TYPE: category_name``.  Individual fields are ``KEY: value``
        lines.  Returns a dict keyed by category name (same structure as
        the old JSON parser).
        """
        text = scrub_unicode(text)
        if "<think>" in text:
            text = _strip_thinking(text)

        result: Dict[str, List[Dict[str, str]]] = {}
        current_entity: Dict[str, str] = {}
        current_type = ""

        for line in text.strip().split("\n"):
            stripped = line.strip()
            if not stripped:
                if current_entity and current_type:
                    result.setdefault(current_type, []).append(dict(current_entity))
                    current_entity = {}
                    current_type = ""
                continue

            if ":" not in stripped:
                continue

            key, _, value = stripped.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key == "type":
                if current_entity and current_type:
                    result.setdefault(current_type, []).append(dict(current_entity))
                    current_entity = {}
                current_type = value
            elif value:
                current_entity[key] = value

        # Flush trailing entity
        if current_entity and current_type:
            result.setdefault(current_type, []).append(dict(current_entity))

        if not result:
            logger.warning(
                "Line‑tagged parse produced empty result for %s. "
                "Raw (first 200 chars): %s",
                context,
                text[:200],
            )

        return result

    @staticmethod
    def _parse_markdown_fallback(text: str) -> Dict[str, Any]:
        """Fallback parser for markdown-formatted keyword lists.

        Handles output patterns like::

            **Keywords:**
            * **Materials:** Titanium, Titanium Alloy, Stainless Steel, ...
            * **Methods:** ELISA, flow cytometry, microCT, ...

            **Bioactive Materials and Composites**
            * **Bioactive Glass:** (description)
            * **Bioactive Ceramic:** (description)

        Each ``**bold header**`` becomes a category, and ``* **bold entity:**``
        items become entities under that category.  Evidence defaults to
        the description text after the colon.
        """
        text = scrub_unicode(text)
        if "<think>" in text:
            text = _strip_thinking(text)

        result: Dict[str, List[Dict[str, str]]] = {}
        current_category = ""

        lines = text.strip().split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Detect **Category Header** (standalone bold line)
            bold_header = re.match(r"^\*\*(.+?)\*\*\s*$", stripped)
            if bold_header:
                name = bold_header.group(1).strip().rstrip(":")
                if name.lower() not in ("keywords", "key variables", "experimental methods"):
                    current_category = name
                continue

            # Detect comma-separated list after bullet (* Materials: Ti, Al, ...)
            # Checked first — comma-delimited items get split into separate entities.
            bullet_list = re.match(r"^\*\s*(?:\*\*(.+?)\*\*:?|(.+?):)\s*(.+)", stripped)
            if bullet_list:
                cat_name = (bullet_list.group(1) or bullet_list.group(2) or "").strip().rstrip(":")
                items_text = (bullet_list.group(3) or "").strip()
                if "," in items_text:
                    items = [i.strip().rstrip("*") for i in items_text.split(",") if len(i.strip()) > 2]
                    if items and cat_name:
                        for item in items:
                            result.setdefault(cat_name, []).append({
                                "entity": item,
                                "evidence": "",
                                "source": "",
                            })
                        continue
                # Fall through to bullet_entity — not a comma list

            # Detect * **Entity:** description (single bold entity per bullet)
            bullet_entity = re.match(r"^\*\s*\*\*(.+?)\*\*:?\s*(.*)", stripped)
            if bullet_entity:
                entity_name = bullet_entity.group(1).strip().rstrip(":")
                description = bullet_entity.group(2).strip()

                if entity_name:
                    result.setdefault(current_category or entity_name, []).append({
                        "entity": entity_name,
                        "evidence": description or "",
                        "source": "",
                    })
                continue

            # Detect * **Entity** (bold, no colon, no description)
            bullet_no_colon = re.match(r"^\*\s*\*\*(.+?)\*\*\s*$", stripped)
            if bullet_no_colon:
                entity_name = bullet_no_colon.group(1).strip()
                if entity_name:
                    result.setdefault(current_category or entity_name, []).append({
                        "entity": entity_name,
                        "evidence": "",
                        "source": "",
                    })
                continue

        if not result:
            return {}

        logger.info(
            "Markdown fallback parsed %d categories from %d-char output",
            len(result), len(text),
        )
        return result
        """Parse line‑tagged entity blocks into the standard dict format.

        Each entity block is separated by a blank line and starts with
        ``TYPE: category_name``.  Individual fields are ``KEY: value``
        lines.  Returns a dict keyed by category name (same structure as
        the old JSON parser).
        """
        text = scrub_unicode(text)
        if "<think>" in text:
            text = _strip_thinking(text)

        result: Dict[str, List[Dict[str, str]]] = {}
        current_entity: Dict[str, str] = {}
        current_type = ""

        for line in text.strip().split("\n"):
            stripped = line.strip()
            if not stripped:
                if current_entity and current_type:
                    result.setdefault(current_type, []).append(dict(current_entity))
                    current_entity = {}
                    current_type = ""
                continue

            if ":" not in stripped:
                continue

            key, _, value = stripped.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key == "type":
                if current_entity and current_type:
                    result.setdefault(current_type, []).append(dict(current_entity))
                    current_entity = {}
                current_type = value
            elif value:
                current_entity[key] = value

        # Flush trailing entity
        if current_entity and current_type:
            result.setdefault(current_type, []).append(dict(current_entity))

        if not result:
            logger.warning(
                "Line‑tagged parse produced empty result for %s. "
                "Raw (first 200 chars): %s",
                context,
                text[:200],
            )

        return result
