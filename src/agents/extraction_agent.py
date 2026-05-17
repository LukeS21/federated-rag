"""
Extraction Agent — schema-less, query-conditioned entity extraction.

Uses Qwen3.6 35B-A3B via Ollama to discover thematic categories and extract
evidence-grounded entities from retrieved document chunks.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.cache.llm_cache import get_cache
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
            max_retries=0,
            streaming=True,
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
            "Group entities that share the SAME evidence sentence together. "
            "State the shared fields once, then list each entity compactly:\n\n"
            "EVIDENCE: Polymeric nanocarriers like polyethyleneimine, dendrimers, and graphene-based materials offer efficient, non-viral alternatives, with magnetic nanoparticles showing promise.\n"
            "SOURCE: Chunk 1 | paper_a.pdf\n"
            "TYPE: material\n"
            "ENTITY: polyethyleneimine\n"
            "ENTITY: dendrimers\n"
            "ENTITY: graphene-based materials | CLAIM: elevated\n\n"
            "EVIDENCE: IL-6 levels were significantly elevated in the treated group compared to controls.\n"
            "SOURCE: Chunk 12 | paper_b.pdf\n"
            "TYPE: cytokine\n"
            "ENTITY: IL-6 | CLAIM: elevated\n\n"
            "EVIDENCE: The sensor exhibited a sensitivity of 0.65 uA.mM-1 with a detection limit of 1.3 x 10-2 M.\n"
            "SOURCE: Chunk 5 | paper_c.pdf\n"
            "TYPE: sensor performance\n"
            "ENTITY: sensitivity | CLAIM: 0.65 uA.mM-1\n"
            "ENTITY: detection limit | CLAIM: 1.3 x 10-2 M\n\n"
            "EVIDENCE: Ti-6Al-4V alloy was used for all implantation experiments.\n"
            "SOURCE: Chunk 5 | paper_a.pdf\n"
            "TYPE: material\n"
            "ENTITY: Ti-6Al-4V\n\n"
            "EVIDENCE: Macrophages polarized toward an M2 anti-inflammatory phenotype after 24h treatment.\n"
            "SOURCE: Chunk 22 | paper_d.pdf\n"
            "TYPE: cell type\n"
            "ENTITY: macrophage | CLAIM: M2 phenotype | CONTEXT: after 24h treatment\n\n"
            "--- END OF FORMAT EXAMPLE ---\n\n"
            "FORMAT RULES (VIOLATIONS WILL BE REJECTED):\n"
            "1. Group entities that share the SAME evidence sentence — do NOT repeat evidence.\n"
            "2. Start each group with EVIDENCE:, SOURCE:, and TYPE: (any order).\n"
            "3. Each entity is a compact line: ENTITY: name | CLAIM: value\n"
            "4. CLAIM captures what the evidence says ABOUT this entity:\n"
            "   a) Qualitative change:  elevated, decreased, increased, reduced,\n"
            "      upregulated, downregulated, up, down.\n"
            "   b) Quantitative measurement:  0.65 uA.mM-1, 11 V, R2 = 0.993,\n"
            "      18 s, 2 Pa.  Quote the value directly from evidence.\n"
            "   c) State, role, or identity:  M2 phenotype, matrix material,\n"
            "      pro-inflammatory cytokine.\n"
            "   Omit CLAIM entirely when the evidence simply mentions the entity\n"
            "   without making a specific claim about it.  NO filler values.\n\n"
            "   WRONG:  ENTITY: PVDF | CLAIM: unchanged       <- omit instead\n"
            "   WRONG:  ENTITY: sensors | CLAIM: N/A           <- omit instead\n"
            "   CORRECT: ENTITY: PVDF                           <- no claim to make\n"
            "   CORRECT: ENTITY: sensitivity | CLAIM: 0.65 uA.mM-1\n"
            "   CORRECT: ENTITY: IL-6 | CLAIM: elevated\n"
            "5. Optional per-entity attributes: CLAIM:, CONTEXT:, CONDITIONS:\n"
            "6. Separate groups with ONE blank line (empty line).\n"
            "7. NO markdown — NO **bold**, NO *italics*, NO bullet points, NO headers.\n"
            "8. NO JSON — NO braces, NO brackets, NO quotes.\n"
            "9. NO preamble — do NOT write 'Here are the entities' or 'Keywords:'.\n"
            "10. NO summary or conclusion at the end — just the entity groups.\n"
            "11. QUOTE evidence VERBATIM from the chunks. Do not paraphrase.\n"
            "12. Plain ASCII only.\n\n"
            "WRONG FORMAT (DO NOT DO THIS):\n"
            '  **Keywords:**  *Materials:* Titanium, ...    <-- WRONG, NO markdown\n'
            '  {"category": "materials", "entities": [...]} <-- WRONG, NO JSON\n'
            '  - Titanium (evidence: ...)                    <-- WRONG, NO bullets\n'
            '  Repeating the same EVIDENCE for every entity  <-- WRONG, GROUP THEM\n'
            '  ENTITY: PVDF | CLAIM: unchanged                <-- WRONG, omit CLAIM\n'
            '  ENTITY: implant | CLAIM: target                <-- WRONG, omit CLAIM\n'
            '  ENTITY: CBCT images | CLAIM: source             <-- WRONG, omit CLAIM\n'
        )

        user_prompt = (
            f"Research Query: {query}\n\n"
            f"Discovered Categories:\n{categories_text}\n\n"
            f"{ner_hint}"
            f"Retrieved Chunks (format: [Chunk N | source] text):\n{chunk_summaries}\n\n"
            "INSTRUCTIONS:\n"
            "1. Group entities that share the SAME evidence sentence together.\n"
            "2. State EVIDENCE:, SOURCE:, and TYPE: once per group. Then one ENTITY: line per entity.\n"
            "3. Use compact entity format: ENTITY: name | CLAIM: value\n"
            "4. CLAIM is what the evidence says about this entity (qualitative change, quantitative measurement, or state/role). Omit CLAIM entirely when the evidence simply mentions the entity.\n"
            "5. Copy evidence VERBATIM from the chunks above — do not summarize.\n"
            "6. NO markdown, NO JSON, NO bullet points.\n"
            "7. Separate groups with one blank line. No introduction, no summary."
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
    #  Batched Entity Extraction (Phase 10.5 — avoids prompt‑size hangs)
    # ------------------------------------------------------------------
    def extract_entities_batched(
        self,
        chunks: List[Dict[str, Any]],
        categories: Dict[str, Any],
        query: str,
        ner_entities: List[Dict[str, Any]] | None = None,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """Extract entities in batches to keep per‑call prompt sizes manageable.

        Each batch of *batch_size* chunks gets its own ``extract_entities()``
        call.  Results are merged and entity names are deduplicated /
        normalised across batches.

        Args:
            chunks: All retrieved chunks for this paper / query.
            categories: Output of ``discover_categories()``.
            query: The original research question.
            ner_entities: Optional deterministic SciSpaCy NER hints.
            batch_size: Maximum chunks per single LLM extraction call (default 8).

        Returns:
            Merged entity dict (same structure as ``extract_entities()``).
        """
        if not chunks:
            return {}

        all_entities: Dict[str, List[Dict[str, Any]]] = {}
        batches = (len(chunks) + batch_size - 1) // batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(
                "Extraction batch %d/%d (%d chunks)", batch_num, batches, len(batch)
            )
            t0 = time.monotonic()
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(
                        self.extract_entities,
                        batch, categories, query,
                        ner_entities=ner_entities,
                    )
                    batch_entities = future.result(timeout=600)
            except FutureTimeoutError:
                elapsed = time.monotonic() - t0
                logger.warning(
                    "Extraction batch %d/%d timed out after %.0fs (%d chunks)",
                    batch_num, batches, elapsed, len(batch),
                )
                continue
            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.warning(
                    "Extraction batch %d/%d failed after %.0fs (%d chunks): %s",
                    batch_num, batches, elapsed, len(batch), exc,
                )
                continue

            elapsed = time.monotonic() - t0
            entity_count = sum(len(v) for v in batch_entities.values())
            logger.info(
                "Extraction batch %d/%d done: %d chunks → %d entities, %.1fs",
                batch_num, batches, len(batch), entity_count, elapsed,
            )

            for category, entity_list in batch_entities.items():
                all_entities.setdefault(category, []).extend(entity_list)

            # Flush Ollama GPU memory between batches to prevent
            # Metal-backend fragmentation from degrading later batches.
            if i + batch_size < len(chunks):  # only between batches, not after last
                try:
                    from src.ingestion.pre_extractor import PreExtractor  # noqa: F811 — lazy to avoid circular import
                    _reset_mode = os.getenv("EXTRACTION_RESET_MODE", "api").strip().lower()
                    if _reset_mode == "process":
                        PreExtractor._restart_ollama_process(timeout=60.0)
                    else:
                        PreExtractor._reset_ollama(timeout=30.0)
                except Exception:
                    pass

        return self._merge_entity_batches(all_entities)

    @staticmethod
    def _merge_entity_batches(
        entities: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Merge entity batches, normalising names and deduplicating.

        Entities with the same normalised name within a category are
        deduplicated — the entry with the longest evidence text is kept.
        """
        result: Dict[str, List[Dict[str, Any]]] = {}

        for category, entity_list in entities.items():
            seen: Dict[str, int] = {}  # normalised_name → index in result list
            result[category] = []

            for ent in entity_list:
                if not isinstance(ent, dict):
                    continue
                name = str(ent.get("entity", "")).strip()
                if not name:
                    continue

                key = name.lower()
                if key in seen:
                    existing_idx = seen[key]
                    existing = result[category][existing_idx]
                    if len(str(ent.get("evidence", ""))) > len(
                        str(existing.get("evidence", ""))
                    ):
                        result[category][existing_idx] = ent
                else:
                    seen[key] = len(result[category])
                    result[category].append(ent)

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

        logger.debug(
            "LLM call: system=%d chars, user=%d chars, model=%s",
            len(system_prompt), len(user_prompt), self.model,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        config: dict[str, Any] = {}
        callbacks: list[Any] = []
        if self.callback:
            callbacks.append(self.callback)
        try:
            from src.streaming_handler import TokenStreamHandler
            callbacks.append(TokenStreamHandler())
        except ImportError:
            pass
        if callbacks:
            config["callbacks"] = callbacks
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
        """Parse line‑tagged entity output into category → entity‑list dict.

        Supports two formats (backward‑compatible):

        **Grouped format (preferred — saves tokens):**
            EVIDENCE: Polymeric nanocarriers like...
            SOURCE: Chunk 1 | paper_a.pdf
            TYPE: material
            ENTITY: polyethyleneimine | DIRECTION: unchanged
            ENTITY: dendrimers | DIRECTION: unchanged

        **Legacy format (still supported):**
            TYPE: material
            ENTITY: Ti-6Al-4V
            DIRECTION: unchanged
            EVIDENCE: Ti-6Al-4V alloy was used...
            SOURCE: Chunk 5 | paper_a.pdf
        """
        text = scrub_unicode(text)
        if "<think>" in text:
            text = _strip_thinking(text)

        result: Dict[str, List[Dict[str, str]]] = {}
        group: Dict[str, str] = {}  # shared: evidence, source, type
        current: Dict[str, str] = {}  # per-entity: entity, direction, context, conditions
        _last_committed: Dict[str, str] = {}  # detect repetition loops

        def _detect_token_spam(value: str, max_consecutive: int = 3) -> bool:
            """Check if a field value contains consecutive repetition.

            Two-pass detection catches the observed degradation signatures:

            **Word-level** (space‑split):
                ``Energy: Energy: Energy: …``  →  same word ≥ *max_consecutive*

            **Character-level** (hyphen‑split within a single “word”):
                ``e-coli-coli-coli-coli…``  →  same sub‑token ≥ *max_consecutive*

            Neither pass uses raw length — legitimate long evidence is never
            flagged; short spam chains are always caught.
            """
            if not value or len(value) < 15:
                return False
            words = value.split()

            # ── Word-level pass ──────────────────────────────────────────
            if len(words) >= max_consecutive:
                run = 1
                for i in range(1, len(words)):
                    a = words[i].lower().rstrip(":,;.")
                    b = words[i - 1].lower().rstrip(":,;.")
                    if a == b:
                        run += 1
                        if run >= max_consecutive:
                            return True
                    else:
                        run = 1

            # ── Character-level pass (hyphenated spam) ───────────────────
            for token in words:
                if "-" not in token or len(token) <= 20:
                    continue
                parts = [p.strip().lower() for p in token.split("-") if p.strip()]
                if len(parts) < max_consecutive:
                    continue
                run = 1
                for i in range(1, len(parts)):
                    if parts[i] == parts[i - 1]:
                        run += 1
                        if run >= max_consecutive:
                            return True
                    else:
                        run = 1

            return False

        def _commit() -> None:
            nonlocal _last_committed
            if not current.get("entity"):
                return
            ent = {**group, **current}
            for field_name, field_value in ent.items():
                if isinstance(field_value, str) and _detect_token_spam(field_value):
                    raise RuntimeError(
                        "Token-level spam detected in %r field: %r. "
                        "Model is degraded (likely Metal backend KV-cache "
                        "corruption from sustained generation). "
                        "Aborting batch extraction." % (field_name, field_value[:120])
                    )
            if ent == _last_committed:
                raise RuntimeError(
                    "LLM repetition loop detected — identical entity block "
                    "committed twice consecutively.  Model is degraded "
                    "(likely Metal backend fragmentation from sustained load). "
                    "Aborting batch extraction."
                )
            _last_committed = ent.copy()
            cat = ent.pop("type", "unknown")
            if cat:
                result.setdefault(cat, []).append(ent)

        def _parse_entity_pipe(value: str) -> Dict[str, str]:
            """Parse 'name | KEY: val | KEY: val' into an entity dict.

            Maps the legacy ``direction`` key to ``claim`` for compatibility
            with pre-Phase 10.5 extractions still stored on disk.
            """
            d: Dict[str, str] = {}
            parts = [p.strip() for p in value.split("|")]
            d["entity"] = parts[0] if parts else ""
            for part in parts[1:]:
                if ":" in part:
                    k, _, v = part.partition(":")
                    k = k.strip().lower()
                    v = v.strip()
                    if k == "direction":
                        k = "claim"
                    if v:
                        d[k] = v
            return d

        for line in text.strip().split("\n"):
            stripped = line.strip()
            if not stripped:
                _commit()
                current = {}
                continue

            if ":" not in stripped:
                continue

            key, _, value = stripped.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key in ("evidence", "source", "type"):
                group[key] = value
            elif key == "entity" and value and "|" in value:
                _commit()
                current = {}
                current = _parse_entity_pipe(value)
                _commit()
                current = {}
            elif key == "entity":
                _commit()
                current = {}
                if value:
                    current["entity"] = value
            elif value:
                current[key] = value

        _commit()

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
