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
import zlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from pathlib import Path
from typing import Any, Dict, List

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage

from src.cache.llm_cache import get_cache
from src.llm import get_chat_model
from src.streaming_handler import ModelDegradedException
from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _combine_evidence(existing: str, new: str) -> str:
    """Combine two evidence strings, deduplicating exact matches."""
    if not existing:
        return new
    if not new:
        return existing
    existing_set = {e.strip() for e in existing.split(" | ") if e.strip()}
    new_set = {e.strip() for e in new.split(" | ") if e.strip()}
    combined = existing_set | new_set
    return " | ".join(sorted(combined, key=len, reverse=True))


def _union_sources(existing: str, new: str) -> str:
    """Union two SOURCE strings, keeping the paper prefix and deduplicating chunk numbers.

    Input:  ``"Chunk 3,7,12 | paper.pdf"`` + ``"Chunk 5,7,12 | paper.pdf"``
    Output: ``"Chunk 3,5,7,12 | paper.pdf"``
    """
    if not existing:
        return new
    if not new:
        return existing

    def _parse(src: str) -> tuple[set[int], str]:
        numbers: set[int] = set()
        paper = src
        if " | " in src:
            prefix, _, paper = src.partition(" | ")
        else:
            prefix = src
        # Also check for "Chunk " prefix
        prefix_stripped = prefix.replace("Chunk ", "").strip()
        for part in re.split(r"[,;\s]+", prefix_stripped):
            try:
                numbers.add(int(part))
            except (ValueError, TypeError):
                pass
        return numbers, paper.strip()

    nums_a, paper_a = _parse(existing)
    nums_b, paper_b = _parse(new)
    all_nums = sorted(nums_a | nums_b)
    paper = paper_a or paper_b
    num_str = ",".join(str(n) for n in all_nums)
    return f"Chunk {num_str} | {paper}" if paper else f"Chunk {num_str}"


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
        self.num_ctx = num_ctx
        self._llm = get_chat_model(
            model=model,
            temperature=temperature,
            max_retries=0,
            streaming=True,
        )
        self.callback = callback
        self._last_output_tokens = 0

    # ── Token counting ────────────────────────────────────────────────

    _tokenizer: tiktoken.Encoding | None = None

    @classmethod
    def _get_tokenizer(cls) -> tiktoken.Encoding:
        """Lazy-load the tiktoken tokenizer shared by all instances."""
        if cls._tokenizer is None:
            cls._tokenizer = tiktoken.get_encoding("cl100k_base")
        return cls._tokenizer

    @staticmethod
    def _format_chunk_text(i: int, chunk: Dict[str, Any]) -> str:
        """Format a single chunk for prompt injection (used for token counting)."""
        text = chunk.get("text", "")
        clean = " ".join(text.split())
        src = (chunk.get("metadata", {}) or {}).get("source", "?")
        return f"[Chunk {i} | {src}] {clean}"

    # ── Extraction statistics (persisted across papers) ────────────────

    _STATS_DIR = Path("projects/default")
    _STATS_FILE = "extraction_stats.json"
    _BAD_CHUNKS_FILE = "bad_chunks.json"

    @classmethod
    def _load_extraction_stats(cls, model: str) -> Dict[str, Any]:
        """Return {output_ratio, boundary_lower, boundary_upper, total_chunk_tokens, total_output_tokens}.

        *boundary_lower* starts at 2500 — calibrated so the first wave
        produces ~8‑chunk batches (matching the empirically‑safe
        batch_size=8 from prior extraction runs).
        *boundary_upper* starts at 16384 — the configured context window.
        Both self‑calibrate from pass/fail data across all runs.
        """
        path = cls._STATS_DIR / cls._STATS_FILE
        default = {
            "output_ratio": 0.50,
            "boundary_lower": 2500,
            "boundary_upper": 16384,
            "total_chunk_tokens": 0,
            "total_output_tokens": 0,
        }
        if path.exists():
            try:
                data = json.loads(path.read_text())
                entry = data.get(model)
                if entry:
                    for k, v in default.items():
                        entry.setdefault(k, v)
                    return entry
            except (json.JSONDecodeError, OSError):
                pass
        return dict(default)

    @classmethod
    def _save_extraction_stats(cls, model: str, stats: Dict[str, Any]) -> None:
        path = cls._STATS_DIR / cls._STATS_FILE
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        data[model] = stats
        cls._STATS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def _update_output_ratio(
        cls, model: str, chunk_tokens: int, output_tokens: int,
    ) -> float:
        """Weighted update: 80% historical, 20% latest.  Returns new ratio."""
        stats = cls._load_extraction_stats(model)
        stats["total_chunk_tokens"] += chunk_tokens
        stats["total_output_tokens"] += output_tokens
        if output_tokens > 0 and chunk_tokens > 0:
            latest = output_tokens / max(chunk_tokens, 1)
            stats["output_ratio"] = 0.80 * stats["output_ratio"] + 0.20 * latest
        cls._save_extraction_stats(model, stats)
        return stats["output_ratio"]

    @classmethod
    def _update_boundary(
        cls, model: str, actual_total: int, passed: bool,
    ) -> None:
        """Narrow the safe‑context boundary from pass/fail data.

        - PASS: raises ``boundary_lower`` (this total was safe).
        - DEGRADE: lowers ``boundary_upper`` (this total was unsafe).

        The gap between lower and upper shrinks with every extraction,
        converging to the model's true effective‑context limit.
        """
        stats = cls._load_extraction_stats(model)
        if passed:
            stats["boundary_lower"] = max(stats["boundary_lower"], actual_total)
        else:
            stats["boundary_upper"] = min(stats["boundary_upper"], actual_total)
        cls._save_extraction_stats(model, stats)

    @classmethod
    def _load_bad_chunks(cls) -> Dict[str, Dict[str, int]]:
        """Return {pmcid: {chunk_idx: fail_count}}."""
        path = cls._STATS_DIR / cls._BAD_CHUNKS_FILE
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    @classmethod
    def _save_bad_chunks(cls, data: Dict[str, Dict[str, int]]) -> None:
        path = cls._STATS_DIR / cls._BAD_CHUNKS_FILE
        cls._STATS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def _record_bad_chunk(cls, pmcid: str, chunk_idx: int) -> None:
        data = cls._load_bad_chunks()
        key = str(pmcid)
        data.setdefault(key, {})
        idx = str(chunk_idx)
        data[key][idx] = data[key].get(idx, 0) + 1
        cls._save_bad_chunks(data)

    @classmethod
    def _is_bad_chunk(cls, pmcid: str, chunk_idx: int, threshold: int = 3) -> bool:
        data = cls._load_bad_chunks()
        return data.get(str(pmcid), {}).get(str(chunk_idx), 0) >= threshold

    # ── Adaptive batch sizing (output‑ratio driven) ────────────────────

    def _calculate_chunk_budget(self, system_prompt: str) -> int:
        """Calculate tokens available for chunk content per batch.

        Uses the self‑calibrating **boundary** (model's proven safe total
        context) and **output_ratio** (chunk → output token multiplier).
        Both improve with every wave across every extraction.

        .. math::
           budget = (boundary_lower × 0.95 - system - overhead) / (1 + ratio)

        Override with ``EXTRACTION_CHUNK_BUDGET`` env var.
        """
        tokenizer = self._get_tokenizer()
        stats = self._load_extraction_stats(self.model)
        output_ratio = stats["output_ratio"]
        boundary_lower = stats["boundary_lower"]

        system_tokens = len(tokenizer.encode(system_prompt))
        user_overhead = 350

        # Safe total context (5% margin below proven passes)
        safe_total = boundary_lower * 0.95
        available = safe_total - system_tokens - user_overhead
        if available <= 0:
            available = 500

        budget = int(available / (1.0 + output_ratio))

        env_override = os.getenv("EXTRACTION_CHUNK_BUDGET", "").strip()
        if env_override and env_override.isdigit():
            budget = min(budget, int(env_override))

        budget = max(100, budget)
        logger.debug(
            "Chunk budget: %d tokens (boundary_lower=%d, system=%d, "
            "overhead=%d, ratio=%.3f)",
            budget, boundary_lower, system_tokens, user_overhead, output_ratio,
        )
        return budget

    def _pack_chunks_into_batches(
        self, chunks: List[Dict[str, Any]], chunk_budget: int,
    ) -> List[List[Dict[str, Any]]]:
        """Greedy token‑based packing of chunks into batches.

        Each chunk's token count is measured *after* formatting (includes
        the ``[Chunk N | source]`` prefix).  A batch is closed when adding
        the next chunk would exceed *chunk_budget*.  This guarantees no
        single batch overflows the model's prompt budget.
        """
        if not chunks:
            return []
        tokenizer = self._get_tokenizer()
        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_tokens = 0

        for i, chunk in enumerate(chunks):
            formatted = self._format_chunk_text(i, chunk)
            ct = len(tokenizer.encode(formatted))

            if current_tokens + ct > chunk_budget and current:
                batches.append(current)
                current = []
                current_tokens = 0

            current.append(chunk)
            current_tokens += ct

        if current:
            batches.append(current)

        logger.info(
            "Packed %d chunks → %d batches (budget=%d tok/batch)",
            len(chunks), len(batches), chunk_budget,
        )
        return batches

    # ── Worker calculation ────────────────────────────────────────────

    @staticmethod
    def _calculate_max_workers(num_ctx: int, total_batches: int) -> int:
        """Calculate maximum concurrent extraction workers.

        Bounded by three factors:
        1. ``OLLAMA_NUM_PARALLEL`` — Ollama's server-side concurrency limit.
        2. GPU memory — model + KV cache × workers must fit.
        3. ``EXTRACTION_MAX_WORKERS`` env var — manual override.

        KV cache formula: ``2 × layers × kv_heads × head_dim × num_ctx``
        bytes per request (q8_0 = 1 byte/element).
        """
        ollama_limit = int(os.getenv("OLLAMA_NUM_PARALLEL", "4"))

        # Per‑request KV cache from model architecture + num_ctx
        model_name = os.getenv("OLLAMA_SMALL_MODEL", "").lower()

        # KV bytes per context token: 2 × layers × kv_heads × head_dim
        # Architecture estimates for common models:
        if "35b" in model_name or "32b" in model_name:
            kv_bytes_per_token = 2 * 64 * 8 * 128   # ~64 layers, qwen‑scale
            model_gb = 25.0
        elif "26b" in model_name:
            kv_bytes_per_token = 2 * 48 * 8 * 128   # ~48 layers
            model_gb = 17.0
        else:  # 4‑8B models (gemma4:e4b, medgemma:4b)
            kv_bytes_per_token = 2 * 32 * 8 * 128   # ~32 layers
            model_gb = 10.5

        kv_gb_per_request = (kv_bytes_per_token * num_ctx) / (1024**3)

        # Target 80% of 36 GB, capped at 28.8 GB
        available_gb = 28.8 - model_gb
        max_by_memory = max(1, int(available_gb / max(kv_gb_per_request, 0.5)))

        workers = min(ollama_limit, max_by_memory, total_batches)

        override = os.getenv("EXTRACTION_MAX_WORKERS", "").strip()
        if override and override.isdigit():
            workers = min(workers, int(override))

        logger.info(
            "Extraction workers: %d (ollama=%d, mem_ceil=%d, batches=%d, "
            "model=%.1fGB, kv=%.2fGB/req)",
            workers, ollama_limit, max_by_memory, total_batches,
            model_gb, kv_gb_per_request,
        )
        return max(1, workers)

    # ── Single‑shot extraction (for parallel waves) ───────────────────

    def _try_extract_once(
        self,
        chunks: List[Dict[str, Any]],
        categories: Dict[str, Any],
        query: str,
        ner_entities: List[Dict[str, Any]] | None = None,
        output_file=None,
    ):
        """Run extraction once.  Returns (entities, degraded, salvage, output_tokens).

        Never raises — degradation and errors are captured so the wave
        loop can re‑queue the batch and aggregate output statistics.
        """
        try:
            entities = self.extract_entities(
                chunks, categories, query,
                ner_entities=ner_entities,
                output_file=output_file,
            )
            return entities, False, {}, self._last_output_tokens
        except ModelDegradedException as e:
            salvage = e.parsed if hasattr(e, "parsed") and e.parsed else {}
            ot = len(self._get_tokenizer().encode(e.text)) if e.text else 0
            return {}, True, salvage, ot
        except (FutureTimeoutError, Exception) as exc:
            logger.warning("Non‑degradation extraction error: %s", exc)
            return {}, True, {}, 0

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
        output_file=None,
    ) -> Dict[str, Any]:
        """Extract structured entities grouped by the discovered categories.

        The LLM outputs a **line‑tagged** format.  Categories are ordered
        by entity density so complex categories get priority output.
        """
        scrubbed_chunks = [
            {**ch, "text": scrub_unicode(ch["text"])} for ch in chunks
        ]
        chunk_summaries = self._format_chunks_for_prompt(scrubbed_chunks)
        categories_text = self._categories_to_line_tagged_sorted(categories)

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

        system_prompt = self._build_all_entities_prompt()
        user_prompt = self._build_user_prompt(
            chunk_summaries, categories_text, query, ner_hint,
        )

        try:
            raw_output = self._call_llm_with_detection(system_prompt, user_prompt, output_file=output_file)
            self._last_output_tokens = len(self._get_tokenizer().encode(raw_output))
        except ModelDegradedException as e:
            logger.warning(
                "Model degradation during extraction: %s. "
                "Parsing partial output and re‑raising.",
                e,
            )
            raw_output = e.text or ""
            self._last_output_tokens = len(self._get_tokenizer().encode(raw_output))
            result = self._parse_line_tagged(raw_output, context="entity_extraction")
            e.parsed = result
            raise

        result = self._parse_line_tagged(raw_output, context="entity_extraction")

        if not result:
            result = self._parse_markdown_fallback(raw_output)
            if result:
                logger.info("Markdown fallback parsed %d categories", len(result))
            else:
                logger.warning(
                    "Both line‑tagged and markdown parsers failed. "
                    "Raw (first 300 chars): %s",
                    raw_output[:300],
                )

        return result

    # ------------------------------------------------------------------
    #  Pulsed‑Wave Parallel Extraction (self‑calibrating, token‑budgeted)
    # ------------------------------------------------------------------
    def extract_paper_recursive(
        self,
        chunks: List[Dict[str, Any]],
        categories: Dict[str, Any],
        query: str,
        ner_entities: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Extract entities with self‑calibrating token‑budgeted batching.

        Architecture:
        1. Load per‑model output ratio from ``extraction_stats.json``
           (aggregated across all papers).  Compute a per‑batch chunk
           budget that accounts for *estimated* output tokens.
        2. Pack chunks greedily by actual tiktoken count.  Known‑bad
           chunks (failed ≥3 times before) are pre‑emptively isolated
           into single‑chunk batches.
        3. Partition batches into **waves** — GPU restart, then all
           batches in the wave run in parallel.
        4. After each wave, update the output ratio from actual output
           token counts.  The next wave uses the updated ratio.
        5. Degraded batches split in half, re‑queued (priority: smaller
           sub‑batches first).  Base‑case chunks recorded in
           ``bad_chunks.json`` for future pre‑emption.
        6. Stats persisted to ``extraction_stats.json`` after the paper.
        """
        if not chunks:
            return {}

        # ── 0. Load persistent state ───────────────────────────────────
        stats = self._load_extraction_stats(self.model)
        output_ratio = stats["output_ratio"]

        bad_chunks = self._load_bad_chunks()
        pmcid = ""
        if chunks:
            meta = (chunks[0].get("metadata", {}) or {}).get("pmcid", "")
            pmcid = str(meta) if meta else ""

        # ── 1. Budget ─────────────────────────────────────────────────
        system_prompt = self._build_all_entities_prompt()
        system_tokens = len(self._get_tokenizer().encode(system_prompt))
        chunk_budget = self._calculate_chunk_budget(system_prompt)

        # ── 2. Pack (pre‑emptively isolate known‑bad chunks) ──────────
        normal = []
        singletons = []
        for i, ch in enumerate(chunks):
            ch_idx = (ch.get("metadata", {}) or {}).get("chunk_index", i)
            if pmcid and self._is_bad_chunk(pmcid, ch_idx):
                singletons.append(ch)
            else:
                normal.append(ch)

        batches = self._pack_chunks_into_batches(normal, chunk_budget)
        # Known‑bad chunks run alone — low risk, fast to complete
        for ch in singletons:
            batches.insert(0, [ch])  # front of queue — finish fast
        logger.info(
            "Batches: %d normal + %d isolated (bad chunks)",
            len(batches) - len(singletons), len(singletons),
        )

        # ── 3. Workers ────────────────────────────────────────────────
        max_workers = self._calculate_max_workers(self.num_ctx, len(batches))

        # ── 4. Pulsed‑wave execution ──────────────────────────────────
        all_entities: Dict[str, List[Dict[str, Any]]] = {}
        queue: List[Any] = [(b, 0) for b in batches]  # (chunks_list, depth)
        max_depth = 12
        failed = 0
        wave_total_chunk_tokens = 0
        wave_total_output_tokens = 0
        _LOGS_DIR = Path("logs/extraction")
        wave_num = 0

        while queue:
            wave_num += 1
            wave_ok = 0
            wave_degraded = 0

            # ── GPU restart at wave start ─────────────────────────
            try:
                from src.ingestion.pre_extractor import PreExtractor
                PreExtractor._restart_ollama_process(timeout=60.0)
            except Exception:
                pass

            # Priority: smaller sub‑batches completed first
            queue.sort(key=lambda item: len(item[0]))

            wave_size = min(max_workers, len(queue))
            wave = queue[:wave_size]
            queue = queue[wave_size:]

            # ── Set up per‑worker log files ───────────────────────
            _LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_paths: List[Path] = []
            for w, (chunks_list, depth) in enumerate(wave):
                n = len(chunks_list)
                first_idx = (chunks_list[0].get("metadata", {}) or {}).get(
                    "chunk_index", "?")
                fname = (
                    f"wave_{wave_num:03d}"
                    f"_chunk-{first_idx}"
                    f"_{n}chunks.txt"
                )
                log_paths.append(_LOGS_DIR / fname)

            logger.info(
                "Wave %d: %d workers, %d queued (workers=%d, ratio=%.3f)",
                wave_num, wave_size, len(queue), max_workers, output_ratio,
            )
            if wave_size > 1:
                logger.info(
                    "Worker log files in %s/\n  %s",
                    _LOGS_DIR,
                    "\n  ".join(
                        f"tail -f {p}" for p in log_paths
                    ),
                )

            # ── Open log files (one per worker) ───────────────────
            worker_files: list = []
            for lp in log_paths:
                try:
                    worker_files.append(open(str(lp), "w"))
                except OSError:
                    worker_files.append(None)

            try:
                with ThreadPoolExecutor(max_workers=wave_size) as executor:
                    futures = {}
                    for idx, (chunks_list, depth) in enumerate(wave):
                        out_f = worker_files[idx]
                        f = executor.submit(
                            self._try_extract_once,
                            chunks_list, categories, query, ner_entities,
                            output_file=out_f,
                        )
                        futures[f] = (chunks_list, depth)

                    for future in as_completed(futures):
                        chunks_list, depth = futures[future]
                        entities, degraded, salvage, output_tokens = future.result()
                        n = len(chunks_list)

                        # Accumulate output token stats for this wave
                        ct = sum(
                            len(self._get_tokenizer().encode(
                                self._format_chunk_text(j, ch)))
                            for j, ch in enumerate(chunks_list)
                        )
                        wave_total_chunk_tokens += ct
                        wave_total_output_tokens += output_tokens

                        if degraded:
                            wave_degraded += 1
                            # ── Collect salvage ───────────────────────
                            for cat, ent_list in salvage.items():
                                all_entities.setdefault(cat, []).extend(ent_list)

                            if depth >= max_depth or n <= 1:
                                failed += 1
                                if n >= 1 and pmcid:
                                    ch_idx = (chunks_list[0].get("metadata", {}) or {}).get("chunk_index", 0)
                                    self._record_bad_chunk(pmcid, ch_idx)
                                if n >= 1:
                                    self._save_failed_chunk(chunks_list[0])
                                logger.warning(
                                    "Base case (depth %d, %d chunks) — "
                                    "saved to failed_chunks/", depth, n,
                                )
                                # Base‑case degradation is a data problem,
                                # not a context‑window problem — skip boundary update.
                                continue

                            # ── Non‑base degradation: update boundary ──
                            actual_total = system_tokens + 350 + ct + output_tokens
                            self._update_boundary(
                                self.model, actual_total, passed=False,
                            )

                            # ── Split and re‑queue ────────────────────
                            mid = n // 2
                            if mid == 0:
                                mid = 1
                            left = chunks_list[:mid]
                            right = chunks_list[mid:]
                            logger.info(
                                "Degraded (depth %d, %d chunks) → "
                                "split [%d, %d] for next wave",
                                depth, n, len(left), len(right),
                            )
                            for sub in sorted([left, right], key=len):
                                queue.append((sub, depth + 1))
                        else:
                            wave_ok += 1
                            for cat, ent_list in entities.items():
                                all_entities.setdefault(cat, []).extend(ent_list)
                            logger.info(
                                "OK (depth %d, %d chunks) → %d entities",
                                depth, n, sum(len(v) for v in entities.values()),
                            )

                            # ── Pass: update boundary ─────────────────
                            actual_total = system_tokens + 350 + ct + output_tokens
                            self._update_boundary(
                                self.model, actual_total, passed=True,
                            )

            finally:
                # Close worker log files
                for fh in worker_files:
                    try:
                        if fh:
                            fh.close()
                    except Exception:
                        pass

            # ── Wave summary ────────────────────────────────────
            logger.info(
                "Wave %d: %d/%d passed, %d degraded → %d queued",
                wave_num, wave_ok, wave_size, wave_degraded, len(queue),
            )

            # ── Per‑wave ratio update ────────────────────────────
            if wave_total_chunk_tokens > 0 and wave_total_output_tokens > 0:
                output_ratio = self._update_output_ratio(
                    self.model, wave_total_chunk_tokens,
                    wave_total_output_tokens,
                )

            # ── Recompute budget for next wave (boundary may have moved) ──
            chunk_budget = self._calculate_chunk_budget(system_prompt)

        total = sum(len(v) for v in all_entities.values())
        logger.info(
            "Extraction done: %d entities, %d failed, %d categories "
            "(final ratio=%.3f)",
            total, failed, len(all_entities), output_ratio,
        )
        return self._merge_entity_batches(all_entities)

    def _save_failed_chunk(self, chunk: Dict[str, Any] | None) -> None:
        """Save a chunk that failed at the base case to disk for review."""
        if not chunk:
            return
        chunk_dir = Path("projects/default/failed_chunks")
        chunk_dir.mkdir(parents=True, exist_ok=True)
        meta = chunk.get("metadata", {}) or {}
        chunk_id = meta.get("pmcid", "unknown")
        chunk_idx = meta.get("chunk_index", "?")
        ts = int(time.monotonic())
        path = chunk_dir / f"{chunk_id}_chunk_{chunk_idx}_{ts}.txt"
        try:
            path.write_text(str(chunk.get("text", "") or "")[:10000])
            logger.info("Failed chunk saved: %s", path)
        except OSError as exc:
            logger.warning("Could not save failed chunk: %s", exc)

    @staticmethod
    def _merge_entity_batches(
        entities: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Merge entity batches with (name, claim) dedup and evidence union.

        Deduplication rules:
        - Same (name, claim) → combine evidence sentences (preserves full context)
        - Same name, different claims → keep both (different facts)
        - Same name, no claim vs has claim → merge evidence into claimed entry
        - Same name, no claim, no claimed variant → keep the no‑claim entry

        Chunk sources are unioned so every supporting chunk is traceable.
        """
        result: Dict[str, List[Dict[str, Any]]] = {}
        index: Dict[str, Dict[str, int]] = {}  # name → claim → idx

        for category, entity_list in entities.items():
            result.setdefault(category, [])
            index.setdefault(category, {})

            for ent in entity_list:
                if not isinstance(ent, dict):
                    continue
                name = str(ent.get("entity", "")).strip()
                if not name:
                    continue
                claim = str(ent.get("claim", "")).strip()

                name_key = name.lower()
                cat_idx = index[category]

                if name_key in cat_idx:
                    existing_idx = cat_idx[name_key]
                    existing = result[category][existing_idx]
                    existing_claim = str(existing.get("claim", "")).strip()

                    if claim and claim == existing_claim:
                        # Same claim — combine evidence and union sources
                        existing["evidence"] = _combine_evidence(
                            str(existing.get("evidence", "")),
                            str(ent.get("evidence", "")),
                        )
                        existing["source"] = _union_sources(
                            str(existing.get("source", "")),
                            str(ent.get("source", "")),
                        )
                    elif claim and not existing_claim:
                        # New claim beats no-claim — replace
                        old_evidence = str(existing.get("evidence", ""))
                        new_evidence = str(ent.get("evidence", ""))
                        result[category][existing_idx] = {
                            **ent,
                            "evidence": _combine_evidence(old_evidence, new_evidence),
                            "source": _union_sources(
                                str(existing.get("source", "")),
                                str(ent.get("source", "")),
                            ),
                        }
                    elif claim and existing_claim and claim != existing_claim:
                        # Different claims — keep both as separate entries
                        cat_idx[name_key] = len(result[category])
                        result[category].append(ent)
                    elif not claim and existing_claim:
                        # No-claim variant → merge evidence into claimed entry
                        existing["evidence"] = _combine_evidence(
                            str(existing.get("evidence", "")),
                            str(ent.get("evidence", "")),
                        )
                        existing["source"] = _union_sources(
                            str(existing.get("source", "")),
                            str(ent.get("source", "")),
                        )
                    else:
                        # Both no-claim — keep longest evidence
                        if len(str(ent.get("evidence", ""))) > len(
                            str(existing.get("evidence", ""))
                        ):
                            result[category][existing_idx] = ent
                else:
                    cat_idx[name_key] = len(result[category])
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

    # ── Prompt builders ──────────────────────────────────────────────

    @staticmethod
    def _build_all_entities_prompt() -> str:
        """System prompt for Pass 2 — extract every entity from every chunk."""
        return (
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

    @staticmethod
    def _build_user_prompt(
        chunk_summaries: str,
        categories_text: str,
        query: str,
        ner_hint: str = "",
        extra_instructions: str = "",
    ) -> str:
        """Build the user prompt for entity extraction.

        Args:
            chunk_summaries: Formatted chunk text.
            categories_text: Line‑tagged categories (sorted by density).
            query: The research question.
            ner_hint: Optional NER entity hints.
            extra_instructions: Additional mode‑specific instructions.
        """
        parts = [
            f"Research Query: {query}\n",
            f"Discovered Categories:\n{categories_text}\n",
            ner_hint,
            f"Retrieved Chunks (format: [Chunk N | source] text):\n{chunk_summaries}\n",
        ]
        if extra_instructions:
            parts.append(extra_instructions + "\n")
        parts.append(
            "INSTRUCTIONS:\n"
            "1. Group entities that share the SAME evidence sentence together.\n"
            "2. State EVIDENCE:, SOURCE:, and TYPE: once per group. Then one ENTITY: line per entity.\n"
            "3. Use compact entity format: ENTITY: name | CLAIM: value\n"
            "4. CLAIM is what the evidence says about this entity (qualitative change, quantitative measurement, or state/role). Omit CLAIM entirely when the evidence simply mentions the entity.\n"
            "5. Copy evidence VERBATIM from the chunks above — do not summarize.\n"
            "6. NO markdown, NO JSON, NO bullet points.\n"
            "7. Separate groups with one blank line. No introduction, no summary."
        )
        return "".join(parts)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return the full response text.

        Responses are cached by (system_prompt, user_prompt) hash with 24h TTL
        since temperature=0 makes outputs deterministic.

        Raises :class:`ModelDegradedException` when the streaming handler
        detects model degradation (KV‑cache corruption, repetition spam,
        format loss).  The exception carries the captured text so callers
        can salvage partial entities and retry with a fresh GPU.
        """
        from src.streaming_handler import ModelDegradedException, TokenStreamHandler

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
        handler = TokenStreamHandler()
        callbacks.append(handler)
        if callbacks:
            config["callbacks"] = callbacks
        response = self._llm.invoke(messages, config=config)
        result = (response.content or "").strip()

        if handler.degraded:
            raise ModelDegradedException(
                handler.degraded_reason,
                text=result,
            )

        cache.set(system_prompt, user_prompt, result, model=self.model)
        return result

    def _call_llm_with_detection(self, system_prompt: str, user_prompt: str, output_file=None) -> str:
        """Extraction‑only LLM call with real‑time degradation detection.

        Uses ``self._llm.stream()`` instead of ``invoke()`` so the loop
        can **break immediately** when the :class:`TokenStreamHandler`
        detects spam, repetition, or format loss.  The ``finally`` block
        calls ``stream.close()`` to send ``GeneratorExit``, forcing
        LangChain's underlying httpx connection to drop — Ollama stops
        generating in milliseconds rather than waiting for ``max_tokens``
        of garbage.

        Raises :class:`ModelDegradedException` on detection so callers can
        salvage partial text and retry with a fresh GPU.

        If *output_file* is provided, live tokens are written there instead
        of stdout (avoids jumbled output in parallel mode).
        """
        from src.streaming_handler import TokenStreamHandler

        cache = get_cache()
        cached = cache.get(system_prompt, user_prompt, model=self.model)
        if cached is not None:
            return cached

        logger.debug(
            "LLM call (streaming): system=%d chars, user=%d chars, model=%s",
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
        handler = TokenStreamHandler(output_file=output_file)
        callbacks.append(handler)
        if callbacks:
            config["callbacks"] = callbacks

        all_tokens = ""
        stream = self._llm.stream(messages, config=config)
        try:
            for chunk in stream:
                token = getattr(chunk, "content", "") or ""
                all_tokens += token
                if handler.degraded:
                    logger.warning(
                        "Model degradation detected mid‑stream — aborting generation. "
                        "Reason: %s",
                        handler.degraded_reason,
                    )
                    break
        except ModelDegradedException:
            raise
        except Exception as e:
            if handler.degraded:
                raise ModelDegradedException(
                    handler.degraded_reason,
                    text=all_tokens.strip(),
                ) from e
            raise
        finally:
            try:
                stream.close()  # GeneratorExit → httpx disconnect → Ollama stops
            except Exception:
                pass

        if handler.degraded:
            raise ModelDegradedException(
                handler.degraded_reason,
                text=all_tokens.strip(),
            )

        result = all_tokens.strip()
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
    def _categories_to_line_tagged_sorted(categories: Dict[str, Any]) -> str:
        """Format categories as line‑tagged text, ordered by entity density.

        Categories with the most examples_found come first — the LLM
        naturally extracts from categories in prompt order, so complex
        categories get priority output before simpler ones.
        """
        lines: List[str] = []
        discovered = categories.get("discovered_categories", [])
        if isinstance(discovered, list):
            sorted_cats = sorted(
                discovered,
                key=lambda c: len(c.get("examples_found", [])) if isinstance(c, dict) else 0,
                reverse=True,
            )
            for cat in sorted_cats:
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

        junk_streak = 0
        _MAX_JUNK_LINES = 20

        try:
            for line in text.strip().split("\n"):
                stripped = line.strip()
                if not stripped:
                    _commit()
                    current = {}
                    continue

                if ":" not in stripped:
                    # Check raw line for token spam (e-coli-coli..., etc.)
                    if _detect_token_spam(stripped):
                        raise RuntimeError(
                            "Token spam detected on raw line (no ':' format): %r. "
                            "Model degraded — Metal backend KV‑cache corruption."
                            % stripped[:120]
                        )
                    junk_streak += 1
                    if junk_streak >= _MAX_JUNK_LINES:
                        raise RuntimeError(
                            "Model degraded: %d consecutive junk lines "
                            "(no ':' format separator). Model has lost "
                            "line‑tagged output format — aborting batch." % junk_streak
                        )
                    continue

                # Line has ':' — model is still producing formatted output
                junk_streak = 0

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
        except RuntimeError:
            if result:
                logger.warning(
                    "Batch extraction degraded — returning %d partial entities "
                    "across %d categories.",
                    sum(len(v) for v in result.values()), len(result),
                )
            else:
                raise

        if not result and len(text) > 500:
            try:
                compressed = zlib.compress(text.encode("utf-8"))
                ratio = len(text) / max(len(compressed), 1)
                if ratio >= 12.0:
                    raise ModelDegradedException(
                        f"Compression ratio {ratio:.1f}:1 indicates repetitive "
                        f"degradation (text={len(text)} chars, no entities parsed)",
                        text=text,
                    )
            except ModelDegradedException:
                raise
            except Exception:
                pass

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
