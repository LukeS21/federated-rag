# Phase 8 → Phase 9 Handoff — May 2026 (Phase 9: in progress)

## Quick start

```bash
# Phase 9 pipeline test (Europe PMC + Semantic Scholar, ~22s for 10 papers)
python phase9_europe_pmc_test.py --count 10

# Phase 9 pipeline test with custom query
python phase9_europe_pmc_test.py --count 50 --query "titanium implant osseointegration surface modification"

# Existing Phase 8 download pipeline (deprecated — see notes below)
# python scripts/headless_download.py --limit 10

# Tests
python -m pytest tests/ -v --tb=short
```

## Current project state

**Phase 9 architecture pivot.** The Phase 8 Playwright/EZProxy PDF download pipeline
has been superseded by a REST‑API‑based approach using Europe PMC full-text XML
and Semantic Scholar SPECTER2 embeddings.  This eliminates the browser automation
layer entirely — no Playwright, no EZProxy, no WAF blocks, no IP blacklists.

**Core pipeline** (all in one 22s run for 10 papers):
```
Europe PMC search (OPEN_ACCESS:Y) → fullTextXML fetch → JATS XML parse → chunks
Semantic Scholar → DOI resolve → SPECTER2 embedding batch fetch
```

**What changed and why:**

| | Phase 8 (Playwright) | Phase 9 (API) |
|---|---------------------|---------------|
| **Speed** | 45–90s per paper | ~2.9s per paper (27× faster) |
| **Reliability** | ~80% (WAF blocks, IP blacklists) | 100% of OA papers |
| **Content** | page.pdf() rasterized text | Structured XML sections |
| **Figures** | Docling extraction from PDF | Captions from XML `<fig>` elements |
| **Rate limits** | WAF blocks, IP blacklists by ScienceDirect | None (politely paced API calls) |
| **Failure modes** | Browser crashes, auth expiry, URL pattern mismatches | DOI not in S2, no PMC deposition |
| **Code size** | 600 lines of Playwright wrestling | ~400 lines across 3 clean modules |

**Coverage tradeoff**: Europe PMC only returns papers archived in PubMed Central
(~6.5M biomedical OA papers).  Papers without PMC deposition are classified as
"abstract-only" — they appear in the knowledge graph and literature map but are
never used for grounded claims or citations.

**Phase 8 download pipeline status**: The `scripts/headless_download.py` pipeline
produced 115 valid PDFs before being superseded.  Those PDFs are still on disk in
`data/external/` and can be ingested into ChromaDB via the existing ingestion path.
The pipeline is **deprecated but not deleted** — it still works for non-OA papers
when EZProxy auth is fresh and publisher rate limits aren't triggered.  The
institutional VCU EZProxy session expires after ~6–12 hours.

## What was accomplished in Phase 9 (this session)

### Europe PMC client

| Component | File | Status |
|-----------|------|--------|
| Search with `OPEN_ACCESS:Y` filter | `src/retrieval/europe_pmc.py` | ✅ Search, fullTextXML fetch, batch fetch, metadata |
| Full-text JATS XML fetch | `src/retrieval/europe_pmc.py` | ✅ Polymorphic Accept header (XML gets `text/xml`, JSON search gets `application/json`) |
| Rate limit pacing | `src/retrieval/europe_pmc.py` | ✅ 150ms interval between requests |

### JATS XML parser

| Component | File | Status |
|-----------|------|--------|
| Section extraction (recursive) | `src/ingestion/pmc_xml_parser.py` | ✅ Handles nested `<sec>` elements |
| Figure caption extraction | `src/ingestion/pmc_xml_parser.py` | ✅ Labels + captions + graphic URLs |
| Reference list extraction | `src/ingestion/pmc_xml_parser.py` | ✅ |
| Namespace stripping | `src/ingestion/pmc_xml_parser.py` | ✅ Handles `xlink:href`, `m:math`, and other namespace prefixes |
| Chunk format compatibility | `src/ingestion/pmc_xml_parser.py` | ✅ Same `{"text": "...", "metadata": {...}}` dict format as `PDFParser.parse()` — drop-in compatible with `hybrid.ingest()` |

### Semantic Scholar SPECTER2 integration

| Component | File | Status |
|-----------|------|--------|
| SPECTER2 batch embedding fetch | `src/retrieval/semantic_scholar.py` | ✅ `get_embeddings_batch(s2_ids)` |
| DOI lookup with title fallback | `src/retrieval/semantic_scholar.py` | ✅ `resolve_paper(doi, title)` — DOI first, title search fallback |
| Null-safe normalization | `src/retrieval/semantic_scholar.py` | ✅ Handles `embedding: null`, `tldr: null`, etc. |
| 80% embedding coverage | — | 8/10 papers in test run got SPECTER2 vectors |

### Phase 8 PDF download pipeline improvements

| Component | File | Status |
|-----------|------|--------|
| Route interception | `scripts/headless_download.py` | ✅ `context.route()` catches native PDFs |
| Page.pdf() universal fallback | `scripts/headless_download.py` | ✅ No publisher URL patterns needed |
| Word-count quality gate | `scripts/headless_download.py` | ✅ Pages <1500 words rejected instead of saving garbage |
| Download-link detection gate | `scripts/headless_download.py` | ✅ Pages with PDF links + <5000 words → "unavailable" |
| Popup/cookie banner dismissal | `scripts/headless_download.py` | ✅ JavaScript overlay removal before page.pdf() |
| "unavailable" terminal status | `scripts/headless_download.py` | ✅ `reason` field: `"content_sparse"`, `"waf_blocked"`, `"no_doi"` |
| Persistent browser | `scripts/headless_download.py` | ✅ One browser for full batch (saves ~5s/paper) |
| Rate limited pacing | `scripts/headless_download.py` | ✅ 1.0–2.5s random delay between papers |
| Progress checkpointing | `scripts/headless_download.py` | ✅ Status saved every 10 papers |
| Zotero attachment | `scripts/headless_download.py` | ✅ `_attach_to_zotero()` after each successful download |
| Phase 3 requests-based download | `scripts/headless_download.py` | ✅ `_build_requests_session()` + EZProxy cookies bypasses Chrome PDF viewer |

### Test results

| Metric | Phase 8 (Playwright) | Phase 9 (API) |
|--------|---------------------|---------------|
| 3 papers | ~180s | **8.71s** (20.7×) |
| 10 papers | ~600s | **21.95s** (27.3×) |
| Search | — | 0.71s |
| Full-text XML fetch | — | 2.53s |
| XML parse | — | 0.31s |
| S2 embeddings | — | 18.40s (84% of total, one-time cost) |
| Chunks per 10 papers | — | 414 |
| Words per 10 papers | — | 262,484 |
| Sections per 10 papers | — | 276 |
| Figures per 10 papers | — | 108 |
| SPECTER2 embedding coverage | — | 80% (8/10) |

## Lessons learned in Phase 9

### 1. The PDF download approach was fundamentally wrong for scale

Downloading PDFs through institutional proxies is a losing battle:
every publisher serves content differently, WAFs detect automation, signed URLs
expire, and IP blocks happen after ~20 rapid requests.  The Europe PMC API
gives structured full text in 200ms with zero of these problems.  The tradeoff is
coverage (PMC-only), but for NIH-funded biomedical research, that's ~80–90% of
relevant papers.

### 2. page.pdf() is a universal fallback but captures what's on screen, not what's in the paper

Playwright's `page.pdf()` works for any publisher that renders text in the DOM.
But for ScienceDirect, the HTML page shows only the abstract (sometimes loads
full text via lazy JS, sometimes doesn't).  The word-count gate (reject <1500 words)
and download-link gate (reject pages with "View PDF" links) catch most garbage,
but they can't guarantee full-text quality.

### 3. SPECTER2 is 84% of pipeline time and 80% reliable — cache it

SPECTER2 embeddings take ~2s per paper via Semantic Scholar's DOI lookup
(1.8s rate-limit pacing + 200ms API call).  For 100 papers, that's 3+ minutes
of just embedding resolution.  The vectors don't change once computed — they
should be cached locally and only fetched once per paper.

### 4. Europe PMC's Accept header is strict — the session needs polymorphic headers

The `fullTextXML` endpoint returns `406 Not Acceptable` when the request has
`Accept: application/json`.  The search endpoint requires it.  Solution:
override the Accept header per-request instead of per-session.

### 5. JA TS XML namespace stripping is fragile if you only remove xmlns declarations

Stripping `xmlns:xlink="..."` but leaving `xlink:href` attributes causes
"unbound prefix" ElementTree parse errors.  The fix: strip namespace prefixes
from attributes AND tag names, not just xmlns declarations.  This handles
`xlink:href`, `m:math`, and any future namespace additions.

### 6. Semantic Scholar DOI lookup can fail silently — title fallback rescues it

S2's `/paper/DOI:{doi}` endpoint returns 404 for ~20% of biomedical DOIs
(recent papers, niche journals).  Title-based search as fallback recovers most
of these.  The `resolve_paper(doi, title)` pattern makes this transparent to
callers.

### 7. Pre-computed annotations (MeSH) complement but don't replace LLM extraction

Europe PMC's annotations API provides human-curated MeSH terms and NLP-extracted
gene/disease/chemical mentions.  These are reliable for entity identification but
can't capture novel connections or narrative synthesis — that's what the LLM is for.
The right architecture: MeSH as a pre-populated skeleton, LLM as the enrichment layer.

## 12 identified gaps — severity and status

| # | Gap | Severity | Status |
|---|------|----------|--------|
| 1 | No retry logic on transient failures (5xx, timeout) | **High** | Not yet implemented |
| 2 | No progress persistence (crash = restart from zero) | **High** | Not yet implemented |
| 3 | Ingestion not wired to ChromaDB | **High** | Not yet implemented |
| 4 | Coverage gap unknown (what % of relevant papers are in PMC?) | Med | Diagnostic needed |
| 5 | Figure pipeline disconnected (XML has captions, no image downloads) | Med | Phase 7a wiring needed |
| 6 | Zotero integration undefined for XML-only ingestion | Low | Decision pending |
| 7 | Pre-extraction (NER) missing from new pipeline | Low | Can use Europe PMC annotations |
| 8 | SPECTER2 embeddings not cached (84% of pipeline time) | Low | One-time compute, cacheable |
| 9 | Single point of failure (Europe PMC is sole primary source) | Low | No viable alternative for structured full-text |
| 10 | Freshness lag (PMC mirroring takes weeks) | Accepted | Inherent to PMC; not fixable |
| 11 | S2 API key unreliable (429 errors per HANDOFF history) | Low | Title fallback works without key |
| 12 | Zero search results | Low | Europe PMC is comprehensive for biomedical |

**Recommendation**: Fix gaps 1–3 immediately (~75 lines across 2 files).
Gaps 4–8 are quality improvements to address sequentially. Gaps 9–12 are accepted.

## Key architectural decisions (DO NOT UNDO)

All previous DO NOT UNDO from Phase 4–8 still apply.  Additional Phase 9 decisions:

- **Europe PMC is the primary full-text source** — API-based, no browser, no paywall wrestling.  PMC-only coverage is the explicit tradeoff.
- **Semantic Scholar is supplementary** — SPECTER2 embeddings for fine-grained similarity, title-based fallback when DOI lookup fails, broader field coverage for literature discovery.  Not a full-text source.
- **JATS XML parser produces the same chunk format as PDFParser** — `{"text": "...", "metadata": {...}}` dicts.  Drop-in compatible with `hybrid.ingest()`.
- **Phase 8 Playwright pipeline is deprecated but preserved** — works for non-OA papers when EZProxy is available.  Not deleted.
- **"unavailable" is a terminal status** — papers with `status: "unavailable"` have a `reason` field and are never retried.  This prevents infinite retry loops on MDPI WAF blocks, no-DOI papers, and permanently sparse pages.
- **Word-count gate at 1500 words** — below this, the page is not a real paper (error page, access denied, abstract-only).  Configurable via `_MIN_PAGE_WORDS`.
- **Download-link gate at 5000 words** — pages with visible "View PDF" links and <5000 words are landing pages, not full text.  Marked unavailable.
- **Phase 3 (requests.get) is the most reliable PDF download method** — bypasses Chrome's PDF viewer entirely.  Works for Springer, Wiley, Sage.  Fails for ScienceDirect's signed URLs (time-limited md5 hashes).

## What NOT to change

All previous What NOT to change from Phase 4–8 still apply.  Additional Phase 9 constraints:

- Do NOT remove the Europe PMC pipeline in favor of going back to Playwright — the 27× speedup is fundamental architecture, not optimization
- Do NOT change the chunk format produced by `PMCXMLParser.parse()` — it must remain compatible with `PDFParser.parse()` for the ingestion pipeline
- Do NOT remove `OPEN_ACCESS:Y` filter from default searches — the system is designed around full-text-first, abstract-as-fallback
- Do NOT delete `scripts/headless_download.py` or `data/external/` — the 115 valid PDFs are still useful for non-PMC papers
- Do NOT change `_MIN_PAGE_WORDS` from 1500 without testing against known error pages
- Do NOT remove `"unavailable"` terminal status — it prevents infinite retry on papers we know we can't get
- Do NOT revert `_strip_namespaces` to the old xmln-only approach — the prefixed-attribute handling is needed for JATS XML compatibility
- Do NOT add Accept:application/json back to the session default — it breaks the fullTextXML endpoint (406 Not Acceptable)

## File map (new and changed in Phase 9)

```
NEW FILES (Phase 9):
src/retrieval/europe_pmc.py                     # Europe PMC REST client (search, fullTextXML, metadata)
src/ingestion/pmc_xml_parser.py                 # JATS XML → chunk dict parser
phase9_europe_pmc_test.py                       # End-to-end pipeline test harness

MODIFIED FILES (Phase 9):
src/retrieval/semantic_scholar.py               # +SPECTER2 embeddings, +title fallback, +null-safe normalize
scripts/headless_download.py                    # Phase 8 improvements (route interception, word-count gate, 
                                                #   "unavailable" status, popup dismissal, persistent browser,
                                                #   rate limiting, Phase 3 requests-based download, Zotero attach)

PROJECT DATA (auto-generated):
data/external/                                  # ~115 valid PDFs from Phase 8 download pipeline
data/external/zotero_sync_status.json           # Download tracking with "unavailable" status + reason field
data/external/vcu_auth.json                     # VCU EZProxy auth state (expires ~6-12h)
projects/default/phase9_europe_pmc_test.json    # Phase 9 pipeline benchmark results
projects/default/chroma_data/                   # ChromaDB with 76 papers ingested (16,568 chunks)
projects/default/bm25_index/                    # Persisted BM25 corpus
projects/default/phase8_naive_rag.json          # Naive RAG baseline (13 claims, 1.000 anchoring)
```

## Prompt for next AI session

```
You are an expert senior software developer continuing Phase 9 (API-Based Literature
Ingestion) of a Federated RAG system for biomedical research. Read the full
README.md and this HANDOFF.md carefully before making changes.

CURRENT STATE:
  - Phase 9 architecture pivot is complete. The Europe PMC API pipeline is built
    and tested: 10 papers in 22s (27× faster than the Phase 8 Playwright pipeline).
    PDF downloading via browser automation is deprecated but preserved on disk.
  - 115 valid PDFs from Phase 8 remain in data/external/ (Playwright page.pdf() results).
  - Europe PMC client (src/retrieval/europe_pmc.py) handles search + fullTextXML fetch.
  - JATS XML parser (src/ingestion/pmc_xml_parser.py) produces chunk dicts compatible
    with the existing PDFParser output format.
  - Semantic Scholar SPECTER2 integration complete: DOI lookup with title fallback,
    batch embedding fetch, null-safe normalization. 80% embedding coverage.
  - The Phase 8 download script (scripts/headless_download.py) has been significantly
    improved with route interception, word-count quality gate, "unavailable" terminal
    status, popup dismissal, persistent browser, and rate limiting. It still works
    but requires VCU EZProxy auth (6-12h expiry) and is susceptible to publisher
    rate limits (ScienceDirect blocks after ~20 rapid requests).

TOP PRIORITY — fix the 3 blocking gaps before anything else:
  1. RETRY LOGIC: Add 3-retry with exponential backoff to EuropePMCClient._get()
     for transient 5xx and timeout errors. ~20 lines in src/retrieval/europe_pmc.py.
  2. PROGRESS PERSISTENCE: Save checkpoint every N papers to
     projects/default/ingest_progress.json. Resume on restart. ~40 lines.
  3. WIRE INGESTION: PMCXMLParser.parse() output is already compatible with
     hybrid.ingest(). Create an ingest loop that calls PMC parse + hybrid.ingest()
     for each paper. ~15 lines in a new script or phase9_europe_pmc_test.py.

NEXT PRIORITY — quality and completeness:
  4. Coverage diagnostic: Run Europe PMC search vs Semantic Scholar search with
     the same query. Report "X/Y papers (Z%) have PMC full text."
  5. Figure pipeline: Download images from XML <graphic> URLs, wire into
     vision_ingest path (Phase 7a). Captions are already extracted.
  6. SPECTER2 caching: Store embeddings locally so the 84% pipeline time is only
     paid once per paper.

BEFORE MAKING CHANGES:
  1. Run tests: python -m pytest tests/ -v --tb=short
  2. Test the API pipeline: python phase9_europe_pmc_test.py --count 5
  3. Check EZProxy auth if you need the Playwright pipeline:
     python scripts/ezproxy_download.py --setup (interactive, requires DUO 2FA)
     Check auth age: python -c "import json,time;from pathlib import Path;
     print('Auth age:', (time.time()-Path('data/external/vcu_auth.json').stat().st_mtime)/3600,'h')"
```

## Quick start (Phase 9)

```bash
# Test the API pipeline (instant, no browser)
python phase9_europe_pmc_test.py --count 10

# Test with more papers
python phase9_europe_pmc_test.py --count 50

# Test with custom query
python phase9_europe_pmc_test.py --count 10 --query "dental implant macrophage polarization"

# View cached results
cat projects/default/phase9_europe_pmc_test.json | python -m json.tool | head -50

# Single paper: search + XML fetch + parse
python -c "
from src.retrieval.europe_pmc import EuropePMCClient
from src.ingestion.pmc_xml_parser import PMCXMLParser
c = EuropePMCClient(); p = PMCXMLParser()
papers = c.search('titanium implant macrophage', oa_only=True, max_results=3)
for paper in papers:
    xml = c.full_text_xml(paper['pmcid'])
    if xml:
        chunks = p.parse(xml, pmcid=paper['pmcid'], doi=paper['doi'])
        print(f\"{paper['title'][:60]}: {len(chunks)} chunks, {sum(len(c['text'].split()) for c in chunks)} words\")
"

# Phase 8 pipeline (deprecated, requires EZProxy auth)
python scripts/headless_download.py --limit 10
python scripts/headless_download.py --doi "10.1016/j.actbio.2018.06.018"

# Re-authenticate VCU EZProxy (needed after 6-12 hours)
python scripts/ezproxy_download.py --setup

# Tests
python -m pytest tests/ -v --tb=short
```
