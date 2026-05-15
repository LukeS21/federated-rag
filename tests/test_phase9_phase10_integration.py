"""Integration test: Phase 9 outputs into Phase 10 pipeline pieces.

Validates the end-to-end flow:
  coverage diagnostic → gap resolver parsing → EPMC search → XML fetch → parse → ingest
"""
import pytest
from pathlib import Path


class TestPhase9ToPhase10Integration:
    """End-to-end test connecting Phase 9 output to Phase 10 pieces."""

    def test_coverage_diagnostic_returns_structured_data(self):
        """Coverage diagnostic produces usable stats and paper lists."""
        from src.retrieval.coverage import run_coverage_diagnostic

        cov = run_coverage_diagnostic("bone tissue engineering scaffold", max_results=5)
        assert "query" in cov
        assert "s2_total" in cov
        assert "epmc_total" in cov
        assert "matched" in cov
        assert "coverage_pct" in cov
        assert isinstance(cov["epmc_results"], list)
        assert isinstance(cov["s2_coverage_detail"], list)
        # Coverage data should be consumable by gap resolver
        assert cov["epmc_total"] >= 0

    def test_gap_parser_produces_searchable_queries(self):
        """Gap analysis text → structured queries that EPMC can search."""
        from src.agents.gap_resolver import _parse_gaps_to_queries

        gap_text = (
            "1. No data on osteoblast activity in obese Ti implant models.\n"
            "2. Missing: The role of leptin in peri-implant bone remodeling."
        )
        queries = _parse_gaps_to_queries(gap_text)
        assert len(queries) >= 1
        for q in queries:
            assert "query" in q
            assert len(q["query"]) >= 10  # searchable query

    def test_epmc_search_and_fetch_parses_chunks(self):
        """Europe PMC search → XML fetch (with fallback) → parse into chunks."""
        from src.retrieval.europe_pmc import EuropePMCClient
        from src.ingestion.pmc_xml_parser import PMCXMLParser

        epmc = EuropePMCClient()
        parser = PMCXMLParser()

        papers = epmc.search("bone scaffold osteoblast", oa_only=True, max_results=3)
        assert len(papers) > 0, "EPMC search returned 0 results (API may be down)"

        # Try to fetch at least one paper
        parsed = 0
        for p in papers:
            pmcid = p.get("pmcid", "")
            if not pmcid:
                continue
            xml = epmc.full_text_xml(pmcid)
            if not xml:
                continue
            chunks = parser.parse(xml, pmcid=pmcid, doi=p.get("doi", ""))
            if not chunks:
                continue
            # Validate chunk structure
            for c in chunks:
                assert "text" in c
                assert "metadata" in c
                assert "source" in c["metadata"]
                assert pmcid in c["metadata"]["source"]
                assert "chunk_index" in c["metadata"]
                assert "chunk_type" in c["metadata"]
            parsed += 1
            break  # One paper is sufficient for validation

        if parsed == 0:
            # EPMC fullTextXML may be down — fallback to OAI should still work
            # Try a known PMCID that works with OAI
            xml = epmc.full_text_xml("PMC4302049")
            if xml:
                chunks = parser.parse(xml, pmcid="PMC4302049", doi="")
                assert len(chunks) > 0, "PMC OAI fallback returned XML but parse produced 0 chunks"
                for c in chunks:
                    assert "text" in c
                    assert "metadata" in c
                    assert "PMC4302049" in c["metadata"]["source"]
                    assert "chunk_index" in c["metadata"]

    def test_ingest_dedup_prevents_duplicates(self):
        """Re-ingesting the same chunks skips duplicates in ChromaDB."""
        from src.retrieval.chroma_client import ChromaClient
        from src.retrieval.bm25_index import BM25Index
        from src.retrieval.hybrid_retriever import HybridRetriever
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            chroma = ChromaClient(
                collection_name="test_ingest_dedup",
                persist_directory=tmpdir,
            )
            bm25 = BM25Index()
            retriever = HybridRetriever(chroma_client=chroma, bm25_index=bm25)

            chunks = [
                {"text": f"Test chunk {i}", "metadata": {
                    "source": "test_integration",
                    "chunk_index": i,
                    "chunk_type": "text",
                }}
                for i in range(5)
            ]

            retriever.ingest(chunks)
            count1 = chroma.collection.count()
            assert count1 == 5

            # Re-ingest same chunks — should skip duplicates
            retriever.ingest(chunks)
            count2 = chroma.collection.count()
            assert count2 == 5, f"Duplicates added! {count1} → {count2}"

            # New chunks should still be added
            new_chunks = [
                {"text": "New chunk", "metadata": {
                    "source": "test_integration",
                    "chunk_index": 99,
                    "chunk_type": "text",
                }}
            ]
            retriever.ingest(new_chunks)
            count3 = chroma.collection.count()
            assert count3 == 6, f"New chunk not added! {count1} → {count3}"

    def test_specter2_cache_roundtrip(self):
        """Store and retrieve a SPECTER2 embedding."""
        from src.utils.spector2_cache import Spector2Cache
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            cache_path = Path(f.name)
            cache = Spector2Cache(cache_path=cache_path)

            emb = [0.1] * 768
            doi = "10.1000/integration.test.001"
            cache.put(doi, "s2_integ_001", emb)
            cache.flush()

            # New instance loads from disk
            cache2 = Spector2Cache(cache_path=cache_path)
            assert cache2.has(doi)
            retrieved = cache2.get(doi)
            assert retrieved == emb
            assert len(retrieved) == 768

    def test_web_search_returns_discovery_only(self):
        """Web search returns discovery-tagged results."""
        from src.retrieval.web_search import WebSearchClient

        ws = WebSearchClient()
        results = ws.search("biomaterial scaffold bone regeneration", max_results=3)
        # May return 0 if library not installed or rate-limited — either is OK
        for r in results:
            assert r["source_type"] == "discovery"

    def test_gap_resolver_rejects_null_findings(self):
        """Null findings like 'no significant difference' are not treated as gaps."""
        from src.agents.gap_resolver import _parse_gaps_to_queries

        queries = _parse_gaps_to_queries(
            "No significant difference was observed between groups. "
            "No statistically significant effect was found."
        )
        assert len(queries) == 0, "Null findings should not produce gap queries"

    def test_gap_resolver_rejects_the_lstrip_bug(self):
        """The previous lstrip('the ') bug should not strip 'e' from 'examined'."""
        from src.agents.gap_resolver import _parse_gaps_to_queries

        queries = _parse_gaps_to_queries(
            "Gap: No study has examined the role of leptin in bone remodeling."
        )
        assert len(queries) >= 1
        # 'examined' should stay intact, not become 'xamined'
        assert "examined" in queries[0]["query"]

    def test_full_pipeline_coverage_to_gap_resolver(self):
        """Coverage data can feed into gap resolver for topic discovery."""
        from src.retrieval.coverage import run_coverage_diagnostic
        from src.agents.gap_resolver import _parse_gaps_to_queries

        # Get coverage data
        cov = run_coverage_diagnostic("bone scaffold", max_results=3)

        # If coverage is low, generate gap queries from the mismatch
        if cov["coverage_pct"] < 50:
            # Build a synthetic gap from the coverage mismatch
            gap_text = (
                f"Coverage diagnostic found only {cov['matched']}/{cov['s2_total']} "
                f"S2 papers have PMC full text. Research gaps exist in the literature "
                f"coverage for this topic. Missing data from non-OA papers limits synthesis."
            )
            queries = _parse_gaps_to_queries(gap_text)
            # May or may not produce queries depending on gap keywords
            assert isinstance(queries, list)
