"""
Phase 1 Demo — See your building blocks in action.
Run with: python demo_phase1.py
"""
from src.unicode_map import scrub_unicode
from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index
from src.citation_manager.zotero_adapter import ZoteroAdapter
from src.state import AgentState

print("=" * 60)
print("🔬 SECURE FEDERATED RAG — PHASE 1 DEMO")
print("=" * 60)

# ── 1. UNICODE SCRUBBER ──────────────────────────────────────
print("\n📝 1. Unicode Scrubbing (PDF → Clean Text)")
print("-" * 40)

biomedical_texts = [
    "TNF-α inhibitors reduced IL‑1β in TiO₂‑coated implants",
    "The Ti-6Al-4V alloy showed ΔE of 0.5±0.1 V at 37°C",
    "15 μg/mL of TGF-β1 was administered to the murine model",
]

for original in biomedical_texts:
    cleaned = scrub_unicode(original)
    print(f"  RAW:    {original}")
    print(f"  CLEAN:  {cleaned}")
    print()

# ── 2. CHROMADB & BM25 SEARCH ─────────────────────────────────
print("📚 2. Hybrid Search Demo (Concept + Keyword)")
print("-" * 40)

# Simulate a small biomedical corpus
documents = [
    "Titanium dioxide nanotubes enhance osseointegration in rat tibia models.",
    "IL-6 and TNF-alpha levels correlate with implant rejection in murine studies.",
    "Ti-6Al-4V alloy surfaces treated with NaOH showed improved cell adhesion.",
    "Stem cell therapy combined with BMP-2 accelerated bone regeneration.",
    "The PMID 12345678 article describes TiO2 nanoparticle toxicity in vitro.",
]

# Build both indexes
chroma = ChromaClient("demo_public")
chroma.add_documents(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
)

bm25 = BM25Index()
bm25.add_documents(documents)

# Test a concept search (ChromADB) vs exact match (BM25)
queries = ["bone growth", "IL-6", "TiO2", "PMID 12345678"]

for query in queries:
    print(f"  Query: '{query}'")
    
    # Dense search (semantic)
    chroma_results = chroma.query(query, n_results=2)
    top_dense = chroma_results["documents"][0]
    print(f"    Dense (concept):   {top_dense[0][:70]}...")
    
    # Sparse search (keyword)
    bm25_results = bm25.query(query, n_results=1)
    if bm25_results:
        print(f"    Sparse (keyword):  {bm25_results[0][:70]}...")
    print()

# ── 3. CITATION MANAGER ───────────────────────────────────────
print("📎 3. Citation Manager (Zotero-ready)")
print("-" * 40)

zm = ZoteroAdapter()

# Simulate adding a few papers
papers = [
    {"title": "Osseointegration of TiO2 Nanotubes in Rat Tibiae", "year": 2023},
    {"title": "IL-6 Mediated Inflammatory Response to Titanium Implants", "year": 2024},
    {"title": "A Review of Biomaterial Surface Modifications", "year": 2022},
]

for paper in papers:
    key = zm.add_item(paper)
    formatted = zm.format_citation_key(key)
    print(f"  Added: {paper['title'][:50]}...")
    print(f"    → Citation Key: {formatted}")
print(f"\n  ⚠️  Currently using placeholder keys — real Zotero integration in Phase 2")

# ── 4. AGENT STATE ────────────────────────────────────────────
print("\n🧠 4. Agent State Object (LangGraph-ready)")
print("-" * 40)

# This is what flows through every node of the graph
state: AgentState = {
    "user_query": "Show me studies about TiO2 nanotubes and IL-6 response",
    "query_scope": "public",
    "public_context": [
        {"id": "doc_0", "content": documents[0], "metadata": {"source": "PubMed"}},
        {"id": "doc_1", "content": documents[1], "metadata": {"source": "PubMed"}},
    ],
    "secure_context": [],
    "extracted_entities": {
        "materials": ["TiO2", "nanotubes"],
        "models": ["rat", "tibia"],
        "cytokines": ["IL-6", "TNF-alpha"],
    },
    "synthesis_draft": "",
    "citations_used": ["@placeholder_Osseointegration_of_TiO2_"],
    "final_output": "",
    "human_approved": False,
    "routes": None,
    "mode": "deep",
    "discovered_categories": {},
    "knowledge_graph_snapshot": {},
    "critic_feedback": "",
    "synthesis_revised": "",
    "anchoring_score": 0.0,
}

print(f"  Query:          {state['user_query']}")
print(f"  Scope:          {state['query_scope']}")
print(f"  Public docs:    {len(state['public_context'])}")
print(f"  Secure docs:    {len(state['secure_context'])} (air-gapped)")
print(f"  Entities found: {state['extracted_entities']}")
print(f"  Citations:      {state['citations_used']}")
print(f"  Approved:       {state['human_approved']}")

# ── 5. SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ Phase 1 Complete — All building blocks operational")
print("=" * 60)
print("""
  What you're seeing:
  • Unicode scrubber sanitizes biomedical PDF text
  • ChromaDB finds documents by meaning (bone growth → osseointegration)
  • BM25 finds documents by exact keyword (IL-6 → only IL-6)
  • Citation manager generates reference keys for Zotero
  • AgentState carries all data through the pipeline

  Next (Phase 2): Wire these together into a HybridRetriever node
  and pipe real PDFs through the ingestion pipeline.
""")