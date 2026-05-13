#!/usr/bin/env python3
"""Investigate BM25 vs ChromaDB contribution to hybrid retrieval."""

from dotenv import load_dotenv
load_dotenv(override=True)

import sys, json, re, statistics
sys.path.insert(0, '.')
from pathlib import Path
from src.retrieval.chroma_client import ChromaClient
from src.retrieval.bm25_index import BM25Index
from src.anchoring.evidence_check import decompose_claims, _split_chunks_into_sentences
from src.unicode_map import scrub_unicode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

chroma = ChromaClient(collection_name='public_corpus', persist_directory='projects/default/chroma_data')
all_data = chroma.collection.get(include=['documents','metadatas'])
docs = all_data.get('documents') or []
metas = all_data.get('metadatas') or []

chunks = []
for doc, meta in zip(docs, metas):
    if (meta or {}).get('chunk_type') == 'reference':
        continue
    chunks.append({'text': scrub_unicode(str(doc)), 'metadata': meta or {}})

sentences = _split_chunks_into_sentences(chunks)

survey = json.loads(Path('projects/default/survey_result.json').read_text())
all_claims = []
for ts in survey.get('per_theme_syntheses', {}).values():
    for c in decompose_claims(scrub_unicode(ts.get('synthesis','') or '')):
        all_claims.append(c)

print(f'Testing {len(all_claims)} claims against {len(sentences)} evidence sentences ({len(chunks)} chunks)')
print()

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
ev_matrix = vectorizer.fit_transform(sentences)

bm25 = BM25Index()
bm25.add_documents(sentences)

bm25_scores = []
chroma_scores = []
hybrid_scores = []
bm25_missed = 0
chroma_missed = 0

for idx, claim in enumerate(all_claims):
    claim_vec = vectorizer.transform([scrub_unicode(claim)])
    full_sims = cosine_similarity(claim_vec, ev_matrix)[0]
    
    # BM25-only
    bm_results = list(bm25.query(claim, n_results=5))
    if bm_results:
        best_bm = 0.0
        for r in bm_results:
            rt = scrub_unicode(str(r))
            if rt in sentences:
                s = float(full_sims[sentences.index(rt)])
                if s > best_bm:
                    best_bm = s
    else:
        best_bm = 0.0
        bm25_missed += 1
    bm25_scores.append(best_bm)
    
    # ChromaDB-only
    dense = (chroma.query(claim, n_results=3) or {}).get('documents', [[]])[0]
    if dense:
        best_chroma = 0.0
        for chunk_text in dense:
            chunk_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', scrub_unicode(str(chunk_text))) if s.strip()]
            for cs in chunk_sents:
                try:
                    s = float(cosine_similarity(claim_vec, vectorizer.transform([cs]))[0, 0])
                    if s > best_chroma:
                        best_chroma = s
                except:
                    pass
    else:
        best_chroma = 0.0
        chroma_missed += 1
    chroma_scores.append(best_chroma)
    
    # Hybrid
    candidates = set()
    for r in bm_results:
        candidates.add(scrub_unicode(str(r)))
    for chunk_text in dense:
        chunk_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', scrub_unicode(str(chunk_text))) if s.strip()]
        for cs in chunk_sents:
            candidates.add(cs)
    if candidates:
        best_hybrid = 0.0
        for c in candidates:
            if c in sentences:
                s = float(full_sims[sentences.index(c)])
            else:
                s = float(cosine_similarity(claim_vec, vectorizer.transform([c]))[0, 0])
            if s > best_hybrid:
                best_hybrid = s
    else:
        best_hybrid = 0.0
    hybrid_scores.append(best_hybrid)

print(f"{'Retrieval':<20} {'Mean':>8} {'Min':>8} {'<0.35':>8} {'Missed':>8}")
print(f"{'-'*55}")
print(f"{'BM25-only':<20} {statistics.mean(bm25_scores):>8.4f} {min(bm25_scores):>8.4f} {sum(1 for s in bm25_scores if s < 0.35):>8} {bm25_missed:>8}")
print(f"{'ChromaDB-only':<20} {statistics.mean(chroma_scores):>8.4f} {min(chroma_scores):>8.4f} {sum(1 for s in chroma_scores if s < 0.35):>8} {chroma_missed:>8}")
print(f"{'Hybrid (both)':<20} {statistics.mean(hybrid_scores):>8.4f} {min(hybrid_scores):>8.4f} {sum(1 for s in hybrid_scores if s < 0.35):>8} {'N/A':>8}")

bm25_better = 0
chroma_better = 0
tie = 0
for b, c in zip(bm25_scores, chroma_scores):
    if b > c + 0.01:
        bm25_better += 1
    elif c > b + 0.01:
        chroma_better += 1
    else:
        tie += 1

print(f'\n{"BM25 better match":<25} {bm25_better:>4} claims ({bm25_better/len(all_claims)*100:.1f}%)')
print(f'{"ChromaDB better match":<25} {chroma_better:>4} claims ({chroma_better/len(all_claims)*100:.1f}%)')
print(f'{"Tie (<0.01 diff)":<25} {tie:>4} claims ({tie/len(all_claims)*100:.1f}%)')

print('\n=== Claims where BM25 significantly outperformed ChromaDB ===')
bm_wins = sorted(
    [(i, all_claims[i], bm25_scores[i] - chroma_scores[i])
     for i in range(len(all_claims))
     if bm25_scores[i] > chroma_scores[i] + 0.02],
    key=lambda x: x[2], reverse=True
)
for idx, claim, delta in bm_wins[:5]:
    bm_top = list(bm25.query(claim, n_results=1))[0]
    print(f'  Delta +{delta:.3f}  BM25={bm25_scores[idx]:.3f}  Chroma={chroma_scores[idx]:.3f}')
    print(f'  Claim: {claim[:130]}')
    print(f'  BM25 found: {str(bm_top)[:130]}')
    print()

print('=== Claims where ChromaDB significantly outperformed BM25 ===')
chroma_wins = sorted(
    [(i, all_claims[i], chroma_scores[i] - bm25_scores[i])
     for i in range(len(all_claims))
     if chroma_scores[i] > bm25_scores[i] + 0.02],
    key=lambda x: x[2], reverse=True
)
for idx, claim, delta in chroma_wins[:5]:
    dense = (chroma.query(claim, n_results=1) or {}).get('documents', [[]])[0]
    chroma_top = dense[0][:130] if dense else '(none)'
    print(f'  Delta +{delta:.3f}  Chroma={chroma_scores[idx]:.3f}  BM25={bm25_scores[idx]:.3f}')
    print(f'  Claim: {claim[:130]}')
    print(f'  Chroma found: {chroma_top[:130]}')
    print()
