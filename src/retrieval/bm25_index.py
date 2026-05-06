from typing import List
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self.corpus: List[str] = []
        self.index = None

    def add_documents(self, documents: List[str]):
        self.corpus.extend(documents)
        tokenized = [doc.split() for doc in self.corpus]
        self.index = BM25Okapi(tokenized)

    def query(self, query: str, n_results: int = 5) -> List[str]:
        if not self.index:
            return []
        tokenized_query = query.split()
        scores = self.index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        return [self.corpus[i] for i in top_indices]
