from typing import List
from pathlib import Path
import pickle
import logging
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class BM25Index:
    def __init__(self, persist_dir: str | Path | None = None):
        self.corpus: List[str] = []
        self.index = None
        self._persist_dir = Path(persist_dir) if persist_dir else None
        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    def add_documents(self, documents: List[str]):
        self.corpus.extend(documents)
        self._rebuild_index()
        if self._persist_dir:
            self.save()

    def _rebuild_index(self):
        tokenized = [doc.split() for doc in self.corpus]
        self.index = BM25Okapi(tokenized)

    def query(self, query: str, n_results: int = 5) -> List[str]:
        if not self.index:
            return []
        tokenized_query = query.split()
        scores = self.index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        return [self.corpus[i] for i in top_indices]

    def save(self, path: str | Path | None = None) -> None:
        """Persist BM25 corpus to disk so index can be rebuilt without re-reading ChromaDB."""
        dest = Path(path) if path else self._persist_dir
        if dest is None:
            return
        dest.mkdir(parents=True, exist_ok=True)
        corpus_path = dest / "bm25_corpus.pkl"
        with open(corpus_path, "wb") as f:
            pickle.dump(self.corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25: persisted %d documents to %s", len(self.corpus), corpus_path)

    def load(self, path: str | Path | None = None) -> bool:
        """Load BM25 corpus from disk and rebuild index. Returns True if loaded."""
        src = Path(path) if path else self._persist_dir
        if src is None:
            return False
        corpus_path = src / "bm25_corpus.pkl"
        if not corpus_path.exists():
            return False
        try:
            with open(corpus_path, "rb") as f:
                self.corpus = pickle.load(f)
            self._rebuild_index()
            logger.info("BM25: loaded %d documents from %s", len(self.corpus), corpus_path)
            return True
        except Exception as e:
            logger.warning("BM25: failed to load from %s: %s", corpus_path, e)
            return False

    def __len__(self) -> int:
        return len(self.corpus)
