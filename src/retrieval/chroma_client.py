import chromadb
from chromadb.config import Settings
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)


class ChromaClient:
    def __init__(self, collection_name: str, persist_directory: str = None):
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, ids, documents, metadatas=None):
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def add_documents_deduped(self, ids, documents, metadatas=None) -> int:
        """Add documents, skipping any with IDs that already exist in the collection.

        Returns the number of documents actually added.
        """
        existing = self.get_existing_ids(ids)
        new_mask = [i for i, doc_id in enumerate(ids) if doc_id not in existing]
        if not new_mask:
            return 0
        new_ids = [ids[i] for i in new_mask]
        new_docs = [documents[i] for i in new_mask]
        new_metas = [metadatas[i] for i in new_mask] if metadatas else None
        self.collection.add(ids=new_ids, documents=new_docs, metadatas=new_metas)
        skipped = len(ids) - len(new_ids)
        if skipped:
            logger.debug("Skipped %d/%d duplicate IDs in ChromaDB", skipped, len(ids))
        return len(new_ids)

    def get_existing_ids(self, ids: List[str]) -> set:
        """Check which of the given IDs already exist in the collection."""
        if not ids:
            return set()
        try:
            result = self.collection.get(ids=ids, include=[])
            return set(result.get("ids", []))
        except Exception:
            return set()

    def query(self, query_text: str, n_results: int = 5, include_distances: bool = False):
        include = ["documents", "metadatas"]
        if include_distances:
            include.append("distances")
        return self.collection.query(query_texts=[query_text], n_results=n_results, include=include)
