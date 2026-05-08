import chromadb
from chromadb.config import Settings

class ChromaClient:
    def __init__(self, collection_name: str, persist_directory: str = None):
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, ids, documents, metadatas=None):
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_text: str, n_results: int = 5, include_distances: bool = False):
        include = ["documents", "metadatas"]
        if include_distances:
            include.append("distances")
        return self.collection.query(query_texts=[query_text], n_results=n_results, include=include)
