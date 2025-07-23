import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection("docs")
        self.embedder = SentenceTransformer(embedding_model)

    def add_documents(self, docs: List[Dict]):
        texts = [doc["content"] for doc in docs]
        metadatas = [{k: v for k, v in doc.items() if k != "content"} for doc in docs]
        embeddings = self.embedder.encode(texts).tolist()
        ids = [str(i) for i in range(len(texts))]
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        self.logger.info(f"Added {len(texts)} documents to vector store.")

    def query(self, query_text: str, n_results: int = 3) -> List[Dict]:
        embedding = self.embedder.encode([query_text]).tolist()[0]
        results = self.collection.query(query_embeddings=[embedding], n_results=n_results)
        return [
            {"content": doc, **meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ] 