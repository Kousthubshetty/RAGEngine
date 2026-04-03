import logging

import chromadb
from cachetools import TTLCache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.services.document_loader import load_and_split_documents

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection_name = "knowledge"
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
        )
        self._cache: TTLCache = TTLCache(maxsize=128, ttl=300)

    def clear_and_reload(self) -> tuple[int, int]:
        # Delete existing collection and recreate
        try:
            self._client.delete_collection(self._collection_name)
        except ValueError:
            pass  # Collection doesn't exist yet

        chunks = load_and_split_documents()
        if not chunks:
            self._vectorstore = Chroma(
                client=self._client,
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
            )
            return 0, 0

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            client=self._client,
            collection_name=self._collection_name,
        )

        # Invalidate cache
        self._cache.clear()

        doc_sources = {c.metadata.get("source", "") for c in chunks}
        logger.info("Loaded %d chunks from %d documents", len(chunks), len(doc_sources))
        return len(doc_sources), len(chunks)

    def as_retriever(self, top_k: int | None = None):
        k = top_k or settings.top_k
        return self._vectorstore.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, top_k: int | None = None):
        k = top_k or settings.top_k
        cache_key = f"{query}::{k}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        results = self._vectorstore.similarity_search(query, k=k)
        self._cache[cache_key] = results
        return results
