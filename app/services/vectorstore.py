import logging

import chromadb
from cachetools import TTLCache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.services.document_loader import load_and_split_documents, list_knowledge_bases

logger = logging.getLogger(__name__)


def _collection_name(knowledge_name: str) -> str:
    return f"knowledge_{knowledge_name}"


class VectorStoreService:
    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._vectorstores: dict[str, Chroma] = {}
        self._cache: TTLCache = TTLCache(maxsize=128, ttl=300)

        # Initialize vectorstores for existing knowledge bases
        for kb in list_knowledge_bases():
            col = _collection_name(kb)
            self._vectorstores[kb] = Chroma(
                client=self._client,
                collection_name=col,
                embedding_function=self._embeddings,
            )

    def _get_or_create_vectorstore(self, knowledge_name: str) -> Chroma:
        if knowledge_name not in self._vectorstores:
            self._vectorstores[knowledge_name] = Chroma(
                client=self._client,
                collection_name=_collection_name(knowledge_name),
                embedding_function=self._embeddings,
            )
        return self._vectorstores[knowledge_name]

    def clear_and_reload(self, knowledge_name: str | None = None) -> tuple[int, int]:
        if knowledge_name:
            return self._reload_single(knowledge_name)
        else:
            return self._reload_all()

    def _reload_single(self, knowledge_name: str) -> tuple[int, int]:
        col = _collection_name(knowledge_name)
        try:
            self._client.delete_collection(col)
        except ValueError:
            pass

        chunks = load_and_split_documents(knowledge_name)
        if not chunks:
            self._vectorstores[knowledge_name] = Chroma(
                client=self._client,
                collection_name=col,
                embedding_function=self._embeddings,
            )
            return 0, 0

        self._vectorstores[knowledge_name] = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            client=self._client,
            collection_name=col,
        )

        # Invalidate cache entries for this knowledge
        keys_to_remove = [k for k in self._cache if k.startswith(f"{knowledge_name}::")]
        for k in keys_to_remove:
            del self._cache[k]

        doc_sources = {c.metadata.get("source", "") for c in chunks}
        logger.info("Loaded %d chunks from %d documents for '%s'", len(chunks), len(doc_sources), knowledge_name)
        return len(doc_sources), len(chunks)

    def _reload_all(self) -> tuple[int, int]:
        total_docs, total_chunks = 0, 0
        for kb in list_knowledge_bases():
            docs, chunks = self._reload_single(kb)
            total_docs += docs
            total_chunks += chunks
        return total_docs, total_chunks

    def as_retriever(self, knowledge_name: str, top_k: int | None = None):
        k = top_k or settings.top_k
        vs = self._get_or_create_vectorstore(knowledge_name)
        return vs.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, knowledge_name: str, top_k: int | None = None):
        k = top_k or settings.top_k
        cache_key = f"{knowledge_name}::{query}::{k}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        vs = self._get_or_create_vectorstore(knowledge_name)
        results = vs.similarity_search(query, k=k)
        self._cache[cache_key] = results
        return results
