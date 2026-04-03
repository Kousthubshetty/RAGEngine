from app.services.vectorstore import VectorStoreService
from app.services.rag_chain import RAGService

_vectorstore_service: VectorStoreService | None = None
_rag_service: RAGService | None = None


def init_services():
    global _vectorstore_service, _rag_service
    _vectorstore_service = VectorStoreService()
    _rag_service = RAGService(_vectorstore_service)


def get_vectorstore_service() -> VectorStoreService:
    assert _vectorstore_service is not None, "Services not initialized"
    return _vectorstore_service


def get_rag_service() -> RAGService:
    assert _rag_service is not None, "Services not initialized"
    return _rag_service
