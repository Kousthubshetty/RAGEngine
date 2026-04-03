import time
import logging

from fastapi import APIRouter, Depends, Request

from app.dependencies import get_vectorstore_service
from app.middleware.rate_limiter import limiter
from app.models.requests import RefreshRequest
from app.models.responses import RefreshResponse, KnowledgeListResponse
from app.services.document_loader import list_knowledge_bases
from app.services.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/knowledge", response_model=KnowledgeListResponse)
async def get_knowledge_bases():
    return KnowledgeListResponse(knowledge_bases=list_knowledge_bases())


@router.post("/refresh-knowledge", response_model=RefreshResponse)
@limiter.limit("2/minute")
async def refresh_knowledge(
    request: Request,
    body: RefreshRequest = RefreshRequest(),
    vectorstore: VectorStoreService = Depends(get_vectorstore_service),
):
    start = time.time()
    doc_count, chunk_count = vectorstore.clear_and_reload(knowledge_name=body.knowledge)
    processing_time = round(time.time() - start, 2)

    knowledge_label = body.knowledge if body.knowledge else list_knowledge_bases()

    logger.info(
        "Knowledge refresh: %d docs, %d chunks in %.2fs (knowledge=%s)",
        doc_count, chunk_count, processing_time, knowledge_label,
    )
    return RefreshResponse(
        status="success",
        knowledge=knowledge_label,
        doc_count=doc_count,
        chunk_count=chunk_count,
        processing_time=processing_time,
    )
