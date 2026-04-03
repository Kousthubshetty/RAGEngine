import time
import logging

from fastapi import APIRouter, Depends, Request

from app.dependencies import get_vectorstore_service
from app.middleware.rate_limiter import limiter
from app.models.responses import RefreshResponse
from app.services.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/refresh-knowledge", response_model=RefreshResponse)
@limiter.limit("2/minute")
async def refresh_knowledge(
    request: Request,
    vectorstore: VectorStoreService = Depends(get_vectorstore_service),
):
    start = time.time()
    doc_count, chunk_count = vectorstore.clear_and_reload()
    processing_time = round(time.time() - start, 2)

    logger.info(
        "Knowledge refresh: %d docs, %d chunks in %.2fs",
        doc_count, chunk_count, processing_time,
    )
    return RefreshResponse(
        status="success",
        doc_count=doc_count,
        chunk_count=chunk_count,
        processing_time=processing_time,
    )
