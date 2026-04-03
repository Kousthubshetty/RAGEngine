import logging

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from app.dependencies import get_rag_service
from app.middleware.rate_limiter import limiter
from app.models.requests import AskRequest
from app.models.responses import AskResponse
from app.services.rag_chain import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask")
@limiter.limit("10/minute")
async def ask(
    request: Request,
    body: AskRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    if body.stream:
        return EventSourceResponse(
            rag_service.stream_query(body.question, top_k=body.top_k),
            media_type="text/event-stream",
        )

    response: AskResponse = await rag_service.query(
        body.question, top_k=body.top_k,
    )
    return response
