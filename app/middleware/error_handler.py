import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    if isinstance(exc, FileNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    exc_type = type(exc).__name__

    # ChromaDB errors
    if "chroma" in exc_type.lower() or "Chroma" in type(exc).__module__:
        logger.error("ChromaDB error: %s", exc, exc_info=True)
        return JSONResponse(status_code=503, content={"detail": "Vector store unavailable"})

    # Anthropic API errors
    if "api" in exc_type.lower() or "anthropic" in getattr(type(exc), "__module__", ""):
        logger.error("API error: %s", exc, exc_info=True)
        return JSONResponse(status_code=502, content={"detail": "LLM service error"})

    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
