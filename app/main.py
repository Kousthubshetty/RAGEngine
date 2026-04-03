import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pythonjsonlogger import jsonlogger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.dependencies import init_services
from app.middleware.error_handler import global_exception_handler
from app.middleware.rate_limiter import limiter
from app.routers import knowledge, query

# Structured logging
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s",
))
logging.root.handlers = [handler]
logging.root.setLevel(settings.log_level)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing services...")
    init_services()
    logger.info("Services ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

# Middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(Exception, global_exception_handler)

# Register routers
app.include_router(knowledge.router)
app.include_router(query.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
