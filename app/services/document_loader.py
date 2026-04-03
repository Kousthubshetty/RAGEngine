import logging
from pathlib import Path

from langchain_community.document_loaders import (
    JSONLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)

LOADER_MAP = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PyPDFLoader,
    ".json": lambda path: JSONLoader(file_path=path, jq_schema=".", text_content=False),
}


def load_and_split_documents() -> list:
    knowledge_dir = Path(settings.knowledge_dir)
    if not knowledge_dir.exists():
        logger.warning("Knowledge directory %s does not exist", knowledge_dir)
        return []

    documents = []
    files = list(knowledge_dir.iterdir())
    doc_count = 0

    for file_path in files:
        if file_path.suffix.lower() not in LOADER_MAP:
            logger.debug("Skipping unsupported file: %s", file_path.name)
            continue

        try:
            loader_factory = LOADER_MAP[file_path.suffix.lower()]
            if callable(loader_factory) and file_path.suffix.lower() == ".json":
                loader = loader_factory(str(file_path))
            else:
                loader = loader_factory(str(file_path))
            docs = loader.load()
            documents.extend(docs)
            doc_count += 1
            logger.info("Loaded %d pages from %s", len(docs), file_path.name)
        except Exception:
            logger.warning("Failed to load %s, skipping", file_path.name, exc_info=True)

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks", doc_count, len(chunks))
    return chunks
