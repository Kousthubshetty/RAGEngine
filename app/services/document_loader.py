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


def list_knowledge_bases() -> list[str]:
    knowledge_dir = Path(settings.knowledge_dir)
    if not knowledge_dir.exists():
        return []
    return sorted(
        d.name for d in knowledge_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def _load_files_from_dir(directory: Path) -> list:
    documents = []
    for file_path in directory.iterdir():
        if file_path.is_dir():
            continue
        if file_path.suffix.lower() not in LOADER_MAP:
            logger.debug("Skipping unsupported file: %s", file_path.name)
            continue

        try:
            loader_factory = LOADER_MAP[file_path.suffix.lower()]
            loader = loader_factory(str(file_path))
            docs = loader.load()
            documents.extend(docs)
            logger.info("Loaded %d pages from %s", len(docs), file_path.name)
        except Exception:
            logger.warning("Failed to load %s, skipping", file_path.name, exc_info=True)

    return documents


def load_and_split_documents(knowledge_name: str | None = None) -> list:
    knowledge_dir = Path(settings.knowledge_dir)
    if not knowledge_dir.exists():
        logger.warning("Knowledge directory %s does not exist", knowledge_dir)
        return []

    if knowledge_name:
        target_dir = knowledge_dir / knowledge_name
        if not target_dir.is_dir():
            raise FileNotFoundError(f"Knowledge base '{knowledge_name}' not found")
        documents = _load_files_from_dir(target_dir)
    else:
        # Load from all subfolders
        documents = []
        for subdir in knowledge_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                documents.extend(_load_files_from_dir(subdir))

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks
