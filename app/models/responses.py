from pydantic import BaseModel


class SourceDocument(BaseModel):
    content: str
    metadata: dict


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


class RefreshResponse(BaseModel):
    status: str
    doc_count: int
    chunk_count: int
    processing_time: float
