from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    knowledge: str = Field(..., min_length=1, max_length=100, description="Knowledge base subfolder name (e.g. 'faq', 'mahabharata')")
    top_k: int = Field(default=4, ge=1, le=20)
    stream: bool = False


class RefreshRequest(BaseModel):
    knowledge: str | None = Field(default=None, min_length=1, max_length=100, description="Specific knowledge base to refresh. If omitted, refreshes all.")
