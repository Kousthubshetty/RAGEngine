import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.models.responses import AskResponse, SourceDocument
from app.services.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.
Use only the context below to answer. If the context doesn't contain enough information, say so.

Context:
{context}"""

USER_TEMPLATE = "{question}"


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class RAGService:
    def __init__(self, vectorstore_service: VectorStoreService):
        self._vectorstore = vectorstore_service
        self._llm = ChatAnthropic(
            model=settings.model_name,
            api_key=settings.anthropic_api_key,
            streaming=True,
            temperature=0,
        )
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            ("human", USER_TEMPLATE),
        ])

    def _build_chain(self, top_k: int):
        retriever = self._vectorstore.as_retriever(top_k=top_k)
        return (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | self._prompt
            | self._llm
            | StrOutputParser()
        )

    async def query(self, question: str, top_k: int = 4) -> AskResponse:
        chain = self._build_chain(top_k)
        source_docs = self._vectorstore.similarity_search(question, top_k=top_k)
        answer = await chain.ainvoke(question)

        sources = [
            SourceDocument(content=doc.page_content, metadata=doc.metadata)
            for doc in source_docs
        ]
        return AskResponse(answer=answer, sources=sources)

    async def stream_query(self, question: str, top_k: int = 4):
        chain = self._build_chain(top_k)
        source_docs = self._vectorstore.similarity_search(question, top_k=top_k)

        try:
            async for token in chain.astream(question):
                yield f"data: {json.dumps({'token': token})}\n\n"

            sources = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in source_docs
            ]
            yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"
        except Exception as e:
            logger.error("Streaming error: %s", e, exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
