from pydantic import BaseModel


class RAGSource(BaseModel):
    document_name: str
    chunk_index: int
    text: str
    similarity: float
    reranking_score: float


class RAGResult(BaseModel):
    answer: str
    sources: list[RAGSource]