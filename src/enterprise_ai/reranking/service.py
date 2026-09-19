from enterprise_ai.document_processing.models import (
    RerankedDocumentChunk,
    RetrievedDocumentChunk,
)
from enterprise_ai.reranking.model import RerankingModel


def rerank_chunks(
    query: str,
    chunks: list[RetrievedDocumentChunk],
    reranking_model: RerankingModel,
) -> list[RerankedDocumentChunk]:
    """Rerank retrieved chunks using a cross-encoder."""

    if not chunks:
        return []

    documents = [
        chunk.content
        for chunk in chunks
    ]

    scores = reranking_model.score(
        query=query,
        documents=documents,
    )

    if len(scores) != len(chunks):
        raise ValueError(
            "Reranking model must return one score per chunk."
        )

    reranked_chunks = [
        RerankedDocumentChunk(
            document_name=chunk.document_name,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            similarity=chunk.similarity,
            reranking_score=score,
        )
        for chunk, score in zip(chunks, scores)
    ]

    return sorted(
        reranked_chunks,
        key=lambda chunk: chunk.reranking_score,
        reverse=True,
    )