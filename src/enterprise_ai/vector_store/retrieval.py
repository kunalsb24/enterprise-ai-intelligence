from enterprise_ai.database.connection import get_engine
from enterprise_ai.document_processing.models import (
    RerankedDocumentChunk,
    RetrievedDocumentChunk,
)
from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.vector_store.repository import (
    search_similar_chunks,
)

from enterprise_ai.reranking.model import RerankingModel
from enterprise_ai.reranking.service import rerank_chunks

def retrieve_chunks(
    query: str,
    limit: int = 3,
    embedding_model: EmbeddingModel | None = None,
) -> list[RetrievedDocumentChunk]:
    """Retrieve document chunks that are semantically similar to a query."""

    if embedding_model is None:
        embedding_model = EmbeddingModel()

    query_embedding = embedding_model.embed_text(query)

    engine = get_engine()

    try:
        with engine.connect() as connection:
            return search_similar_chunks(
                connection,
                query_embedding,
                limit=limit,
            )
    finally:
        engine.dispose()

def retrieve_and_rerank_chunks(
    query: str,
    candidate_limit: int = 10,
    final_limit: int = 3,
    embedding_model: EmbeddingModel | None = None,
    reranking_model: RerankingModel | None = None,
) -> list[RerankedDocumentChunk]:
    """Retrieve candidate chunks and rerank them for better relevance."""

    if reranking_model is None:
        reranking_model = RerankingModel()

    candidates = retrieve_chunks(
        query=query,
        limit=candidate_limit,
        embedding_model=embedding_model,
    )

    reranked_chunks = rerank_chunks(
        query=query,
        chunks=candidates,
        reranking_model=reranking_model,
    )

    return reranked_chunks[:final_limit]