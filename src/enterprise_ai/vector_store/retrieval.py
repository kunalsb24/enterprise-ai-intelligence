from enterprise_ai.database.connection import get_engine
from enterprise_ai.document_processing.models import (
    RetrievedDocumentChunk,
)
from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.vector_store.repository import (
    search_similar_chunks,
)


def retrieve_chunks(
    query: str,
    limit: int = 3,
) -> list[RetrievedDocumentChunk]:
    """Retrieve document chunks that are semantically similar to a query."""

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