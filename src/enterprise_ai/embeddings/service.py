from enterprise_ai.document_processing.models import (
    DocumentChunk,
    EmbeddedDocumentChunk,
)
from enterprise_ai.embeddings.model import EmbeddingModel


def embed_chunks(
    chunks: list[DocumentChunk],
    embedding_model: EmbeddingModel,
) -> list[EmbeddedDocumentChunk]:
    """Generate embeddings for a collection of document chunks."""

    embedded_chunks = []

    for chunk in chunks:
        embedding = embedding_model.embed_text(
            chunk.content
        )

        embedded_chunk = EmbeddedDocumentChunk(
            document_name=chunk.document_name,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            embedding=embedding,
        )

        embedded_chunks.append(embedded_chunk)

    return embedded_chunks