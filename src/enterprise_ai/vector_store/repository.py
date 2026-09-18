from sqlalchemy import text
from sqlalchemy.engine import Connection

from enterprise_ai.document_processing.models import (
    EmbeddedDocumentChunk,
    RetrievedDocumentChunk,
)


def embedding_to_vector_string(
    embedding: list[float],
) -> str:
    """Convert a Python embedding into pgvector text format."""

    return "[" + ",".join(
        str(value) for value in embedding
    ) + """]"""


def insert_document_chunk(
    connection: Connection,
    chunk: EmbeddedDocumentChunk,
) -> None:
    """Insert one embedded document chunk into PostgreSQL."""

    query = text(
        """
        INSERT INTO document_chunks (
            document_name,
            chunk_index,
            content,
            embedding
        )
        VALUES (
            :document_name,
            :chunk_index,
            :content,
            CAST(:embedding AS vector)
        )
        """
    )

    connection.execute(
        query,
        {
            "document_name": chunk.document_name,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "embedding": embedding_to_vector_string(
                chunk.embedding
            ),
        },
    )

def insert_document_chunks(
    connection: Connection,
    chunks: list[EmbeddedDocumentChunk],
) -> None:
    """Insert multiple embedded document chunks into PostgreSQL."""

    for chunk in chunks:
        insert_document_chunk(
            connection,
            chunk,
        )

def delete_document_chunks(
    connection: Connection,
    document_name: str,
) -> None:
    """Delete all stored chunks belonging to one document."""

    query = text(
        """
        DELETE FROM document_chunks
        WHERE document_name = :document_name
        """
    )

    connection.execute(
        query,
        {
            "document_name": document_name,
        },
    )

def search_similar_chunks(
    connection: Connection,
    query_embedding: list[float],
    limit: int = 3,
) -> list[RetrievedDocumentChunk]:
    """Return document chunks most similar to a query embedding."""

    query = text(
        """
        SELECT
            document_name,
            chunk_index,
            content,
            1 - (
                embedding <=> CAST(:query_embedding AS vector)
            ) AS similarity
        FROM document_chunks
        ORDER BY
            embedding <=> CAST(:query_embedding AS vector)
        LIMIT :limit
        """
    )

    result = connection.execute(
        query,
        {
            "query_embedding": embedding_to_vector_string(
                query_embedding
            ),
            "limit": limit,
        },
    )

    return [
        RetrievedDocumentChunk(
            document_name=row.document_name,
            chunk_index=row.chunk_index,
            content=row.content,
            similarity=float(row.similarity),
        )
        for row in result
    ]