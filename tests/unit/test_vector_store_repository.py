from unittest.mock import MagicMock, patch

from enterprise_ai.document_processing.models import (
    EmbeddedDocumentChunk,
)
from enterprise_ai.vector_store.repository import (
    embedding_to_vector_string,
    insert_document_chunks,
)


def test_embedding_to_vector_string():
    """Python embeddings should be converted to pgvector format."""

    embedding = [0.1, 0.2, 0.3]

    result = embedding_to_vector_string(embedding)

    assert result == "[0.1,0.2,0.3]"

def test_insert_document_chunks_inserts_every_chunk():
    """Every embedded chunk should be passed to the insert function."""

    chunks = [
        EmbeddedDocumentChunk(
            document_name="incident.txt",
            chunk_index=0,
            content="Payment failures increased.",
            embedding=[0.1, 0.2, 0.3],
        ),
        EmbeddedDocumentChunk(
            document_name="incident.txt",
            chunk_index=1,
            content="Engineering corrected the configuration.",
            embedding=[0.4, 0.5, 0.6],
        ),
    ]

    fake_connection = MagicMock()

    with patch(
        "enterprise_ai.vector_store.repository.insert_document_chunk"
    ) as mock_insert:
        insert_document_chunks(
            fake_connection,
            chunks,
        )

    assert mock_insert.call_count == 2

    mock_insert.assert_any_call(
        fake_connection,
        chunks[0],
    )

    mock_insert.assert_any_call(
        fake_connection,
        chunks[1],
    )

def test_search_similar_chunks_returns_ranked_results():
    """Semantic search should convert database rows into result models."""

    from enterprise_ai.vector_store.repository import (
        search_similar_chunks,
    )

    fake_connection = MagicMock()

    row_1 = MagicMock()
    row_1.document_name = "incident.txt"
    row_1.chunk_index = 1
    row_1.content = "Enterprise accounts experienced payment failures."
    row_1.similarity = 0.82

    row_2 = MagicMock()
    row_2.document_name = "incident.txt"
    row_2.chunk_index = 2
    row_2.content = "Engineering investigated the billing migration."
    row_2.similarity = 0.67

    fake_connection.execute.return_value = [
        row_1,
        row_2,
    ]

    results = search_similar_chunks(
        fake_connection,
        [0.1, 0.2, 0.3],
        limit=2,
    )

    assert len(results) == 2

    assert results[0].document_name == "incident.txt"
    assert results[0].chunk_index == 1
    assert results[0].similarity == 0.82

    assert results[1].chunk_index == 2
    assert results[1].similarity == 0.67