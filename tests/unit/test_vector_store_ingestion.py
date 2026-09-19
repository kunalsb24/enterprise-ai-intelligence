from pathlib import Path
from unittest.mock import MagicMock, patch

from enterprise_ai.document_processing.models import (
    DocumentChunk,
)
from enterprise_ai.vector_store.ingestion import (
    ingest_document,
)


def test_ingest_document_replaces_existing_chunks():
    """Document ingestion should replace old chunks with new ones."""

    path = Path("data/raw/documents/incident.txt")

    chunks = [
        DocumentChunk(
            document_name="incident.txt",
            chunk_index=0,
            content="Payment failures increased.",
        )
    ]

    embedded_chunks = MagicMock()

    fake_engine = MagicMock()
    fake_connection = MagicMock()

    fake_engine.begin.return_value.__enter__.return_value = (
        fake_connection
    )

    with (
        patch(
            "enterprise_ai.vector_store.ingestion.process_document",
            return_value=chunks,
        ),
        patch(
            "enterprise_ai.vector_store.ingestion.EmbeddingModel"
        ) as mock_model_class,
        patch(
            "enterprise_ai.vector_store.ingestion.embed_chunks",
            return_value=embedded_chunks,
        ),
        patch(
            "enterprise_ai.vector_store.ingestion.get_engine",
            return_value=fake_engine,
        ),
        patch(
            "enterprise_ai.vector_store.ingestion.delete_document_chunks"
        ) as mock_delete,
        patch(
            "enterprise_ai.vector_store.ingestion.insert_document_chunks"
        ) as mock_insert,
    ):
        fake_model = mock_model_class.return_value

        ingest_document(path)

    mock_delete.assert_called_once_with(
        fake_connection,
        "incident.txt",
    )

    mock_insert.assert_called_once_with(
        fake_connection,
        embedded_chunks,
    )

    fake_engine.dispose.assert_called_once()

    mock_model_class.assert_called_once()

def test_ingest_documents_reuses_embedding_model():
    paths = [
        Path("document_one.txt"),
        Path("document_two.txt"),
    ]

    with patch(
        "enterprise_ai.vector_store.ingestion.EmbeddingModel"
    ) as mock_model_class, patch(
        "enterprise_ai.vector_store.ingestion.ingest_document"
    ) as mock_ingest_document:

        from enterprise_ai.vector_store.ingestion import (
            ingest_documents,
        )

        ingest_documents(paths)

        mock_model_class.assert_called_once()

        embedding_model = mock_model_class.return_value

        assert mock_ingest_document.call_count == 2

        mock_ingest_document.assert_any_call(
            paths[0],
            embedding_model=embedding_model,
        )

        mock_ingest_document.assert_any_call(
            paths[1],
            embedding_model=embedding_model,
        )