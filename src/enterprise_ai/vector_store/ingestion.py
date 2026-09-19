from pathlib import Path

from enterprise_ai.database.connection import get_engine
from enterprise_ai.document_processing.processor import (
    process_document,
)
from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.embeddings.service import embed_chunks
from enterprise_ai.vector_store.repository import (
    delete_document_chunks,
    insert_document_chunks,
)


def ingest_document(
    path: Path,
    embedding_model: EmbeddingModel | None = None,
) -> None:
    """Process, embed, and store one document."""

    chunks = process_document(path)

    if embedding_model is None:
        embedding_model = EmbeddingModel()

    embedded_chunks = embed_chunks(
        chunks,
        embedding_model,
    )

    engine = get_engine()

    try:
        with engine.begin() as connection:
            delete_document_chunks(
                connection,
                path.name,
            )

            insert_document_chunks(
                connection,
                embedded_chunks,
            )
    finally:
        engine.dispose()

def ingest_documents(
    paths: list[Path],
) -> None:
    """Process, embed, and store multiple documents."""

    embedding_model = EmbeddingModel()

    for path in paths:
        ingest_document(
            path,
            embedding_model=embedding_model,
        )