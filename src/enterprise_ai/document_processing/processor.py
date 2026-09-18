from pathlib import Path

from enterprise_ai.document_processing.chunking import chunk_text
from enterprise_ai.document_processing.loaders import load_text_document
from enterprise_ai.document_processing.models import DocumentChunk


def process_document(
    path: Path,
    max_characters: int = 500,
) -> list[DocumentChunk]:
    """Load a document and convert it into metadata-rich chunks."""

    text = load_text_document(path)

    text_chunks = chunk_text(
        text,
        max_characters=max_characters,
    )

    document_chunks = [
        DocumentChunk(
            document_name=path.name,
            chunk_index=index,
            content=content,
        )
        for index, content in enumerate(text_chunks)
    ]

    return document_chunks