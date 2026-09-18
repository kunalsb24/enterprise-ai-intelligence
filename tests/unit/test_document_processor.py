from pathlib import Path

from enterprise_ai.document_processing.processor import process_document


def test_process_document_creates_metadata_rich_chunks(
    tmp_path: Path,
):
    """Processing a document should create chunks with source metadata."""

    document = tmp_path / "incident.txt"
    document.write_text(
        "Overview\n\n"
        "Payment processing failures occurred.\n\n"
        "Remediation\n\n"
        "Engineering corrected the configuration.",
        encoding="utf-8",
    )

    chunks = process_document(
        document,
        max_characters=70,
    )

    assert len(chunks) > 0

    for index, chunk in enumerate(chunks):
        assert chunk.document_name == "incident.txt"
        assert chunk.chunk_index == index
        assert chunk.content