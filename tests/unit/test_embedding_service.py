from unittest.mock import MagicMock

from enterprise_ai.document_processing.models import DocumentChunk
from enterprise_ai.embeddings.service import embed_chunks


def test_embed_chunks_adds_embeddings():
    """Each document chunk should receive an embedding."""

    chunks = [
        DocumentChunk(
            document_name="incident.txt",
            chunk_index=0,
            content="Payment failures increased.",
        ),
        DocumentChunk(
            document_name="incident.txt",
            chunk_index=1,
            content="Engineering corrected the configuration.",
        ),
    ]

    fake_embedding_model = MagicMock()

    fake_embedding_model.embed_text.side_effect = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    result = embed_chunks(
        chunks,
        fake_embedding_model,
    )

    assert len(result) == 2

    assert result[0].document_name == "incident.txt"
    assert result[0].chunk_index == 0
    assert result[0].content == "Payment failures increased."
    assert result[0].embedding == [0.1, 0.2, 0.3]

    assert result[1].document_name == "incident.txt"
    assert result[1].chunk_index == 1
    assert result[1].embedding == [0.4, 0.5, 0.6]