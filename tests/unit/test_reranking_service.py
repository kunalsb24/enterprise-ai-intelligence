from unittest.mock import Mock

from enterprise_ai.document_processing.models import (
    RetrievedDocumentChunk,
)
from enterprise_ai.reranking.service import rerank_chunks


def test_rerank_chunks_orders_by_reranking_score():
    chunks = [
        RetrievedDocumentChunk(
            document_name="billing.txt",
            chunk_index=0,
            content="Payment processing failures occurred.",
            similarity=0.90,
        ),
        RetrievedDocumentChunk(
            document_name="portal.txt",
            chunk_index=0,
            content="The portal did not change payment processing.",
            similarity=0.70,
        ),
    ]

    mock_model = Mock()

    # The reranker prefers the portal chunk even though
    # vector similarity preferred the billing chunk.
    mock_model.score.return_value = [0.2, 0.9]

    results = rerank_chunks(
        query="Did the portal cause payment problems?",
        chunks=chunks,
        reranking_model=mock_model,
    )

    assert len(results) == 2

    assert results[0].document_name == "portal.txt"
    assert results[0].similarity == 0.70
    assert results[0].reranking_score == 0.9

    assert results[1].document_name == "billing.txt"
    assert results[1].similarity == 0.90
    assert results[1].reranking_score == 0.2


import pytest


def test_rerank_chunks_raises_error_for_wrong_score_count():
    chunks = [
        RetrievedDocumentChunk(
            document_name="billing.txt",
            chunk_index=0,
            content="Billing failures occurred.",
            similarity=0.90,
        ),
        RetrievedDocumentChunk(
            document_name="portal.txt",
            chunk_index=0,
            content="The portal was updated.",
            similarity=0.70,
        ),
    ]

    mock_model = Mock()

    # Two chunks, but the model incorrectly returns only one score.
    mock_model.score.return_value = [0.5]

    with pytest.raises(
        ValueError,
        match="Reranking model must return one score per chunk",
    ):
        rerank_chunks(
            query="What happened?",
            chunks=chunks,
            reranking_model=mock_model,
        )