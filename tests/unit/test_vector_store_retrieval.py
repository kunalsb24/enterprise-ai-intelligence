from unittest.mock import MagicMock, patch

from enterprise_ai.document_processing.models import (
    RetrievedDocumentChunk,
)
from enterprise_ai.vector_store.retrieval import retrieve_chunks


def test_retrieve_chunks():
    fake_embedding = [0.1, 0.2, 0.3]

    fake_results = [
        RetrievedDocumentChunk(
            document_name="incident.txt",
            chunk_index=1,
            content="Enterprise customers experienced payment failures.",
            similarity=0.85,
        )
    ]

    with patch(
        "enterprise_ai.vector_store.retrieval.EmbeddingModel"
    ) as mock_model_class, patch(
        "enterprise_ai.vector_store.retrieval.get_engine"
    ) as mock_get_engine, patch(
        "enterprise_ai.vector_store.retrieval.search_similar_chunks"
    ) as mock_search:

        mock_model = mock_model_class.return_value
        mock_model.embed_text.return_value = fake_embedding

        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        mock_search.return_value = fake_results

        results = retrieve_chunks(
            "Which customers had payment problems?",
            limit=5,
        )

        mock_model.embed_text.assert_called_once_with(
            "Which customers had payment problems?"
        )

        mock_search.assert_called_once_with(
            mock_engine.connect.return_value.__enter__.return_value,
            fake_embedding,
            limit=5,
        )

        mock_engine.dispose.assert_called_once()

        assert results == fake_results

from unittest.mock import Mock, patch

from enterprise_ai.document_processing.models import (
    RerankedDocumentChunk,
    RetrievedDocumentChunk,
)
from enterprise_ai.vector_store.retrieval import (
    retrieve_and_rerank_chunks,
)


def test_retrieve_and_rerank_chunks():
    candidates = [
        RetrievedDocumentChunk(
            document_name="billing.txt",
            chunk_index=0,
            content="Billing failures occurred.",
            similarity=0.90,
        ),
        RetrievedDocumentChunk(
            document_name="portal.txt",
            chunk_index=0,
            content="The portal did not change payment processing.",
            similarity=0.70,
        ),
    ]

    reranked = [
        RerankedDocumentChunk(
            document_name="portal.txt",
            chunk_index=0,
            content="The portal did not change payment processing.",
            similarity=0.70,
            reranking_score=0.95,
        ),
        RerankedDocumentChunk(
            document_name="billing.txt",
            chunk_index=0,
            content="Billing failures occurred.",
            similarity=0.90,
            reranking_score=0.20,
        ),
    ]

    embedding_model = Mock()
    reranking_model = Mock()

    with patch(
        "enterprise_ai.vector_store.retrieval.retrieve_chunks",
        return_value=candidates,
    ) as mock_retrieve, patch(
        "enterprise_ai.vector_store.retrieval.rerank_chunks",
        return_value=reranked,
    ) as mock_rerank:
        results = retrieve_and_rerank_chunks(
            query="Did the portal cause payment problems?",
            candidate_limit=10,
            final_limit=1,
            embedding_model=embedding_model,
            reranking_model=reranking_model,
        )

    assert len(results) == 1
    assert results[0].document_name == "portal.txt"

    mock_retrieve.assert_called_once_with(
        query="Did the portal cause payment problems?",
        limit=10,
        embedding_model=embedding_model,
    )

    mock_rerank.assert_called_once_with(
        query="Did the portal cause payment problems?",
        chunks=candidates,
        reranking_model=reranking_model,
    )