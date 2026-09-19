from unittest.mock import patch

from enterprise_ai.document_processing.models import (
    RetrievedDocumentChunk,
)
from enterprise_ai.evaluation.models import (
    RetrievalEvaluationCase,
)
from enterprise_ai.evaluation.retrieval_evaluator import (
    evaluate_retrieval,
)


def test_evaluate_retrieval():
    cases = [
        RetrievalEvaluationCase(
            question="What caused payment failures?",
            expected_document="billing.txt",
        ),
        RetrievalEvaluationCase(
            question="What network capacity was added?",
            expected_document="network.txt",
        ),
    ]

    first_results = [
        RetrievedDocumentChunk(
            document_name="billing.txt",
            chunk_index=0,
            content="Payment processing failed.",
            similarity=0.90,
        )
    ]

    second_results = [
        RetrievedDocumentChunk(
            document_name="portal.txt",
            chunk_index=0,
            content="The customer portal was updated.",
            similarity=0.80,
        )
    ]

    with patch(
        "enterprise_ai.evaluation.retrieval_evaluator.EmbeddingModel"
    ) as mock_model_class, patch(
        "enterprise_ai.evaluation.retrieval_evaluator.retrieve_chunks"
    ) as mock_retrieve:

        embedding_model = mock_model_class.return_value

        mock_retrieve.side_effect = [
            first_results,
            second_results,
        ]

        score = evaluate_retrieval(
            cases,
            k=3,
        )

    assert score.recall_at_k == 0.5
    assert score.mean_reciprocal_rank == 0.5

    assert len(score.case_results) == 2

    assert score.case_results[0].question == (
        "What caused payment failures?"
    )
    assert score.case_results[0].expected_document == "billing.txt"
    assert score.case_results[0].retrieved_documents == [
        "billing.txt"
    ]
    assert score.case_results[0].recall_at_k == 1.0
    assert score.case_results[0].reciprocal_rank == 1.0

    assert score.case_results[1].question == (
        "What network capacity was added?"
    )
    assert score.case_results[1].expected_document == "network.txt"
    assert score.case_results[1].retrieved_documents == [
        "portal.txt"
    ]
    assert score.case_results[1].recall_at_k == 0.0
    assert score.case_results[1].reciprocal_rank == 0.0

    assert mock_retrieve.call_count == 2

    mock_model_class.assert_called_once()

    mock_retrieve.assert_any_call(
    "What caused payment failures?",
    limit=3,
    embedding_model=embedding_model,
)

    mock_retrieve.assert_any_call(
        "What network capacity was added?",
        limit=3,
        embedding_model=embedding_model,
    )

def test_evaluate_reranked_retrieval():
    from unittest.mock import Mock, patch

    from enterprise_ai.document_processing.models import (
        RerankedDocumentChunk,
    )
    from enterprise_ai.evaluation.models import (
        RetrievalEvaluationCase,
    )
    from enterprise_ai.evaluation.retrieval_evaluator import (
        evaluate_reranked_retrieval,
    )

    cases = [
        RetrievalEvaluationCase(
            question="Did the portal cause payment failures?",
            expected_document="portal.txt",
        )
    ]

    retrieved_chunks = [
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

    with patch(
        "enterprise_ai.evaluation.retrieval_evaluator.EmbeddingModel"
    ) as mock_embedding_class, patch(
        "enterprise_ai.evaluation.retrieval_evaluator.RerankingModel"
    ) as mock_reranking_class, patch(
        "enterprise_ai.evaluation.retrieval_evaluator."
        "retrieve_and_rerank_chunks",
        return_value=retrieved_chunks,
    ) as mock_retrieve:

        result = evaluate_reranked_retrieval(
            cases,
            k=2,
            candidate_limit=10,
        )

    assert result.recall_at_k == 1.0
    assert result.mean_reciprocal_rank == 1.0

    assert len(result.case_results) == 1
    assert result.case_results[0].retrieved_documents == [
        "portal.txt",
        "billing.txt",
    ]

    mock_retrieve.assert_called_once_with(
        query="Did the portal cause payment failures?",
        candidate_limit=10,
        final_limit=2,
        embedding_model=mock_embedding_class.return_value,
        reranking_model=mock_reranking_class.return_value,
    )