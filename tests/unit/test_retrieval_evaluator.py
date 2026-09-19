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