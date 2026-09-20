from unittest.mock import MagicMock, patch

from enterprise_ai.evaluation.models import RAGEvaluationCase
from enterprise_ai.evaluation.rag_evaluator import evaluate_rag
from enterprise_ai.rag.models import RAGResult


@patch("enterprise_ai.evaluation.rag_evaluator.answer_question")
def test_evaluate_rag(mock_answer_question):
    cases = [
        RAGEvaluationCase(
            question="Which customers had more payment failures?",
            expected_answer_contains=["Germany", "Enterprise"],
            should_abstain=False,
        ),
        RAGEvaluationCase(
            question="What was company revenue?",
            expected_answer_contains=[],
            should_abstain=True,
        ),
    ]

    mock_answer_question.side_effect = [
        RAGResult(
            answer=(
                "Germany Enterprise customers experienced "
                "more payment failures."
            ),
            sources=[],
        ),
        RAGResult(
            answer=(
                "The provided evidence is insufficient "
                "to answer the question."
            ),
            sources=[],
        ),
    ]

    llm = MagicMock()
    embedding_model = MagicMock()
    reranking_model = MagicMock()

    result = evaluate_rag(
        cases=cases,
        llm=llm,
        embedding_model=embedding_model,
        reranking_model=reranking_model,
    )

    assert result.answer_correctness == 1.0
    assert result.abstention_correctness == 1.0

    assert len(result.case_results) == 2

    assert result.case_results[0].answer_correctness == 1.0
    assert result.case_results[0].abstention_correctness == 1.0

    assert result.case_results[1].answer_correctness == 0.0
    assert result.case_results[1].abstention_correctness == 1.0

    assert mock_answer_question.call_count == 2

    first_call = mock_answer_question.call_args_list[0]

    assert first_call.kwargs["llm"] is llm
    assert first_call.kwargs["embedding_model"] is embedding_model
    assert first_call.kwargs["reranking_model"] is reranking_model