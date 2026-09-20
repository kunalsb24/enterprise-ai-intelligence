from enterprise_ai.evaluation.models import (
    RAGCaseResult,
    RAGEvaluationCase,
    RAGEvaluationResult,
)
from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.reranking.model import RerankingModel
from enterprise_ai.evaluation.rag_metrics import (
    abstention_score,
    answer_contains_expected_terms,
)
from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.rag.service import answer_question


def evaluate_rag(
    cases: list[RAGEvaluationCase],
    llm: LocalLLM,
    embedding_model: EmbeddingModel,
    reranking_model: RerankingModel,
) -> RAGEvaluationResult:
    """Evaluate RAG answer correctness and abstention behavior."""

    answer_scores = []
    abstention_scores = []
    case_results = []

    for case in cases:
        result = answer_question(
            question=case.question,
            llm=llm,
            embedding_model=embedding_model,
            reranking_model=reranking_model,
        )

        if case.should_abstain:
            answer_score = 0.0
        else:
            answer_score = answer_contains_expected_terms(
                answer=result.answer,
                expected_terms=case.expected_answer_contains,
            )
            answer_scores.append(answer_score)

        abstain_score = abstention_score(
            answer=result.answer,
            should_abstain=case.should_abstain,
        )

        abstention_scores.append(abstain_score)

        case_results.append(
            RAGCaseResult(
                question=case.question,
                answer=result.answer,
                should_abstain=case.should_abstain,
                answer_correctness=answer_score,
                abstention_correctness=abstain_score,
            )
        )

    mean_answer_correctness = (
        sum(answer_scores) / len(answer_scores)
        if answer_scores
        else 0.0
    )

    mean_abstention_correctness = (
        sum(abstention_scores) / len(abstention_scores)
        if abstention_scores
        else 0.0
    )

    return RAGEvaluationResult(
        answer_correctness=mean_answer_correctness,
        abstention_correctness=mean_abstention_correctness,
        case_results=case_results,
    )