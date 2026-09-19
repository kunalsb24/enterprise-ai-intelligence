from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.evaluation.models import (
    RetrievalCaseResult,
    RetrievalEvaluationCase,
    RetrievalEvaluationResult,
)
from enterprise_ai.evaluation.retrieval_metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from enterprise_ai.vector_store.retrieval import (
    retrieve_chunks,
)


def evaluate_retrieval(
    cases: list[RetrievalEvaluationCase],
    k: int = 3,
) -> RetrievalEvaluationResult:
    """Evaluate retrieval quality using Recall@K and MRR."""

    recall_scores = []
    reciprocal_rank_scores = []
    case_results = []

    embedding_model = EmbeddingModel()

    for case in cases:
        retrieved_chunks = retrieve_chunks(
            case.question,
            limit=k,
            embedding_model=embedding_model,
        )

        retrieved_documents = [
            chunk.document_name
            for chunk in retrieved_chunks
        ]

        recall_score = recall_at_k(
            retrieved_documents=retrieved_documents,
            expected_document=case.expected_document,
            k=k,
        )

        rank_score = reciprocal_rank(
            retrieved_documents=retrieved_documents,
            expected_document=case.expected_document,
        )

        recall_scores.append(recall_score)
        reciprocal_rank_scores.append(rank_score)
        case_results.append(
            RetrievalCaseResult(
                question=case.question,
                expected_document=case.expected_document,
                retrieved_documents=retrieved_documents,
                recall_at_k=recall_score,
                reciprocal_rank=rank_score,
            )
        )

    return RetrievalEvaluationResult(
        recall_at_k=mean_recall_at_k(recall_scores),
        mean_reciprocal_rank=mean_reciprocal_rank(
            reciprocal_rank_scores
        ),
        case_results=case_results,
    )