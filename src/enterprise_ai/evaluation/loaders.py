import json
from pathlib import Path

from enterprise_ai.evaluation.models import (
    RAGEvaluationCase,
    RetrievalEvaluationCase,
)

def load_retrieval_evaluation_cases(
    path: Path,
) -> list[RetrievalEvaluationCase]:
    """Load retrieval evaluation cases from a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [
        RetrievalEvaluationCase(**item)
        for item in data
    ]

def load_rag_evaluation_cases(
    path: Path,
) -> list[RAGEvaluationCase]:
    """Load RAG evaluation cases from a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [
        RAGEvaluationCase(**item)
        for item in data
    ]