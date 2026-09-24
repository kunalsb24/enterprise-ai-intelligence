from typing import Literal, TypedDict
from enterprise_ai.sql_agent.models import SQLQueryResult
from enterprise_ai.rag.models import RAGResult

Route = Literal["sql", "rag", "hybrid"]


class OrchestrationState(TypedDict, total=False):
    """Shared state passed between orchestration nodes."""

    question: str
    route: Route

    sql_evidence: SQLQueryResult
    rag_evidence: RAGResult

    draft_answer: str
    verification: str
    final_answer: str