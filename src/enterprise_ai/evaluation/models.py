from pydantic import BaseModel


class RetrievalEvaluationCase(BaseModel):
    """One question and its expected relevant document."""

    question: str
    expected_document: str

class RetrievalCaseResult(BaseModel):
    """Evaluation result for one retrieval question."""

    question: str
    expected_document: str
    retrieved_documents: list[str]
    recall_at_k: float
    reciprocal_rank: float

class RetrievalEvaluationResult(BaseModel):
    """Summary metrics and per-question results from retrieval evaluation."""

    recall_at_k: float
    mean_reciprocal_rank: float
    case_results: list[RetrievalCaseResult]

class RAGEvaluationCase(BaseModel):
    question: str
    expected_answer_contains: list[str]
    should_abstain: bool = False

class RAGCaseResult(BaseModel):
    question: str
    answer: str
    should_abstain: bool
    answer_correctness: float
    abstention_correctness: float


class RAGEvaluationResult(BaseModel):
    answer_correctness: float
    abstention_correctness: float
    case_results: list[RAGCaseResult]