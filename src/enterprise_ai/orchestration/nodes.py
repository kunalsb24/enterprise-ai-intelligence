from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.orchestration.router import decide_route
from enterprise_ai.orchestration.state import OrchestrationState
from sqlalchemy.engine import Engine
from enterprise_ai.sql_agent.service import answer_sql_question
from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.rag.service import answer_question
from enterprise_ai.reranking.model import RerankingModel

def router_node(
    state: OrchestrationState,
    llm: LocalLLM,
) -> OrchestrationState:
    """Decide which capability should handle the user's question."""

    question = state["question"]

    route = decide_route(
        question=question,
        llm=llm,
    )

    return {
        "route": route,
    }

def sql_node(
    state: OrchestrationState,
    llm: LocalLLM,
    engine: Engine,
) -> OrchestrationState:
    """Answer the question using the SQL analytics agent."""

    question = state["question"]

    result = answer_sql_question(
        question=question,
        llm=llm,
        engine=engine,
    )

    return {
        "sql_evidence": result,
    }


def rag_node(
    state: OrchestrationState,
    llm: LocalLLM,
    embedding_model: EmbeddingModel,
    reranking_model: RerankingModel,
) -> OrchestrationState:
    """Answer the question using the grounded RAG pipeline."""

    question = state["question"]

    result = answer_question(
        question=question,
        llm=llm,
        embedding_model=embedding_model,
        reranking_model=reranking_model,
    )

    return {
        "rag_evidence": result,
    }




def select_route(
    state: OrchestrationState,
) -> str:
    """Return the route selected by the router node."""

    return state["route"]