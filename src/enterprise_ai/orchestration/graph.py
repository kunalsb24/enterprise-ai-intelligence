from langgraph.graph import END, START, StateGraph
from sqlalchemy.engine import Engine

from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.orchestration.nodes import (
    rag_node,
    router_node,
    select_route,
    sql_node,
)
from enterprise_ai.orchestration.state import OrchestrationState
from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.reranking.model import RerankingModel

def select_after_sql(state: OrchestrationState) -> str:
    """Decide whether SQL is the final branch or RAG should also run."""

    if state["route"] == "hybrid":
        return "rag"

    return "end"

def build_router_graph(
    llm: LocalLLM,
    engine: Engine,
    embedding_model: EmbeddingModel,
    reranking_model: RerankingModel,
):
    """Build a LangGraph with conditional SQL, RAG, and hybrid branches."""

    graph = StateGraph(OrchestrationState)

    graph.add_node(
        "router",
        lambda state: router_node(
            state=state,
            llm=llm,
        ),
    )

    graph.add_node(
    "sql",
    lambda state: sql_node(
        state=state,
        llm=llm,
        engine=engine,
    ),
)
    graph.add_node(
    "rag",
    lambda state: rag_node(
        state=state,
        llm=llm,
        embedding_model=embedding_model,
        reranking_model=reranking_model,
    ),
)
    

    graph.add_edge(
        START,
        "router",
    )

    graph.add_conditional_edges(
        "router",
        select_route,
        {
            "sql": "sql",
            "rag": "rag",
            "hybrid": "sql",
        },
    )

    graph.add_conditional_edges(
    "sql",
    select_after_sql,
    {
        "rag": "rag",
        "end": END,
    },
)
    graph.add_edge("rag", END)
    

    return graph.compile()