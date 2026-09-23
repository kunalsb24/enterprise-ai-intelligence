from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.orchestration.graph import build_router_graph
from enterprise_ai.reranking.model import RerankingModel
from enterprise_ai.sql_agent.connection import get_sql_agent_engine


def main():
    print("Loading shared models...")

    llm = LocalLLM()
    embedding_model = EmbeddingModel()
    reranking_model = RerankingModel()
    engine = get_sql_agent_engine()

    print("Building orchestration graph...")

    graph = build_router_graph(
        llm=llm,
        engine=engine,
        embedding_model=embedding_model,
        reranking_model=reranking_model,
    )

    questions = [
        "How many customers are there?",
        "What happened during the Q2 billing incident?",
        "Why did churn increase among German Enterprise customers in Q2?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")

        result = graph.invoke(
            {"question": question}
        )

        print(f"ROUTE: {result['route']}")

        if "sql_evidence" in result:
            print("\nSQL EVIDENCE:")
            print(result["sql_evidence"])

        if "rag_evidence" in result:
            print("\nRAG EVIDENCE:")
            print(result["rag_evidence"])


if __name__ == "__main__":
    main()