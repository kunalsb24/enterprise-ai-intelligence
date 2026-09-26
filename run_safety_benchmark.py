from enterprise_ai.evaluation.safety import (
    evaluate_safety_case,
    evaluate_sql_safety,
    is_sql_blocked,
    load_safety_cases,
)
from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.sql_agent.generator import generate_sql

from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.orchestration.graph import build_router_graph
from enterprise_ai.reranking.model import RerankingModel
from enterprise_ai.sql_agent.connection import get_sql_agent_engine


BENCHMARK_PATH = "data/evaluation/agent_safety_questions.json"

def run_safety_case(case, llm, graph):
    """Run one Responsible AI safety evaluation case."""

    if case.category == "sql_safety":
        generated_sql = generate_sql(
            question=case.question,
            llm=llm,
        )

        blocked = is_sql_blocked(generated_sql)

        result = evaluate_sql_safety(
            case=case,
            generated_sql=generated_sql,
            blocked=blocked,
        )

        return result

    graph_result = graph.invoke(
    {"question": case.question}
)

    print(f"  actual_route={graph_result['route']}")
    print(f"  final_answer={graph_result['final_answer']}")

    result = evaluate_safety_case(
        case=case,
        actual_route=graph_result["route"],
        final_answer=graph_result["final_answer"],
    )

    return result

def main():
    print("Loading Responsible AI safety benchmark...")

    cases = load_safety_cases(BENCHMARK_PATH)

    print(f"Loaded {len(cases)} safety cases.")
    print("Loading local LLM...")

    llm = LocalLLM()

    print("Loading embedding and reranking models...")

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

    print()
    print("Running Responsible AI safety benchmark...")
    print()

    results = []

    for case in cases:
        print(f"Running {case.id} ({case.category})...")

        result = run_safety_case(
            case=case,
            llm=llm,
            graph=graph,
        )

        results.append(result)

        status = "PASS" if result.passed else "FAIL"

        print(
            f"{case.id} | "
            f"category={case.category:<22} | "
            f"{status}"
        )

        if not result.passed:
            print(f"  route_correct={result.route_correct}")
            print(
                f"  required_terms_present="
                f"{result.required_terms_present}"
            )
            print(
                f"  forbidden_phrases_absent="
                f"{result.forbidden_phrases_absent}"
            )
            print(
                f"  abstention_correct="
                f"{result.abstention_correct}"
            )
            print(
                f"  sql_safety_correct="
                f"{result.sql_safety_correct}"
            )

    print()
    print("Safety benchmark complete.")


if __name__ == "__main__":
    main()