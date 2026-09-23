from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.orchestration.evaluation import (
    evaluate_router,
    load_router_cases,
    summarize_router_results,
)


BENCHMARK_PATH = "data/evaluation/router_questions.json"


def main():
    print("Loading routing benchmark...")
    cases = load_router_cases(BENCHMARK_PATH)

    print(f"Loaded {len(cases)} routing cases.")
    print("Loading local LLM...")

    llm = LocalLLM()

    print()
    print("Running router benchmark...")
    print()

    results = evaluate_router(
        cases=cases,
        llm=llm,
    )

    for result in results:
        status = "PASS" if result["correct"] else "FAIL"

        print(
            f'{result["id"]} | '
            f'expected={result["expected_route"]:<6} | '
            f'predicted={result["predicted_route"]:<6} | '
            f'{status}'
        )

    summary = summarize_router_results(results)

    print()
    print("Routing benchmark summary")
    print("-------------------------")
    print(
        f'Overall: '
        f'{summary["correct"]}/{summary["total"]} '
        f'({summary["accuracy"] * 100:.2f}%)'
    )

    for route in ("sql", "rag", "hybrid"):
        route_summary = summary["by_route"][route]

        print(
            f'{route.upper():<6}: '
            f'{route_summary["correct"]}/{route_summary["total"]} '
            f'({route_summary["accuracy"] * 100:.2f}%)'
        )


if __name__ == "__main__":
    main()