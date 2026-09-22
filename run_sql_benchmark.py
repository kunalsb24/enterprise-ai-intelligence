from pathlib import Path

from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.sql_agent.connection import get_sql_agent_engine
from enterprise_ai.sql_agent.evaluation import (
    evaluate_benchmark,
    load_evaluation_cases,
)


def main() -> None:
    evaluation_path = Path(
        "data/evaluation/sql_agent_questions.json"
    )

    print("Loading evaluation cases...")
    cases = load_evaluation_cases(
        evaluation_path
    )

    print(f"Loaded {len(cases)} cases.")

    print("\nLoading local LLM...")
    llm = LocalLLM()

    print("Connecting SQL Agent to PostgreSQL...")
    engine = get_sql_agent_engine()

    print("\nRunning SQL Agent benchmark...\n")

    benchmark = evaluate_benchmark(
        cases=cases,
        llm=llm,
        engine=engine,
    )

    for result in benchmark["results"]:
        print("=" * 70)
        print(f"ID: {result['id']}")
        print(f"Question: {result['question']}")
        print(f"Generated SQL: {result['sql']}")
        print(f"Expected: {result['expected_value']}")
        print(f"Actual: {result['actual_value']}")
        print(f"Correct: {result['correct']}")

        if result["error"]:
            print(f"Error: {result['error']}")

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        f"Correct: "
        f"{benchmark['correct_cases']}/"
        f"{benchmark['total_cases']}"
    )
    print(
        f"Execution accuracy: "
        f"{benchmark['accuracy']:.2%}"
    )


if __name__ == "__main__":
    main()