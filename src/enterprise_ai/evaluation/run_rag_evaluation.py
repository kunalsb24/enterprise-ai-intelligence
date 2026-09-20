from pathlib import Path

from enterprise_ai.embeddings.model import EmbeddingModel
from enterprise_ai.evaluation.loaders import load_rag_evaluation_cases
from enterprise_ai.evaluation.rag_evaluator import evaluate_rag
from enterprise_ai.llm.model import LocalLLM
from enterprise_ai.reranking.model import RerankingModel


def main() -> None:
    print("Loading evaluation cases...")

    cases = load_rag_evaluation_cases(
        Path("data/evaluation/rag_questions.json")
    )

    print(f"Loaded {len(cases)} cases.")

    print("\nLoading models...")

    llm = LocalLLM()
    embedding_model = EmbeddingModel()
    reranking_model = RerankingModel()

    print("\nRunning RAG evaluation...")

    result = evaluate_rag(
        cases=cases,
        llm=llm,
        embedding_model=embedding_model,
        reranking_model=reranking_model,
    )

    print("\n==============================")
    print("RAG EVALUATION RESULTS")
    print("==============================")

    print(
        f"Answer correctness: "
        f"{result.answer_correctness:.4f}"
    )

    print(
        f"Abstention correctness: "
        f"{result.abstention_correctness:.4f}"
    )

    print("\nPer-question results:")

    for index, case_result in enumerate(
        result.case_results,
        start=1,
    ):
        print("\n------------------------------")
        print(f"Question {index}: {case_result.question}")
        print(f"Should abstain: {case_result.should_abstain}")
        print(
            f"Answer correctness: "
            f"{case_result.answer_correctness:.1f}"
        )
        print(
            f"Abstention correctness: "
            f"{case_result.abstention_correctness:.1f}"
        )
        print("Answer:")
        print(case_result.answer)


if __name__ == "__main__":
    main()