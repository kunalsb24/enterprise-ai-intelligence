import json

from enterprise_ai.evaluation.loaders import (
    load_retrieval_evaluation_cases,
)


def test_load_retrieval_evaluation_cases(tmp_path):
    evaluation_data = [
        {
            "question": "What caused payment failures?",
            "expected_document": "billing.txt",
        },
        {
            "question": "What network capacity was added?",
            "expected_document": "network.txt",
        },
    ]

    evaluation_file = tmp_path / "evaluation.json"

    evaluation_file.write_text(
        json.dumps(evaluation_data),
        encoding="utf-8",
    )

    cases = load_retrieval_evaluation_cases(
        evaluation_file,
    )

    assert len(cases) == 2

    assert cases[0].question == "What caused payment failures?"
    assert cases[0].expected_document == "billing.txt"

    assert cases[1].question == "What network capacity was added?"
    assert cases[1].expected_document == "network.txt"