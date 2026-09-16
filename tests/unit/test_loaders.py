from pathlib import Path

import pandas as pd
import pytest

from enterprise_ai.ingestion.loaders import (
    load_csv,
    load_json,
)


def test_load_csv(tmp_path: Path):
    """CSV loader should return the expected DataFrame."""

    test_file = tmp_path / "test.csv"

    expected = pd.DataFrame(
        {
            "customer_id": [
                "CUST_000001",
                "CUST_000002",
            ],
            "country": [
                "Spain",
                "Germany",
            ],
        }
    )

    expected.to_csv(
        test_file,
        index=False,
    )

    result = load_csv(test_file)

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_load_json(tmp_path: Path):
    """JSON loader should return the expected DataFrame."""

    test_file = tmp_path / "test.json"

    expected = pd.DataFrame(
        {
            "ticket_id": [
                "TKT_0000001",
                "TKT_0000002",
            ],
            "category": [
                "Billing",
                "Technical",
            ],
        }
    )

    expected.to_json(
        test_file,
        orient="records",
    )

    result = load_json(test_file)

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_load_csv_missing_file(tmp_path: Path):
    """CSV loader should fail clearly when a file is missing."""

    missing_file = (
        tmp_path / "does_not_exist.csv"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_csv(missing_file)


def test_load_json_missing_file(tmp_path: Path):
    """JSON loader should fail clearly when a file is missing."""

    missing_file = (
        tmp_path / "does_not_exist.json"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_json(missing_file)