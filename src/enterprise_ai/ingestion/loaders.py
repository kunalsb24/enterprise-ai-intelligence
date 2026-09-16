from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    return pd.read_csv(path)


def load_json(path: Path) -> pd.DataFrame:
    """Load a JSON file into a DataFrame."""

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    return pd.read_json(path)