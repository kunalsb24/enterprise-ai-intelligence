from pathlib import Path

import pytest

from enterprise_ai.document_processing.loaders import load_text_document


def test_load_text_document(tmp_path: Path):
    """A valid text document should be loaded successfully."""

    document = tmp_path / "example.txt"
    document.write_text(
        "Enterprise billing incident",
        encoding="utf-8",
    )

    result = load_text_document(document)

    assert result == "Enterprise billing incident"


def test_missing_document_raises_error(tmp_path: Path):
    """A missing document should raise FileNotFoundError."""

    missing_document = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_text_document(missing_document)


def test_unsupported_document_type_raises_error(
    tmp_path: Path,
):
    """A non-text document should currently be rejected."""

    document = tmp_path / "example.pdf"
    document.write_text(
        "Example content",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_text_document(document)