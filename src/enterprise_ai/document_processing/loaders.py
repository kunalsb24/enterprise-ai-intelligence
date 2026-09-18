from pathlib import Path


def load_text_document(path: Path) -> str:
    """Load the contents of a UTF-8 text document."""

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    if path.suffix.lower() != ".txt":
        raise ValueError(
            f"Unsupported document type: {path.suffix}"
        )

    return path.read_text(encoding="utf-8")