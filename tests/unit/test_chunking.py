import pytest

from enterprise_ai.document_processing.chunking import chunk_text


def test_short_text_stays_in_one_chunk():
    """Short text should remain in a single chunk."""

    text = "Overview\n\nThis is a short document."

    chunks = chunk_text(text, max_characters=100)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_paragraphs_are_split_into_multiple_chunks():
    """Paragraphs should be grouped without exceeding the size target."""

    text = (
        "First paragraph."
        "\n\n"
        "Second paragraph."
        "\n\n"
        "Third paragraph."
    )

    chunks = chunk_text(text, max_characters=35)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk) <= 35


def test_empty_text_returns_no_chunks():
    """Empty text should produce an empty chunk list."""

    chunks = chunk_text("")

    assert chunks == []


def test_invalid_max_characters_raises_error():
    """Chunk size must be greater than zero."""

    with pytest.raises(
        ValueError,
        match="max_characters must be greater than zero",
    ):
        chunk_text("Example text", max_characters=0)


def test_section_heading_stays_with_following_content():
    """A section heading should stay with the paragraph it describes."""

    text = (
        "Overview\n\n"
        "A short overview of the incident.\n\n"
        "Technical Findings\n\n"
        "Configuration inconsistencies caused payment failures."
    )

    chunks = chunk_text(
        text,
        max_characters=70,
    )

    technical_chunk = next(
        chunk
        for chunk in chunks
        if "Technical Findings" in chunk
    )

    assert (
        "Configuration inconsistencies caused payment failures."
        in technical_chunk
    )