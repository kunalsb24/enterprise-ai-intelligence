def is_section_heading(paragraph: str) -> bool:
    """Return True when a paragraph looks like a section heading."""

    return (
        "\n" not in paragraph
        and len(paragraph) <= 50
        and not paragraph.endswith((".", "!", "?", ":"))
    )


def combine_headings_with_content(
    paragraphs: list[str],
) -> list[str]:
    """Attach section headings to the paragraph that follows them."""

    combined = []
    index = 0

    while index < len(paragraphs):
        paragraph = paragraphs[index]

        if (
            is_section_heading(paragraph)
            and index + 1 < len(paragraphs)
        ):
            combined.append(
                f"{paragraph}\n\n{paragraphs[index + 1]}"
            )
            index += 2
        else:
            combined.append(paragraph)
            index += 1

    return combined


def chunk_text(
    text: str,
    max_characters: int = 500,
) -> list[str]:
    """Split text into paragraph-aware chunks."""

    if max_characters <= 0:
        raise ValueError(
            "max_characters must be greater than zero."
        )

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    paragraphs = combine_headings_with_content(
        paragraphs
    )

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = (
            f"{current_chunk}\n\n{paragraph}"
            if current_chunk
            else paragraph
        )

        if len(candidate) <= max_characters:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks