import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def clean_markdown(text: str) -> str:
    """Clean documentation artifacts from Markdown."""

    # Remove FastAPI permanent-link anchors.
    text = re.sub(
        r"\[¶\]\([^)]*?\)",
        "",
        text,
    )

    # Normalize excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def split_section(
    section: Document,
    prose_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """
    Split a documentation section while keeping fenced code blocks
    as independent, intact chunks.
    """

    text = section.page_content

    # Find fenced code blocks.
    code_pattern = re.compile(
        r"```[\s\S]*?```",
        re.MULTILINE,
    )

    parts = []
    last_end = 0

    for match in code_pattern.finditer(text):

        # Text before the code block.
        prose_before = text[last_end:match.start()].strip()

        if prose_before:
            parts.append(("prose", prose_before))

        # Keep the complete code block together.
        parts.append(("code", match.group(0).strip()))

        last_end = match.end()

    # Remaining prose after the final code block.
    prose_after = text[last_end:].strip()

    if prose_after:
        parts.append(("prose", prose_after))

    chunks = []

    for part_type, content in parts:

        if part_type == "code":
            # Keep the complete code example together.
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        **section.metadata,
                        "content_type": "code",
                    },
                )
            )

        else:
            # Split prose normally.
            prose_chunks = prose_splitter.split_text(content)

            for chunk in prose_chunks:
                chunks.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **section.metadata,
                            "content_type": "text",
                        },
                    )
                )

    return chunks


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split technical Markdown documents into retrieval-friendly chunks.
    """

    final_chunks = []

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )

    prose_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    for document in documents:

        cleaned_content = clean_markdown(
            document.page_content
        )

        sections = markdown_splitter.split_text(
            cleaned_content
        )

        for section in sections:

            metadata = {
                **document.metadata,
                **section.metadata,
            }

            section_document = Document(
                page_content=section.page_content,
                metadata=metadata,
            )

            section_chunks = split_section(
                section_document,
                prose_splitter,
            )

            final_chunks.extend(section_chunks)

    return final_chunks