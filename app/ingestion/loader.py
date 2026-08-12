from pathlib import Path

from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = BASE_DIR / "docs"


def load_documents() -> list[Document]:
    """Load Markdown documents from the docs directory."""

    documents = []

    for file_path in sorted(DOCS_DIR.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")

        metadata = {
            "source": file_path.name,
        }

        # Extract metadata from the YAML front matter
        lines = content.splitlines()

        if lines and lines[0].strip() == "---":
            metadata_end = None

            for index in range(1, len(lines)):
                if lines[index].strip() == "---":
                    metadata_end = index
                    break

            if metadata_end is not None:
                front_matter = lines[1:metadata_end]

                for line in front_matter:
                    if ":" in line:
                        key, value = line.split(":", 1)

                        metadata[key.strip()] = (
                            value.strip().strip('"')
                        )

                content = "\n".join(lines[metadata_end + 1:]).strip()

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents