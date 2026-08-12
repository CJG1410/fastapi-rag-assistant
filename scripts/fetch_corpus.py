from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"


SOURCES = {
    "routing.md": {
        "title": "FastAPI First Steps",
        "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
    },
    "request_body.md": {
        "title": "FastAPI Request Body",
        "url": "https://fastapi.tiangolo.com/tutorial/body/",
    },
    "dependencies.md": {
        "title": "FastAPI Dependencies",
        "url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
    },
    "response_models.md": {
        "title": "FastAPI Response Model",
        "url": "https://fastapi.tiangolo.com/tutorial/response-model/",
    },
    "error_handling.md": {
        "title": "FastAPI Handling Errors",
        "url": "https://fastapi.tiangolo.com/tutorial/handling-errors/",
    },
}


def fetch_document(url: str) -> str:
    """Fetch and convert the main FastAPI documentation content to Markdown."""

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    main_content = soup.find("main")

    if main_content is None:
        raise RuntimeError(f"Could not find main content at {url}")

    # Remove elements that are not useful for our RAG corpus.
    for element in main_content.select(
        "script, style, nav, footer, aside"
    ):
        element.decompose()

    markdown = md(
        str(main_content),
        heading_style="ATX",
        bullets="-",
        code_language_callback=lambda el: el.get("class", [""])[0].replace(
            "language-", ""
        )
        if el.get("class")
        else None,
    )

    return markdown.strip()


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, metadata in SOURCES.items():
        print(f"Fetching: {metadata['title']}")

        try:
            content = fetch_document(metadata["url"])

            document = f"""---
title: "{metadata['title']}"
source: "{filename}"
url: "{metadata['url']}"
---

{content}
"""

            output_path = DOCS_DIR / filename
            output_path.write_text(document, encoding="utf-8")

            print(f"  ✓ Saved: {output_path}")

        except Exception as exc:
            print(f"  ✗ Failed: {exc}")

    print("\nCorpus fetch completed.")


if __name__ == "__main__":
    main()