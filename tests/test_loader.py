from app.ingestion.loader import load_documents


def main():
    documents = load_documents()

    print(f"\nLoaded documents: {len(documents)}\n")

    for document in documents:
        print("=" * 60)
        print(f"Source: {document.metadata.get('source')}")
        print(f"Title:  {document.metadata.get('title')}")
        print(f"URL:    {document.metadata.get('url')}")
        print(f"Characters: {len(document.page_content)}")
        print()


if __name__ == "__main__":
    main()