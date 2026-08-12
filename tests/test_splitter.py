from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents


def main():
    print("Loading documents...")

    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    print("\nSplitting documents...")

    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    print("\n" + "=" * 70)
    print("FIRST 5 CHUNKS")
    print("=" * 70)

    for index, chunk in enumerate(chunks[:5], start=1):
        print(f"\n--- Chunk {index} ---")

        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Title:  {chunk.metadata.get('title')}")
        print(f"URL:    {chunk.metadata.get('url')}")
        print(f"Size:   {len(chunk.page_content)} characters")
        print(f"Type:   {chunk.metadata.get('content_type')}")

        print("\nContent:")
        print(chunk.page_content[:1000])

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()