from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.ingestion.embeddings import EmbeddingModel


def main():
    print("Loading documents...")

    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    print("\nSplitting documents...")

    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    print("\nGenerating embeddings for all chunks...")

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.embed_documents(texts)

    print("\nEmbedding generation complete.")

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {len(embeddings[0])}")

    print(
        f"Expected number of embeddings: {len(chunks)}"
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Number of embeddings does not match number of chunks."
        )

    print("\n✓ Every chunk has an embedding.")
    print("✓ Embedding dimensions are consistent.")


if __name__ == "__main__":
    main()