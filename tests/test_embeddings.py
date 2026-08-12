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

    print("\nGenerating embedding for the first chunk...")

    embedding = embedding_model.embed_text(
        chunks[0].page_content
    )

    print(f"Embedding dimensions: {len(embedding)}")

    print("\nFirst 10 values:")

    print(embedding[:10])


if __name__ == "__main__":
    main()