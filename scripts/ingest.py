from app.ingestion.embeddings import EmbeddingModel
from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.ingestion.vectorstore import ChromaVectorStore


def main():
    print("=" * 70)
    print("FASTAPI DOCUMENT INGESTION")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------

    print("\n[1/4] Loading documents...")

    documents = load_documents()

    print(f"✓ Loaded {len(documents)} documents.")

    # --------------------------------------------------
    # 2. Split documents
    # --------------------------------------------------

    print("\n[2/4] Splitting documents...")

    chunks = split_documents(documents)

    print(f"✓ Created {len(chunks)} chunks.")

    # --------------------------------------------------
    # 3. Generate embeddings
    # --------------------------------------------------

    print("\n[3/4] Generating embeddings...")

    embedding_model = EmbeddingModel()

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.embed_documents(
        texts
    )

    print(
        f"✓ Generated {len(embeddings)} embeddings."
    )

    print(
        f"✓ Embedding dimensions: "
        f"{len(embeddings[0])}"
    )

    # --------------------------------------------------
    # 4. Store in ChromaDB
    # --------------------------------------------------

    print("\n[4/4] Storing in ChromaDB...")

    vector_store = ChromaVectorStore()

    # Reset the collection so ingestion can safely
    # be run multiple times.
    vector_store.reset_collection()

    ids = [
        f"chunk-{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        chunk.metadata
        for chunk in chunks
    ]

    vector_store.add_documents(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    stored_count = vector_store.count()

    print(
        f"✓ Stored {stored_count} chunks "
        f"in ChromaDB."
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    if stored_count != len(chunks):
        raise RuntimeError(
            f"Expected {len(chunks)} chunks, "
            f"but ChromaDB contains {stored_count}."
        )

    print("\n" + "=" * 70)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()