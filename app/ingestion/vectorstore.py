from pathlib import Path

import chromadb


BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "fastapi_docs"


class ChromaVectorStore:
    """Persistent ChromaDB vector store for the FastAPI corpus."""

    def __init__(
        self,
        persist_directory: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection_name = collection_name

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "FastAPI technical documentation",
            },
        )

    # =====================================================
    # ADD DOCUMENTS
    # =====================================================

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Add document chunks, embeddings, and metadata."""

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # =====================================================
    # RESET COLLECTION
    # =====================================================

    def reset_collection(self) -> None:
        """Delete the existing collection and recreate it."""

        try:
            self.client.delete_collection(
                name=self.collection_name
            )
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "FastAPI technical documentation",
            },
        )

    # =====================================================
    # COUNT
    # =====================================================

    def count(self) -> int:
        """Return the number of stored chunks."""

        return self.collection.count()

    # =====================================================
    # QUERY
    # =====================================================

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict]:
        """Retrieve the most similar document chunks."""

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            retrieved.append(
                {
                    "content": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return retrieved

    # =====================================================
    # LIST INDEXED DOCUMENTS
    # =====================================================

    def list_documents(self) -> list[dict]:
        """
        Return a document-level summary of the indexed corpus.

        Chroma stores chunks, so multiple chunks belonging to
        the same source are grouped together here.
        """

        total_chunks = self.collection.count()

        if total_chunks == 0:
            return []

        results = self.collection.get(
            include=[
                "metadatas",
            ]
        )

        metadatas = results.get(
            "metadatas",
            [],
        )

        documents = {}

        for metadata in metadatas:

            metadata = metadata or {}

            source = metadata.get(
                "source",
                "unknown",
            )

            if source not in documents:

                documents[source] = {
                    "source": source,
                    "title": metadata.get(
                        "title",
                        source,
                    ),
                    "url": metadata.get(
                        "url",
                        "",
                    ),
                    "chunks": 0,
                }

            documents[source]["chunks"] += 1

            # Preserve useful metadata if it appears
            # on a later chunk.
            if (
                not documents[source]["title"]
                or documents[source]["title"] == source
            ):
                documents[source]["title"] = metadata.get(
                    "title",
                    source,
                )

            if (
                not documents[source]["url"]
                and metadata.get("url")
            ):
                documents[source]["url"] = metadata[
                    "url"
                ]

        return sorted(
            documents.values(),
            key=lambda item: item["source"],
        )