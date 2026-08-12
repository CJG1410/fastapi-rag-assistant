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

    def reset_collection(self) -> None:
        """Delete the existing collection and recreate it."""

        try:
            self.client.delete_collection(
                name=self.collection_name
            )
        except Exception:
            # Collection may not exist on the first run.
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "FastAPI technical documentation",
            },
        )

    def count(self) -> int:
        """Return the number of stored chunks."""

        return self.collection.count()

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ):
        """Search for the most similar document chunks."""

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )