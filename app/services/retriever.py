from app.ingestion.embeddings import EmbeddingModel
from app.ingestion.vectorstore import ChromaVectorStore


class Retriever:
    """Retrieve relevant chunks from the FastAPI documentation."""

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = ChromaVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Embed the query and retrieve similar chunks."""

        query_embedding = self.embedding_model.embed_text(
            query
        )

        return self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
        )