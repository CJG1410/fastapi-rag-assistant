from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wrapper around the Sentence Transformers embedding model."""

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded successfully.")

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()