from app.services.retriever import Retriever


def print_results(
    query: str,
    results: list[dict],
):
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for index, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\n--- Result {index} ---")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Source:   {metadata.get('source')}")
        print(f"Title:    {metadata.get('title')}")
        print(f"Type:     {metadata.get('content_type')}")

        print("\nContent:")
        print(result["content"][:500])


def main():

    print("Initializing retriever...")

    retriever = Retriever()

    queries = [
        "How do I use Depends for dependency injection in FastAPI?",
        "How do I define a response model in FastAPI?",
        "How do I raise an HTTPException in FastAPI?",
        "How do I configure PostgreSQL replication and database clustering?",
    ]

    for query in queries:

        results = retriever.retrieve(
            query=query,
            top_k=5,
        )

        print_results(
            query,
            results,
        )


if __name__ == "__main__":
    main()