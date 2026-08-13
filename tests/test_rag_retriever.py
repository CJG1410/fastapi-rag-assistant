from app.services.rag_retriever import RAGRetriever


def main():

    print("=" * 80)
    print("SELF-CORRECTIVE RAG RETRIEVAL TEST")
    print("=" * 80)

    rag_retriever = RAGRetriever()

    # --------------------------------------------------
    # Test 1 — In-domain question
    # --------------------------------------------------

    query = (
        "How do I use Depends for dependency injection "
        "in FastAPI?"
    )

    print("\n" + "=" * 80)
    print("TEST 1 — IN-DOMAIN QUERY")
    print("=" * 80)

    result = rag_retriever.retrieve_with_retry(
        query
    )

    print("\nSuccess:", result["success"])
    print("Original query:", result["original_query"])
    print("Final query:", result["final_query"])
    print("Retry count:", result["retry_count"])
    print("Relevant documents:", len(result["results"]))

    for index, document in enumerate(
        result["results"],
        start=1,
    ):
        print(f"\n--- Relevant Result {index} ---")
        print(
            f"Source: "
            f"{document['metadata'].get('source')}"
        )
        print(
            f"Distance: "
            f"{document['distance']:.4f}"
        )
        print(
            f"Reason: "
            f"{document.get('grade_reason')}"
        )

    # --------------------------------------------------
    # Test 2 — Out-of-domain question
    # --------------------------------------------------

    query = (
        "How do I configure PostgreSQL replication "
        "and database clustering?"
    )

    print("\n" + "=" * 80)
    print("TEST 2 — OUT-OF-DOMAIN QUERY")
    print("=" * 80)

    result = rag_retriever.retrieve_with_retry(
        query
    )

    print("\nSuccess:", result["success"])
    print("Original query:", result["original_query"])
    print("Final query:", result["final_query"])
    print("Retry count:", result["retry_count"])
    print("Relevant documents:", len(result["results"]))


if __name__ == "__main__":
    main()