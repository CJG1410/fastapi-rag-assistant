from app.services.query_rewriter import QueryRewriter


def main():
    print("Initializing Query Rewriter...")

    rewriter = QueryRewriter()

    query = (
        "How do I configure PostgreSQL replication "
        "and database clustering?"
    )

    failed_documents = [
        """
        FastAPI has a powerful Dependency Injection system.
        You can use Depends to declare dependencies in
        path operation functions.
        """,
        """
        FastAPI response models define the shape of data
        returned by an API.
        """,
        """
        FastAPI provides default exception handlers for
        HTTPException and validation errors.
        """,
    ]

    print("\nOriginal query:")
    print(query)

    print("\nRewriting query...")

    result = rewriter.rewrite(
        original_query=query,
        failed_documents=failed_documents,
    )

    print("\n" + "=" * 70)
    print("REWRITTEN QUERY")
    print("=" * 70)

    print(f"\nQuery:  {result.query}")
    print(f"Reason: {result.reason}")


if __name__ == "__main__":
    main()