from app.services.tavily_search import TavilySearchService


def main():
    print("=" * 80)
    print("TAVILY WEB SEARCH TEST")
    print("=" * 80)

    search_service = TavilySearchService()

    query = (
        "PostgreSQL streaming replication "
        "high availability configuration"
    )

    print(f"\nQuery: {query}")
    print("\nSearching web...\n")

    results = search_service.search(
        query=query,
        max_results=5,
    )

    print(f"Results returned: {len(results)}")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 80)
        print(f"RESULT {index}")
        print("-" * 80)

        print(
            f"Title: "
            f"{result.get('title', 'N/A')}"
        )

        print(
            f"URL: "
            f"{result.get('url', 'N/A')}"
        )

        print(
            f"Content:\n"
            f"{result.get('content', 'N/A')[:500]}"
        )


if __name__ == "__main__":
    main()