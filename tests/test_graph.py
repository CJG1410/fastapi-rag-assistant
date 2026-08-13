from app.graph.workflow import build_graph


def run_test(
    graph,
    question: str,
):
    print("\n" + "#" * 80)
    print(f"QUESTION: {question}")
    print("#" * 80)

    initial_state = {
        "question": question,
        "current_query": question,
        "documents": [],
        "relevant_documents": [],
        "web_results": [],
        "retry_count": 0,
        "max_retries": 2,
        "answer": "",
        "sources": [],
        "source_type": "",
        "documents_relevant": False,
        "web_search_used": False,
    }

    result = graph.invoke(
        initial_state
    )

    print("\n" + "=" * 80)
    print("FINAL GRAPH STATE")
    print("=" * 80)

    print(
        f"Original question: "
        f"{result['question']}"
    )

    print(
        f"Final query: "
        f"{result['current_query']}"
    )

    print(
        f"Retry count: "
        f"{result['retry_count']}"
    )

    print(
        f"Relevant local documents: "
        f"{len(result['relevant_documents'])}"
    )

    print(
        f"Web search used: "
        f"{result['web_search_used']}"
    )

    print(
        f"Web results: "
        f"{len(result['web_results'])}"
    )

    print(
        f"Source type: "
        f"{result['source_type']}"
    )

    print(
        f"Documents relevant: "
        f"{result['documents_relevant']}"
    )

    # --------------------------------------------------
    # Generated answer
    # --------------------------------------------------

    if result["answer"]:

        print("\n" + "=" * 80)
        print("GENERATED ANSWER")
        print("=" * 80)

        print(result["answer"])

    else:

        print("\nNo answer was generated.")

    # --------------------------------------------------
    # Sources
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    if result["sources"]:

        for index, source in enumerate(
            result["sources"],
            start=1,
        ):
            print(
                f"\n{index}. "
                f"{source['title']}"
            )

            print(
                f"   Type: "
                f"{source['type']}"
            )

            print(
                f"   URL: "
                f"{source['url']}"
            )

    else:

        print("No sources.")


def main():

    print("=" * 80)
    print("LANGGRAPH SELF-CORRECTIVE RAG + WEB FALLBACK TEST")
    print("=" * 80)

    graph = build_graph()

    # --------------------------------------------------
    # Test 1 — Local documentation
    # --------------------------------------------------

    run_test(
        graph,
        "How do I use Depends for dependency injection in FastAPI?",
    )

    # --------------------------------------------------
    # Test 2 — Web fallback
    # --------------------------------------------------

    run_test(
        graph,
        "How do I configure PostgreSQL replication and database clustering?",
    )


if __name__ == "__main__":
    main()