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
        "retry_count": 0,
        "max_retries": 2,
        "answer": "",
        "documents_relevant": False,
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
        f"Relevant documents: "
        f"{len(result['relevant_documents'])}"
    )

    print(
        f"Documents relevant: "
        f"{result['documents_relevant']}"
    )

    # --------------------------------------------------
    # Display generated answer
    # --------------------------------------------------

    if result["answer"]:

        print("\n" + "=" * 80)
        print("GENERATED ANSWER")
        print("=" * 80)

        print(result["answer"])

    else:

        print("\nNo answer was generated.")


def main():

    print("=" * 80)
    print("LANGGRAPH SELF-CORRECTIVE RAG TEST")
    print("=" * 80)

    graph = build_graph()

    # --------------------------------------------------
    # Test 1 — In-domain question
    # --------------------------------------------------

    run_test(
        graph,
        "How do I use Depends for dependency injection in FastAPI?",
    )

    # --------------------------------------------------
    # Test 2 — Out-of-domain question
    # --------------------------------------------------

    run_test(
        graph,
        "How do I configure PostgreSQL replication and database clustering?",
    )


if __name__ == "__main__":
    main()