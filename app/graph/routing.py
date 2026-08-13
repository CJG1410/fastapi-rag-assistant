from app.graph.state import GraphState


def route_after_grading(
    state: GraphState,
) -> str:
    """
    Decide what happens after document grading.

    Returns:

        "generate"
            Relevant local documents found.

        "rewrite_query"
            No relevant documents and retries remain.

        "web_search"
            No relevant documents and retry limit reached.
    """

    # --------------------------------------------------
    # Local documentation is relevant
    # --------------------------------------------------

    if state["documents_relevant"]:
        return "generate"

    # --------------------------------------------------
    # We still have retries available
    # --------------------------------------------------

    if state["retry_count"] < state["max_retries"]:
        return "rewrite_query"

    # --------------------------------------------------
    # Local retrieval failed completely
    # --------------------------------------------------

    return "web_search"