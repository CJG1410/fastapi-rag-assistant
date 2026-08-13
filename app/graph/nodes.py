from app.graph.state import GraphState
from app.services.gemini import GeminiService
from app.services.grader import DocumentGrader
from app.services.query_rewriter import QueryRewriter
from app.services.retriever import Retriever
from app.services.tavily_search import TavilySearchService


# ---------------------------------------------------------
# Services
# ---------------------------------------------------------

retriever = Retriever()
grader = DocumentGrader()
query_rewriter = QueryRewriter()
gemini = GeminiService()
tavily = TavilySearchService()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TOP_K = 5

# This is only a candidate-quality gate.
# Gemini remains the final relevance judge.
DISTANCE_THRESHOLD = 1.0

MAX_RETRIES = 2


# ---------------------------------------------------------
# Node 1 — Retrieval
# ---------------------------------------------------------

def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieve candidate documents from ChromaDB."""

    query = state["current_query"]

    print("\n" + "=" * 70)
    print("NODE: RETRIEVE DOCUMENTS")
    print("=" * 70)

    print(f"Query: {query}")

    results = retriever.retrieve(
        query=query,
        top_k=TOP_K,
    )

    print(
        f"Retrieved {len(results)} candidate documents."
    )

    return {
        **state,
        "documents": results,
    }


# ---------------------------------------------------------
# Node 2 — Document Grading
# ---------------------------------------------------------

def grade_documents(state: GraphState) -> GraphState:
    """Grade retrieved documents using Gemini."""

    query = state["current_query"]
    documents = state["documents"]

    print("\n" + "=" * 70)
    print("NODE: GRADE DOCUMENTS")
    print("=" * 70)

    relevant_documents = []

    for document in documents:

        distance = document["distance"]

        # -------------------------------------------------
        # Distance quality gate
        # -------------------------------------------------

        if distance > DISTANCE_THRESHOLD:
            print(
                f"Skipping document with distance "
                f"{distance:.4f}"
            )
            continue

        # -------------------------------------------------
        # Gemini relevance grading
        # -------------------------------------------------

        grade = grader.grade(
            query=query,
            document=document["content"],
        )

        source = document["metadata"].get(
            "source",
            "unknown",
        )

        print(
            f"Source: {source} | "
            f"Distance: {distance:.4f} | "
            f"Relevant: {grade.relevant}"
        )

        if grade.relevant:
            document["grade_reason"] = grade.reason
            relevant_documents.append(document)

    documents_relevant = (
        len(relevant_documents) > 0
    )

    print(
        f"\nRelevant documents found: "
        f"{len(relevant_documents)}"
    )

    return {
        **state,
        "relevant_documents": relevant_documents,
        "documents_relevant": documents_relevant,
    }


# ---------------------------------------------------------
# Node 3 — Query Rewriting
# ---------------------------------------------------------

def rewrite_query(state: GraphState) -> GraphState:
    """Rewrite the query after unsuccessful retrieval."""

    current_query = state["current_query"]
    documents = state["documents"]

    print("\n" + "=" * 70)
    print("NODE: REWRITE QUERY")
    print("=" * 70)

    failed_documents = [
        document["content"]
        for document in documents
    ]

    try:

        rewritten = query_rewriter.rewrite(
            original_query=current_query,
            failed_documents=failed_documents,
        )

        new_query = rewritten.query

        print(
            f"Original query: {current_query}"
        )

        print(
            f"Rewritten query: {new_query}"
        )

    except Exception as exc:

        print(
            "\nWarning: Query rewriting failed."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "Keeping the current query and "
            "continuing the retry cycle."
        )

        new_query = current_query

    return {
        **state,
        "current_query": new_query,
        "retry_count": state["retry_count"] + 1,
    }

# ---------------------------------------------------------
# Node 4 — Web Search
# ---------------------------------------------------------

def web_search(state: GraphState) -> GraphState:
    """Search the web using Tavily when local retrieval fails."""

    query = state["current_query"]

    print("\n" + "=" * 70)
    print("NODE: TAVILY WEB SEARCH")
    print("=" * 70)

    print(f"Web search query: {query}")

    results = tavily.search(
        query=query,
        max_results=5,
    )

    print(
        f"Tavily returned {len(results)} results."
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\nResult {index}: "
            f"{result.get('title', 'Unknown')}"
        )

        print(
            f"URL: "
            f"{result.get('url', 'Unknown')}"
        )

    return {
        **state,
        "web_results": results,
        "web_search_used": True,
    }

# ---------------------------------------------------------
# Node 5 — Answer Generation
# ---------------------------------------------------------

def generate_answer(state: GraphState) -> GraphState:
    """Generate a grounded answer using local or web context."""

    question = state["question"]

    relevant_documents = state["relevant_documents"]

    web_results = state["web_results"]

    web_search_used = state["web_search_used"]

    print("\n" + "=" * 70)
    print("NODE: GENERATE ANSWER")
    print("=" * 70)

    context_parts = []
    sources = []

    # -----------------------------------------------------
    # Local documentation
    # -----------------------------------------------------

    for index, document in enumerate(
        relevant_documents,
        start=1,
    ):

        metadata = document["metadata"]

        source = metadata.get(
            "source",
            "unknown",
        )

        title = metadata.get(
            "title",
            "unknown",
        )

        url = metadata.get(
            "url",
            "",
        )

        context_parts.append(
            f"""
LOCAL DOCUMENT {index}

Source: {source}
Title: {title}
URL: {url}

Content:
{document["content"]}
"""
        )

        sources.append(
            {
                "title": title,
                "source": source,
                "url": url,
                "type": "local",
            }
        )

    # -----------------------------------------------------
    # Tavily web results
    # -----------------------------------------------------

    if web_search_used:

        for index, result in enumerate(
            web_results,
            start=1,
        ):

            title = result.get(
                "title",
                "Unknown",
            )

            url = result.get(
                "url",
                "",
            )

            content = result.get(
                "content",
                "",
            )

            context_parts.append(
                f"""
WEB RESULT {index}

Title: {title}
URL: {url}

Content:
{content}
"""
            )

            sources.append(
                {
                    "title": title,
                    "source": url,
                    "url": url,
                    "type": "web",
                }
            )

    context = "\n\n".join(context_parts)

    # -----------------------------------------------------
    # Source type
    # -----------------------------------------------------

    if web_search_used:
        source_type = "web"
    else:
        source_type = "local"

    # -----------------------------------------------------
    # Generation prompt
    # -----------------------------------------------------

    prompt = f"""
You are a technical documentation assistant.

Answer the user's question using ONLY the provided context.

USER QUESTION:
{question}

CONTEXT:
{context}

Rules:

1. Answer the question directly.
2. Do not invent technical facts.
3. Do not invent APIs, parameters, commands, or behavior.
4. If local documentation is available, prefer it.
5. If web results are provided, use them only because the
   local documentation was insufficient.
6. If the context is insufficient, clearly say so.
7. Keep the answer technically accurate and concise.
8. Do not mention LangGraph, document grading, retrieval,
   or internal workflow details.
9. Do not create or invent source URLs.
10. Do not include a separate Sources section in the answer.
    Sources will be returned separately by the application.

Provide the final answer now.
"""

    response = gemini.client.models.generate_content(
        model=gemini.model,
        contents=prompt,
    )

    answer = response.text

    print("\nGenerated answer:")
    print(answer)

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "source_type": source_type,
    }