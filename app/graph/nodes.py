from app.graph.state import GraphState
from app.services.gemini import GeminiService
from app.services.grader import DocumentGrader
from app.services.query_rewriter import QueryRewriter
from app.services.retriever import Retriever


# ---------------------------------------------------------
# Services
# ---------------------------------------------------------

retriever = Retriever()
grader = DocumentGrader()
query_rewriter = QueryRewriter()
gemini = GeminiService()


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

    rewritten = query_rewriter.rewrite(
        original_query=current_query,
        failed_documents=failed_documents,
    )

    print(
        f"Original query: {current_query}"
    )

    print(
        f"Rewritten query: {rewritten.query}"
    )

    return {
        **state,
        "current_query": rewritten.query,
        "retry_count": state["retry_count"] + 1,
    }


# ---------------------------------------------------------
# Node 4 — Answer Generation
# ---------------------------------------------------------

def generate_answer(state: GraphState) -> GraphState:
    """Generate a grounded answer using relevant documents."""

    question = state["question"]
    relevant_documents = state["relevant_documents"]

    print("\n" + "=" * 70)
    print("NODE: GENERATE ANSWER")
    print("=" * 70)

    # -----------------------------------------------------
    # Build grounded context
    # -----------------------------------------------------

    context_parts = []

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

        context_parts.append(
            f"""
DOCUMENT {index}
Source: {source}
Title: {title}

Content:
{document["content"]}
"""
        )

    context = "\n\n".join(context_parts)

    # -----------------------------------------------------
    # Generation prompt
    # -----------------------------------------------------

    prompt = f"""
You are a technical documentation assistant.

Answer the user's question using ONLY the provided
technical documentation.

USER QUESTION:
{question}

DOCUMENTATION CONTEXT:
{context}

Rules:

1. Answer the question directly.
2. Use only information supported by the documentation.
3. Do not invent APIs, parameters, behavior, or examples.
4. If the documentation contains a relevant code example,
   you may include it.
5. If the documentation does not contain enough information,
   clearly say so.
6. Keep the answer concise but technically useful.
7. Do not mention the retrieval system, document grader,
   LangGraph, or internal processing.
8. Mention the relevant source document when appropriate.

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
    }