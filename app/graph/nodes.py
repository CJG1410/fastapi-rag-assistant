from pydantic import BaseModel

from app.graph.state import GraphState

from app.services.gemini import GeminiService
from app.services.grader import DocumentGrader
from app.services.query_rewriter import QueryRewriter
from app.services.retriever import Retriever
from app.services.tavily_search import TavilySearchService
from app.services.hallucination_checker import HallucinationChecker


# =========================================================
# STRUCTURED RESULT FOR CONVERSATION QUERY PREPARATION
# =========================================================

class PreparedQuery(BaseModel):
    """Structured result from conversation-aware query preparation."""

    query: str
    reason: str


# =========================================================
# SERVICES
# =========================================================

retriever = Retriever()
grader = DocumentGrader()
query_rewriter = QueryRewriter()
gemini = GeminiService()
tavily = TavilySearchService()
hallucination_checker = HallucinationChecker()


# =========================================================
# CONFIGURATION
# =========================================================

TOP_K = 5

DISTANCE_THRESHOLD = 1.0

MAX_RETRIES = 2

MAX_GENERATION_RETRIES = 1


# =========================================================
# NODE 0 — CONVERSATION-AWARE QUERY PREPARATION
# =========================================================

def prepare_query(state: GraphState) -> GraphState:
    """
    Convert the user's current question into a
    retrieval-ready query using conversation history.

    If there is no previous conversation, the original
    question is preserved.
    """

    question = state["question"]

    conversation_history = state.get(
        "conversation_history",
        [],
    )

    print("\n" + "=" * 70)
    print("NODE: PREPARE CONVERSATION-AWARE QUERY")
    print("=" * 70)

    print(f"Original question: {question}")

    # -----------------------------------------------------
    # No previous conversation
    # -----------------------------------------------------

    if not conversation_history:

        print(
            "No previous conversation found."
        )

        print(
            "Using original question as retrieval query."
        )

        return {
            **state,
            "current_query": question,
        }

    # -----------------------------------------------------
    # Format conversation history
    # -----------------------------------------------------

    history_parts = []

    for message in conversation_history:

        role = message.get(
            "role",
            "unknown",
        )

        content = message.get(
            "content",
            "",
        )

        if content.strip():

            history_parts.append(
                f"{role.upper()}: {content}"
            )

    conversation_text = "\n".join(
        history_parts
    )

    # -----------------------------------------------------
    # Conversation-aware query prompt
    # -----------------------------------------------------

    prompt = f"""
You are a query preparation component in a technical
documentation RAG system.

Your task is to transform the user's current question
into a clear, self-contained search query for semantic
retrieval.

The user may ask a follow-up question that contains
ambiguous references such as:

- "it"
- "this"
- "that"
- "one"
- "they"
- "the above"
- "how do I do that?"
- "what about this?"

Use the previous conversation to resolve those references.

CURRENT USER QUESTION:
{question}

PREVIOUS CONVERSATION:
{conversation_text}

Rules:

1. Preserve the user's actual intent.

2. Use the previous conversation only when it helps
   understand the current question.

3. Resolve ambiguous references into explicit technical
   terminology.

4. Make the resulting query self-contained.

5. Do not answer the question.

6. Do not invent technical facts.

7. Do not add unrelated information.

8. If the current question is already clear, keep it
   semantically close to the original question.

9. The result should be suitable for semantic search
   against technical documentation.

Return:

- query: the retrieval-ready query
- reason: a short explanation of how the conversation
  context was used.
"""

    try:

        response = gemini.client.models.generate_content(
            model=gemini.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": (
                    PreparedQuery.model_json_schema()
                ),
            },
        )

        prepared = PreparedQuery.model_validate_json(
            response.text
        )

        print(
            f"Prepared query: {prepared.query}"
        )

        print(
            f"Reason: {prepared.reason}"
        )

        return {
            **state,
            "current_query": prepared.query,
        }

    except Exception as exc:

        print(
            "\nWarning: Conversation-aware query "
            "preparation failed."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "Using the original question."
        )

        return {
            **state,
            "current_query": question,
        }


# =========================================================
# NODE 1 — RETRIEVAL
# =========================================================

def retrieve_documents(state: GraphState) -> GraphState:
    """
    Retrieve candidate documents from ChromaDB.
    """

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


# =========================================================
# NODE 2 — DOCUMENT GRADING
# =========================================================

def grade_documents(state: GraphState) -> GraphState:
    """
    Grade retrieved documents using Gemini.
    """

    query = state["current_query"]

    documents = state["documents"]

    print("\n" + "=" * 70)
    print("NODE: GRADE DOCUMENTS")
    print("=" * 70)

    relevant_documents = []

    for document in documents:

        distance = document["distance"]

        if distance > DISTANCE_THRESHOLD:

            print(
                f"Skipping document with distance "
                f"{distance:.4f}"
            )

            continue

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

            relevant_documents.append(
                document
            )

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


# =========================================================
# NODE 3 — QUERY REWRITING
# =========================================================

def rewrite_query(state: GraphState) -> GraphState:
    """
    Rewrite the query after unsuccessful retrieval.

    This is different from prepare_query():

    prepare_query()
        -> resolves conversation context

    rewrite_query()
        -> fixes a failed retrieval attempt
    """

    current_query = state["current_query"]

    documents = state["documents"]

    conversation_history = state.get(
        "conversation_history",
        [],
    )

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
            conversation_history=conversation_history,
        )

        new_query = rewritten.query

        print(
            f"Original query: {current_query}"
        )

        print(
            f"Rewritten query: {new_query}"
        )

        print(
            f"Reason: {rewritten.reason}"
        )

    except Exception as exc:

        print(
            "\nWarning: Query rewriting failed."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "Keeping the current query."
        )

        new_query = current_query

    return {
        **state,
        "current_query": new_query,
        "retry_count": state["retry_count"] + 1,
    }


# =========================================================
# NODE 4 — TAVILY WEB SEARCH
# =========================================================

def web_search(state: GraphState) -> GraphState:
    """
    Search the web using Tavily after local retrieval fails.
    """

    query = state["current_query"]

    print("\n" + "=" * 70)
    print("NODE: TAVILY WEB SEARCH")
    print("=" * 70)

    print(
        f"Web search query: {query}"
    )

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


# =========================================================
# HELPER — BUILD GENERATION CONTEXT
# =========================================================

def _build_generation_context(
    state: GraphState,
) -> tuple[str, list[dict]]:

    relevant_documents = state[
        "relevant_documents"
    ]

    web_results = state[
        "web_results"
    ]

    web_search_used = state[
        "web_search_used"
    ]

    context_parts = []

    sources_by_url = {}

    # -----------------------------------------------------
    # LOCAL DOCUMENTATION
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

        source_key = url or source

        if source_key not in sources_by_url:

            sources_by_url[source_key] = {
                "title": title,
                "source": source,
                "url": url,
                "type": "local",
            }

    # -----------------------------------------------------
    # TAVILY RESULTS
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

            if url and url not in sources_by_url:

                sources_by_url[url] = {
                    "title": title,
                    "source": url,
                    "url": url,
                    "type": "web",
                }

    context = "\n\n".join(
        context_parts
    )

    sources = list(
        sources_by_url.values()
    )

    return context, sources


# =========================================================
# NODE 5 — ANSWER GENERATION
# =========================================================

def generate_answer(state: GraphState) -> GraphState:
    """
    Generate a grounded answer using local or web context.
    """

    question = state["question"]

    web_search_used = state[
        "web_search_used"
    ]

    print("\n" + "=" * 70)
    print("NODE: GENERATE ANSWER")
    print("=" * 70)

    context, sources = (
        _build_generation_context(state)
    )

    if web_search_used:
        source_type = "web"
    else:
        source_type = "local"

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
10. Do not include a separate Sources section.
    Sources are returned separately by the application.

Provide the final answer now.
"""

    try:

        response = gemini.client.models.generate_content(
            model=gemini.model,
            contents=prompt,
        )

        answer = response.text

    except Exception as exc:

        print(
            f"\nError during answer generation: {exc}"
        )

        answer = (
            "I was unable to generate an answer "
            "at this time."
        )

    print("\nGenerated answer:")
    print(answer)

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "source_type": source_type,
    }


# =========================================================
# NODE 6 — VERIFY GENERATED ANSWER
# =========================================================

def verify_answer(state: GraphState) -> GraphState:
    """
    Verify whether the generated answer is supported
    by the retrieved context.
    """

    question = state["question"]

    answer = state["answer"]

    print("\n" + "=" * 70)
    print("NODE: VERIFY ANSWER")
    print("=" * 70)

    context_parts = []

    for document in state[
        "relevant_documents"
    ]:

        context_parts.append(
            document["content"]
        )

    if state["web_search_used"]:

        for result in state[
            "web_results"
        ]:

            content = result.get(
                "content",
                "",
            )

            if content:

                context_parts.append(
                    content
                )

    context = "\n\n".join(
        context_parts
    )

    if not context.strip():

        print(
            "No verification context available."
        )

        return {
            **state,
            "hallucination_checked": True,
            "answer_supported": False,
            "verification_reason": (
                "No supporting context was available."
            ),
        }

    try:

        result = hallucination_checker.check(
            question=question,
            answer=answer,
            context=context,
        )

        print(
            f"Supported: {result.supported}"
        )

        print(
            f"Reason: {result.reason}"
        )

        return {
            **state,
            "hallucination_checked": True,
            "answer_supported": result.supported,
            "verification_reason": result.reason,
        }

    except Exception as exc:

        print(
            "\nWarning: Answer verification failed."
        )

        print(
            f"Reason: {exc}"
        )

        return {
            **state,
            "hallucination_checked": False,
            "answer_supported": True,
            "verification_reason": (
                "Verification service unavailable."
            ),
        }


# =========================================================
# NODE 7 — PREPARE ANSWER REGENERATION
# =========================================================

def prepare_regeneration(
    state: GraphState,
) -> GraphState:

    retry_count = (
        state["generation_retry_count"] + 1
    )

    print("\n" + "=" * 70)
    print("NODE: PREPARE ANSWER REGENERATION")
    print("=" * 70)

    print(
        f"Generation retry: "
        f"{retry_count}/"
        f"{state['max_generation_retries']}"
    )

    return {
        **state,
        "generation_retry_count": retry_count,
    }