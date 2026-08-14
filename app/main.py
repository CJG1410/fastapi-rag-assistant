from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.graph.workflow import build_graph
from app.services.memory import ConversationMemory


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="FastAPI Technical Documentation Assistant",
    description=(
        "A self-corrective RAG assistant for "
        "technical documentation."
    ),
    version="1.0.0",
)


# =========================================================
# LANGGRAPH
# =========================================================

graph = build_graph()


# =========================================================
# CONVERSATION MEMORY
# =========================================================

memory = ConversationMemory()


# =========================================================
# REQUEST MODEL
# =========================================================

class AskRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Technical question to answer.",
    )

    session_id: str = Field(
        default="default",
        min_length=1,
        description=(
            "Conversation session identifier. "
            "Questions using the same session_id "
            "share conversation history."
        ),
    )


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "service": "fastapi-rag-assistant",
    }


# =========================================================
# ASK ENDPOINT
# =========================================================

@app.post("/ask")
async def ask_question(
    request: AskRequest,
):

    question = request.question

    session_id = request.session_id

    # -----------------------------------------------------
    # Get previous conversation history
    # -----------------------------------------------------

    history = memory.get_history(
        session_id
    )

    # -----------------------------------------------------
    # Store current user message
    # -----------------------------------------------------

    memory.add_user_message(
        session_id,
        question,
    )

    # -----------------------------------------------------
    # Initialize LangGraph state
    # -----------------------------------------------------

    initial_state = {

        # User question
        "question": question,

        # Initial retrieval query
        "current_query": question,

        # Local retrieval
        "documents": [],
        "relevant_documents": [],

        # Web search
        "web_results": [],

        # Retrieval retry configuration
        "retry_count": 0,
        "max_retries": 2,

        # Answer
        "answer": "",

        # Sources
        "sources": [],
        "source_type": "",

        # Retrieval status
        "documents_relevant": False,
        "web_search_used": False,

        # Hallucination verification
        "hallucination_checked": False,
        "answer_supported": False,
        "verification_reason": "",

        # Answer regeneration
        "generation_retry_count": 0,
        "max_generation_retries": 1,

        # Conversation memory
        "conversation_history": history,
    }

    # -----------------------------------------------------
    # Execute LangGraph
    # -----------------------------------------------------

    result = graph.invoke(
        initial_state
    )

    # -----------------------------------------------------
    # Store assistant response
    # -----------------------------------------------------

    memory.add_assistant_message(
        session_id,
        result["answer"],
    )

    # -----------------------------------------------------
    # Return API response
    # -----------------------------------------------------

    return {
        "question": question,

        "answer": result["answer"],

        "sources": result["sources"],

        "source_type": result["source_type"],

        "retry_count": result["retry_count"],

        "web_search_used": result[
            "web_search_used"
        ],

        "session_id": session_id,

        "hallucination_checked": result[
            "hallucination_checked"
        ],

        "answer_supported": result[
            "answer_supported"
        ],

        "verification_reason": result[
            "verification_reason"
        ],
    }


# =========================================================
# CLEAR CONVERSATION
# =========================================================

@app.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
):

    memory.clear_session(
        session_id
    )

    return {
        "message": "Conversation session cleared.",
        "session_id": session_id,
    }