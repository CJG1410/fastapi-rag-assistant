from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.graph.workflow import build_graph


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="FastAPI RAG Assistant",
    description=(
        "A self-corrective technical documentation "
        "assistant powered by LangGraph, Gemini, "
        "ChromaDB, and Tavily."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# LangGraph
# ---------------------------------------------------------

graph = build_graph()


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Technical question to answer.",
    )


# ---------------------------------------------------------
# Response model
# ---------------------------------------------------------

class Source(BaseModel):
    title: str
    source: str
    url: str
    type: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    source_type: str
    retry_count: int
    web_search_used: bool


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""

    return {
        "status": "healthy",
        "service": "fastapi-rag-assistant",
    }


# ---------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------

@app.post(
    "/ask",
    response_model=AskResponse,
)
async def ask(request: AskRequest):
    """Answer a technical documentation question."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # -----------------------------------------------------
    # Initial LangGraph state
    # -----------------------------------------------------

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

    try:

        result = graph.invoke(
            initial_state
        )

    except Exception as exc:

        print(
            f"Error while processing question: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while "
                "processing the question."
            ),
        ) from exc

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        source_type=result["source_type"],
        retry_count=result["retry_count"],
        web_search_used=result["web_search_used"],
    )