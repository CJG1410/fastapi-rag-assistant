import json
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field

from langchain_core.documents import Document

from app.graph.workflow import build_graph
from app.services.memory import ConversationMemory

from app.ingestion.splitter import split_documents
from app.ingestion.embeddings import EmbeddingModel
from app.ingestion.vectorstore import ChromaVectorStore


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
# SERVICES
# =========================================================

embedding_model = EmbeddingModel()

vector_store = ChromaVectorStore()


# =========================================================
# FEEDBACK STORAGE
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(
    exist_ok=True
)

FEEDBACK_FILE = DATA_DIR / "feedback.json"


# =========================================================
# REQUEST MODELS
# =========================================================

class AskRequest(BaseModel):
    """Request model for asking a technical question."""

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


class FeedbackRequest(BaseModel):
    """User feedback for a generated answer."""

    session_id: str = Field(
        ...,
        min_length=1,
    )

    question: str = Field(
        ...,
        min_length=1,
    )

    rating: str = Field(
        ...,
        description="Feedback rating: up or down.",
    )

    comment: str | None = Field(
        default=None,
        max_length=2000,
    )


# =========================================================
# HTML TEXT EXTRACTOR
# =========================================================

class HTMLTextExtractor(HTMLParser):
    """Simple HTML-to-text parser for URL ingestion."""

    def __init__(self):
        super().__init__()

        self.parts: list[str] = []

        self.skip_content = False

        self.skip_tags = {
            "script",
            "style",
            "noscript",
            "svg",
        }

        self.current_title: list[str] = []

        self.in_title = False

    def handle_starttag(
        self,
        tag,
        attrs,
    ):
        tag = tag.lower()

        if tag in self.skip_tags:
            self.skip_content = True

        if tag == "title":
            self.in_title = True

    def handle_endtag(
        self,
        tag,
    ):
        tag = tag.lower()

        if tag in self.skip_tags:
            self.skip_content = False

        if tag == "title":
            self.in_title = False

    def handle_data(
        self,
        data,
    ):
        text = data.strip()

        if not text:
            return

        if self.skip_content:
            return

        if self.in_title:
            self.current_title.append(
                text
            )

        self.parts.append(text)

    @property
    def text(self) -> str:
        return "\n\n".join(
            self.parts
        )

    @property
    def title(self) -> str:
        return " ".join(
            self.current_title
        ).strip()


# =========================================================
# GRAPH EXECUTION HELPER
# =========================================================

def execute_query(
    request: AskRequest,
):
    """
    Execute the existing LangGraph RAG pipeline.

    This function is shared by /ask and /query so that
    both endpoints use exactly the same RAG behavior.
    """

    question = request.question

    session_id = request.session_id

    # -----------------------------------------------------
    # Get previous conversation history
    # -----------------------------------------------------

    history = memory.get_history(
        session_id
    )

    # -----------------------------------------------------
    # Store user message
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
    # Execute graph
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
# HEALTH
# =========================================================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "service": "fastapi-rag-assistant",
    }


# =========================================================
# ASK / QUERY
# =========================================================

@app.post("/ask")
@app.post("/query")
async def ask_question(
    request: AskRequest,
):
    """
    Answer a technical documentation question.

    /query is the assignment-required endpoint.
    /ask is preserved for backwards compatibility.
    """

    return execute_query(
        request
    )


# =========================================================
# INGESTION HELPERS
# =========================================================

def create_document_from_text(
    text: str,
    source: str,
    title: str,
    url: str = "",
) -> Document:
    """Create a LangChain Document with standard metadata."""

    metadata = {
        "source": source,
        "title": title,
        "content_type": "text",
    }

    if url:
        metadata["url"] = url

    return Document(
        page_content=text,
        metadata=metadata,
    )


def fetch_url_document(
    url: str,
) -> Document:
    """
    Fetch a documentation page from a URL and convert
    its HTML into text.
    """

    if not (
        url.startswith("http://")
        or url.startswith("https://")
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "URL must start with "
                "http:// or https://."
            ),
        )

    try:

        request = Request(
            url,
            headers={
                "User-Agent": (
                    "FastAPI-RAG-Assistant/1.0"
                )
            },
        )

        with urlopen(
            request,
            timeout=20,
        ) as response:

            raw_content = response.read()

            content_type = response.headers.get(
                "Content-Type",
                "",
            )

    except HTTPError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Unable to fetch URL. "
                f"HTTP status: {exc.code}"
            ),
        )

    except URLError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Unable to fetch URL: {exc.reason}"
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Unable to fetch URL: {exc}"
            ),
        )

    # -----------------------------------------------------
    # Decode
    # -----------------------------------------------------

    text = raw_content.decode(
        "utf-8",
        errors="ignore",
    )

    # -----------------------------------------------------
    # HTML
    # -----------------------------------------------------

    if (
        "html" in content_type.lower()
        or "<html" in text.lower()
    ):

        parser = HTMLTextExtractor()

        parser.feed(text)

        extracted_text = parser.text

        title = parser.title or url

    else:

        extracted_text = text

        title = url

    if not extracted_text.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "The provided URL did not contain "
                "extractable text."
            ),
        )

    return create_document_from_text(
        text=extracted_text,
        source=url,
        title=title,
        url=url,
    )


# =========================================================
# POST /INGEST
# =========================================================

@app.post("/ingest", status_code=201)
async def ingest_document(
    file: UploadFile | None = File(
        default=None,
    ),
    url: str | None = Form(
        default=None,
    ),
):
    """
    Ingest a new Markdown, text, or HTML document.

    Exactly one of:
    - file
    - url

    must be supplied.

    Unlike scripts/ingest.py, this endpoint DOES NOT
    reset ChromaDB. It adds new chunks to the existing
    collection.
    """

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if file is None and not url:

        raise HTTPException(
            status_code=400,
            detail=(
                "Provide either a file upload "
                "or a URL."
            ),
        )

    if file is not None and url:

        raise HTTPException(
            status_code=400,
            detail=(
                "Provide either a file upload "
                "or a URL, not both."
            ),
        )

    # -----------------------------------------------------
    # Load uploaded file
    # -----------------------------------------------------

    if file is not None:

        filename = file.filename or "uploaded_document"

        extension = Path(
            filename
        ).suffix.lower()

        allowed_extensions = {
            ".md",
            ".markdown",
            ".txt",
            ".html",
            ".htm",
        }

        if extension not in allowed_extensions:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported file type. "
                    "Supported types: .md, .markdown, "
                    ".txt, .html, .htm"
                ),
            )

        try:

            raw_content = await file.read()

            text = raw_content.decode(
                "utf-8",
                errors="ignore",
            )

        except Exception as exc:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unable to read uploaded file: "
                    f"{exc}"
                ),
            )

        if not text.strip():

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        title = Path(
            filename
        ).stem.replace(
            "_",
            " ",
        ).replace(
            "-",
            " ",
        ).title()

        document = create_document_from_text(
            text=text,
            source=filename,
            title=title,
        )

    # -----------------------------------------------------
    # Load URL
    # -----------------------------------------------------

    else:

        document = fetch_url_document(
            url
        )

    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------

    documents = split_documents(
        [document]
    )

    if not documents:

        raise HTTPException(
            status_code=400,
            detail=(
                "The document did not produce "
                "any retrievable chunks."
            ),
        )

    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    texts = [
        chunk.page_content
        for chunk in documents
    ]

    embeddings = (
        embedding_model.embed_documents(
            texts
        )
    )

    # -----------------------------------------------------
    # Generate unique IDs
    # -----------------------------------------------------

    ids = [
        f"api-{uuid.uuid4()}"
        for _ in documents
    ]

    # -----------------------------------------------------
    # Store in existing Chroma collection
    # -----------------------------------------------------

    metadatas = [
        chunk.metadata
        for chunk in documents
    ]

    try:

        vector_store.add_documents(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to store document "
                f"in ChromaDB: {exc}"
            ),
        )

    return {
        "message": "Document ingested successfully.",
        "source": document.metadata.get(
            "source",
            "",
        ),
        "title": document.metadata.get(
            "title",
            "",
        ),
        "url": document.metadata.get(
            "url",
            "",
        ),
        "chunks_added": len(documents),
        "total_chunks": vector_store.count(),
    }


# =========================================================
# GET /DOCUMENTS
# =========================================================

@app.get("/documents")
async def list_documents():
    """
    List the documents currently indexed in ChromaDB.
    """

    try:

        documents = (
            vector_store.list_documents()
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to read indexed documents: "
                f"{exc}"
            ),
        )

    return {
        "count": len(documents),
        "total_chunks": vector_store.count(),
        "documents": documents,
    }


# =========================================================
# FEEDBACK STORAGE HELPERS
# =========================================================

def load_feedback() -> list[dict]:
    """Load persisted feedback."""

    if not FEEDBACK_FILE.exists():

        return []

    try:

        with FEEDBACK_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if isinstance(data, list):

            return data

        return []

    except (
        json.JSONDecodeError,
        OSError,
    ):

        return []


def save_feedback(
    feedback: list[dict],
) -> None:
    """Persist feedback to disk."""

    with FEEDBACK_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            feedback,
            file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# POST /FEEDBACK
# =========================================================

@app.post("/feedback", status_code=201)
async def submit_feedback(
    request: FeedbackRequest,
):
    """
    Store thumbs-up/thumbs-down feedback with
    an optional comment.
    """

    rating = request.rating.lower().strip()

    if rating not in {
        "up",
        "down",
    }:

        raise HTTPException(
            status_code=422,
            detail=(
                "rating must be either "
                "'up' or 'down'."
            ),
        )

    feedback = load_feedback()

    entry = {
        "id": str(
            uuid.uuid4()
        ),
        "session_id": request.session_id,
        "question": request.question,
        "rating": rating,
        "comment": request.comment,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    feedback.append(
        entry
    )

    try:

        save_feedback(
            feedback
        )

    except OSError as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to save feedback: {exc}"
            ),
        )

    return {
        "message": "Feedback recorded successfully.",
        "feedback_id": entry["id"],
    }


# =========================================================
# CLEAR CONVERSATION
# =========================================================

@app.delete(
    "/sessions/{session_id}"
)
async def clear_session(
    session_id: str,
):

    memory.clear_session(
        session_id
    )

    return {
        "message": (
            "Conversation session cleared."
        ),
        "session_id": session_id,
    }