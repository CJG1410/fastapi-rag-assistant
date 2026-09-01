# FastAPI Technical Documentation Assistant

A self-corrective **Retrieval-Augmented Generation (RAG)** system for
answering questions over technical documentation. The application uses a
**LangGraph StateGraph** to prepare queries, retrieve and grade
documents, rewrite failed queries, generate grounded answers, verify
generated answers, and optionally fall back to web search when the local
corpus cannot answer the question.

------------------------------------------------------------------------

## 1. Project Overview

The system answers questions about a small corpus of FastAPI technical
documentation.

At a high level:

1.  A user submits a question through the FastAPI API.
2.  The conversation-aware query preparation node resolves ambiguity
    using session history.
3.  The query is embedded and searched against ChromaDB.
4.  Retrieved chunks are graded by an LLM for relevance.
5.  If relevant documents are found, they are passed to generation.
6.  If no relevant documents are found, the query is rewritten and
    retrieval is retried.
7.  If local retrieval still cannot answer the question, the system can
    use Tavily web search.
8.  The generated answer is checked against the retrieved context.
9.  The API returns the answer together with source metadata and
    retrieval/verification information.

The implementation also includes the optional features from the
assignment:

-   Hallucination/groundedness verification
-   Web search fallback
-   Conversation memory
-   Streamlit UI

------------------------------------------------------------------------

## 2. Features

### Core requirements

-   LangGraph `StateGraph`
-   Query analysis / conversation-aware query preparation
-   Query rewriting
-   ChromaDB vector retrieval
-   Top-k retrieval with source metadata
-   LLM-based document relevance grading
-   Conditional routing
-   Retrieval retry limit
-   Grounded answer generation
-   FastAPI API
-   Document ingestion
-   Indexed document listing
-   User feedback endpoint

### Additional features

-   Answer verification against retrieved context
-   Tavily web-search fallback
-   Session-based conversation memory
-   Streamlit user interface
-   URL ingestion
-   Markdown/text/HTML file ingestion
-   Structured API responses
-   Persistent ChromaDB storage

------------------------------------------------------------------------

## 3. Architecture

``` mermaid
flowchart TD
    A[User Question] --> B[FastAPI /query]
    B --> C[Conversation-Aware Query Preparation]

    C --> D[Embedding Model]
    D --> E[ChromaDB Retrieval]

    E --> F[Document Grading]

    F -->|Relevant documents found| G[Answer Generation]
    F -->|No relevant documents| H{Retry limit reached?}

    H -->|No| I[Query Rewriter]
    I --> D

    H -->|Yes| J[Tavily Web Search]

    J --> G

    G --> K[Hallucination / Groundedness Verification]

    K -->|Supported| L[Final Response]
    K -->|Not supported| M[Regenerate / Correct]
    M --> L

    L --> N[Conversation Memory]
```

### Ingestion architecture

``` mermaid
flowchart LR
    A[Markdown / Text / HTML / URL] --> B[Document Loader]
    B --> C[Markdown-Aware Splitter]
    C --> D[Sentence Transformer]
    D --> E[ChromaDB]
```

------------------------------------------------------------------------

## 4. LangGraph Workflow

The main workflow is implemented as a LangGraph `StateGraph`.

### 4.1 Query Preparation

The first stage prepares the raw user question for retrieval.

For a standalone question:

``` text
What is FastAPI dependency injection?
```

the original query can be used directly.

For a follow-up such as:

``` text
How do I define one?
```

the system uses the previous conversation context to resolve the
reference and prepare a more useful retrieval query such as:

``` text
how to define a dependency in FastAPI
```

This prevents ambiguous follow-up questions from being searched
literally.

### 4.2 Retrieval

The prepared query is converted into a vector using:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

The resulting embedding is searched against ChromaDB.

The retriever returns the top five candidate chunks together with:

-   document content
-   source
-   title
-   URL, when available
-   content type
-   similarity distance

### 4.3 Document Grading

The retrieved chunks are passed to an LLM grader.

Each candidate is classified as:

``` text
Relevant
```

or:

``` text
Irrelevant
```

Irrelevant chunks are removed before generation.

This is important because vector similarity alone does not guarantee
that a retrieved chunk actually answers the question.

### 4.4 Self-Correction

If relevant documents are found:

``` text
Retrieve → Grade → Generate
```

If no relevant documents are found:

``` text
Retrieve → Grade → Rewrite → Retrieve
```

The workflow has a retry limit to prevent an infinite loop.

### 4.5 Web Fallback

If local retrieval remains insufficient after the configured retries,
the system can use Tavily web search.

This allows questions outside the local FastAPI corpus to receive an
answer when external information is available.

The API response identifies whether web search was used.

### 4.6 Generation

The generation node receives only the context judged useful by the
grading stage.

The LLM is instructed to answer from the supplied context rather than
inventing unsupported information.

### 4.7 Answer Verification

After generation, a verification node evaluates whether the answer is
supported by the available context.

The response exposes:

-   `hallucination_checked`
-   `answer_supported`
-   `verification_reason`

This provides an additional groundedness check before returning the
answer.

------------------------------------------------------------------------

## 5. Document Corpus

The project uses five FastAPI documentation documents:

``` text
docs/
├── dependencies.md
├── error_handling.md
├── request_body.md
├── response_models.md
└── routing.md
```

The documents are based on FastAPI technical documentation and contain
both explanatory text and code examples.

The corpus is intentionally small because the assignment specifies a
small technical-documentation corpus.

------------------------------------------------------------------------

## 6. Document Ingestion Pipeline

The ingestion pipeline follows:

``` text
Load
  ↓
Clean Markdown
  ↓
Markdown Header Splitting
  ↓
Code / Prose Separation
  ↓
Recursive Text Splitting
  ↓
Embedding
  ↓
ChromaDB
```

### Chunking strategy

The splitter uses:

``` text
Chunk size:     1000 characters
Chunk overlap:   150 characters
```

Markdown headings are considered first so that related documentation
sections remain grouped.

The splitter also detects fenced code blocks.

Code blocks are kept together as independent chunks instead of being
arbitrarily split across multiple chunks. This is useful for technical
documentation because breaking a code example in the middle can make the
retrieved context less useful.

For prose, `RecursiveCharacterTextSplitter` is used with paragraph and
line-oriented separators.

### Why this strategy?

Technical documentation is highly structured. A purely character-based
splitter can separate a heading from its explanation or break a code
example.

The chosen approach therefore combines:

-   Markdown structure
-   code-block preservation
-   controlled chunk size
-   overlap for contextual continuity

This gives the retriever useful semantic units while avoiding
excessively large chunks.

------------------------------------------------------------------------

## 7. Embedding Strategy

The project uses:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding size is:

``` text
384 dimensions
```

Embeddings are normalized before being stored.

### Why this model?

The model is lightweight, runs locally, and provides a practical balance
between retrieval quality and resource usage for a small technical
corpus.

Using a local embedding model also avoids requiring a separate embedding
API key.

------------------------------------------------------------------------

## 8. Vector Store

The project uses **ChromaDB** with persistent local storage:

``` text
chroma_db/
```

Collection:

``` text
fastapi_docs
```

The vector store stores:

-   chunk text
-   embeddings
-   chunk IDs
-   source metadata

The initial corpus currently produces approximately 214 indexed chunks.

The standalone ingestion script can rebuild the collection from the
corpus.

The API ingestion endpoint intentionally does **not** reset the
collection. New documents are added to the existing index.

------------------------------------------------------------------------

## 9. API

The FastAPI application provides the following endpoints.

  Method   Endpoint                   Purpose
  -------- -------------------------- ----------------------------------------
  GET      `/health`                  Health check
  POST     `/query`                   Submit a technical question
  POST     `/ask`                     Backwards-compatible question endpoint
  POST     `/ingest`                  Add a document from a file or URL
  GET      `/documents`               List indexed documents
  POST     `/feedback`                Submit thumbs-up/down feedback
  DELETE   `/sessions/{session_id}`   Clear a conversation session

The four primary assignment endpoints are:

``` text
POST /query
POST /ingest
GET  /documents
POST /feedback
```

------------------------------------------------------------------------

## 10. Example `/query` Request

``` http
POST /query
Content-Type: application/json
```

``` json
{
  "question": "How do I use Depends for dependency injection in FastAPI?",
  "session_id": "demo-session"
}
```

### Example response

``` json
{
  "question": "How do I use Depends for dependency injection in FastAPI?",
  "answer": "To use `Depends` for dependency injection in FastAPI, define a dependency function and pass the function itself to `Depends()` without calling it. FastAPI then executes the dependency and injects its result into the path operation function.",
  "sources": [
    {
      "title": "FastAPI Dependencies",
      "source": "dependencies.md",
      "url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
      "type": "local"
    }
  ],
  "source_type": "local",
  "retry_count": 0,
  "web_search_used": false,
  "session_id": "demo-session",
  "hallucination_checked": true,
  "answer_supported": true,
  "verification_reason": "The generated answer is supported by the retrieved FastAPI dependency documentation."
}
```

Source metadata is returned separately so clients can display the
references alongside the generated answer.

------------------------------------------------------------------------

## 11. Example `/ingest` Request

The endpoint accepts either a file upload or a URL.

### File upload

``` http
POST /ingest
Content-Type: multipart/form-data
```

Example with cURL:

``` bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -F "file=@example.md"
```

### URL ingestion

``` bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -F "url=https://example.com/documentation"
```

The endpoint:

``` text
document
   ↓
split
   ↓
embed
   ↓
add to ChromaDB
```

It does not reset the existing collection.

------------------------------------------------------------------------

## 12. Example `/documents` Response

``` http
GET /documents
```

Example:

``` json
{
  "count": 5,
  "total_chunks": 214,
  "documents": [
    {
      "source": "dependencies.md",
      "title": "FastAPI Dependencies",
      "url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
      "chunks": 40
    }
  ]
}
```

The exact chunk counts depend on the current indexed corpus.

------------------------------------------------------------------------

## 13. Example `/feedback` Request

``` http
POST /feedback
Content-Type: application/json
```

``` json
{
  "session_id": "demo-session",
  "question": "What is FastAPI dependency injection?",
  "rating": "up",
  "comment": "The answer was accurate and useful."
}
```

Supported ratings:

``` text
up
down
```

Feedback is persisted locally in:

``` text
data/feedback.json
```

------------------------------------------------------------------------

## 14. Setup

### Requirements

-   Python 3.10+
-   pip
-   Git
-   Internet access for model downloads and optional web search
-   Google Gemini API key
-   Tavily API key if web fallback is enabled

ChromaDB and the embedding model run locally.

### Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fastapi-rag-assistant
```

### Create a virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

``` bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 15. Environment Variables

Create a `.env` file in the project root.

Example:

``` env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

The Gemini key is required for the LLM-powered components.

The Tavily key is required only when using the web-search fallback.

### Security

Do not commit `.env` to Git.

A safe repository should include:

``` text
.env.example
```

instead of the actual secrets.

Example:

``` env
GEMINI_API_KEY=
TAVILY_API_KEY=
```

------------------------------------------------------------------------

## 16. Initial Document Ingestion

The initial corpus can be indexed using:

``` bash
python -m scripts.ingest
```

The ingestion script:

1.  Loads the documents from `docs/`
2.  Splits them into chunks
3.  Generates embeddings
4.  Rebuilds the ChromaDB collection
5.  Validates the number of stored chunks

The standalone script intentionally resets the collection so that the
corpus can be rebuilt reproducibly.

For incremental ingestion after the application is running, use:

``` text
POST /ingest
```

instead.

------------------------------------------------------------------------

## 17. Run FastAPI

Start the application with:

``` bash
uvicorn app.main:app --reload
```

The API will be available at:

``` text
http://127.0.0.1:8000
```

Interactive API documentation:

``` text
http://127.0.0.1:8000/docs
```

------------------------------------------------------------------------

## 18. Run Streamlit

Start the UI with:

``` bash
streamlit run ui/streamlit_app.py
```

The Streamlit application provides an interactive interface for
submitting questions and viewing answers and sources.

------------------------------------------------------------------------

## 19. Project Structure

``` text
fastapi-rag-assistant/
│
├── app/
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   ├── embeddings.py
│   │   └── vectorstore.py
│   │
│   ├── services/
│   │   ├── gemini.py
│   │   ├── grader.py
│   │   ├── hallucination_checker.py
│   │   ├── memory.py
│   │   ├── query_rewriter.py
│   │   ├── rag_retriever.py
│   │   ├── retriever.py
│   │   └── tavily_search.py
│   │
│   └── main.py
│
├── docs/
│   ├── dependencies.md
│   ├── error_handling.md
│   ├── request_body.md
│   ├── response_models.md
│   └── routing.md
│
├── scripts/
│   └── ingest.py
│
├── tests/
│   ├── test_api.py
│   ├── test_retrieval.py
│   ├── test_splitter.py
│   └── ...
│
├── ui/
│   └── streamlit_app.py
│
├── data/
│   └── feedback.json
│
├── chroma_db/
├── requirements.txt
├── .env.example
└── README.md
```

------------------------------------------------------------------------

## 20. Design Decisions

### LangGraph instead of a linear RAG chain

A linear pipeline would be sufficient for basic retrieval and
generation, but the assignment specifically evaluates graph-based
workflows and self-correction.

LangGraph makes the control flow explicit:

``` text
retrieve
   ↓
grade
   ↓
 ┌─┴─────────────┐
 ↓               ↓
relevant       irrelevant
 ↓               ↓
generate       rewrite
                 ↓
              retrieve
```

This makes retry behavior and state transitions easier to reason about.

### LLM-based document grading

Vector similarity is used for candidate retrieval, but similarity alone
does not determine whether a chunk answers the question.

The grading node provides a second relevance check before generation.

### Separate query rewriting

Query rewriting is only triggered when retrieval fails rather than on
every request.

This reduces unnecessary LLM calls and preserves the original query when
it is already working well.

### Conversation-aware retrieval

A follow-up such as:

``` text
How do I define one?
```

does not contain enough information to retrieve reliably by itself.

The system therefore uses session history to transform ambiguous
follow-ups into explicit retrieval queries.

### Local embeddings

`all-MiniLM-L6-v2` was selected because it can run locally and is
lightweight enough for the small corpus.

This also avoids adding another external API dependency for embeddings.

### ChromaDB

ChromaDB provides persistent local vector storage with a simple API and
is appropriate for the relatively small corpus used in this assignment.

------------------------------------------------------------------------

## 21. Design Tradeoffs

### Retrieval quality vs. simplicity

A more sophisticated production system could use hybrid search,
reranking, metadata filtering, or a larger embedding model.

For this assignment, a lightweight embedding model plus LLM grading
provides a simpler architecture while still demonstrating retrieval
correction.

### LLM grading cost vs. retrieval reliability

Grading every candidate requires additional LLM calls.

The tradeoff is intentional: the grader reduces the chance that
semantically similar but irrelevant chunks are passed to generation.

### Web fallback vs. corpus grounding

The local corpus is the preferred source.

Web search is used only when the local corpus cannot provide relevant
documents after retries.

This prevents normal in-domain questions from unnecessarily depending on
external sources.

### Persistent local storage vs. production infrastructure

ChromaDB is suitable for a take-home project and local development.

A production system would likely require managed vector storage,
database backups, monitoring, authentication, and stronger concurrency
controls.

### Feedback storage

Feedback is persisted to a local JSON file for simplicity.

A production system would use a database or analytics pipeline.

------------------------------------------------------------------------

## 22. Assumptions

-   The technical corpus is small enough for local ChromaDB storage.
-   Markdown documents are UTF-8 encoded.
-   The primary corpus consists of technical documentation.
-   Gemini is available for grading, rewriting, generation, and
    verification.
-   Tavily is available when web fallback is enabled.
-   The application is primarily intended for local evaluation rather
    than production deployment.
-   Session memory is maintained in the application's process memory.
-   API feedback is stored locally.

------------------------------------------------------------------------

## 23. Error Handling and Validation

The API validates required fields and rejects invalid requests.

Examples include:

-   Empty questions
-   Missing ingestion input
-   Supplying both a file and URL to `/ingest`
-   Unsupported file extensions
-   Empty uploaded documents
-   Invalid URLs
-   Invalid feedback ratings

Meaningful HTTP status codes are returned for validation and processing
failures.

------------------------------------------------------------------------

## 24. Testing

The project includes tests for the major parts of the system, including:

-   Document loading
-   Chunking
-   Embedding generation
-   Retrieval
-   Self-corrective retrieval
-   LangGraph workflow
-   API endpoints

Important scenarios tested include:

### In-domain query

``` text
What is FastAPI dependency injection?
```

Expected behavior:

``` text
Retrieve → Grade → Generate → Verify
```

### Ambiguous follow-up

``` text
What is FastAPI dependency injection?
How do I define one?
```

The second query is resolved using conversation context before
retrieval.

### Out-of-domain query

``` text
How do I configure PostgreSQL replication and database clustering?
```

Expected behavior:

``` text
Retrieve
  ↓
Grade
  ↓
No relevant documents
  ↓
Rewrite
  ↓
Retrieve
  ↓
No relevant documents
  ↓
Tavily web fallback
```

### API validation

The API was tested for:

-   `/health`
-   `/query`
-   `/ingest`
-   `/documents`
-   `/feedback`

------------------------------------------------------------------------

## 25. What I Would Improve With More Time

### Better retrieval

I would evaluate:

-   hybrid BM25 + vector search
-   cross-encoder reranking
-   metadata-aware retrieval
-   larger embedding models
-   retrieval evaluation metrics such as Recall@k and MRR

### Better ingestion

I would add:

-   duplicate detection
-   document versioning
-   document deletion
-   incremental updates
-   asynchronous ingestion jobs
-   more robust HTML parsing
-   PDF support

### Production memory

The current conversation memory is process-local.

For production, I would move it to Redis or a persistent database so
that sessions survive application restarts and work across multiple
application instances.

### Production feedback storage

Feedback would move from JSON storage to a database with analytics
capabilities.

### Authentication and authorization

A production deployment would require authentication, API-key
management, rate limiting, and access control.

### Observability

I would add:

-   structured logging
-   tracing
-   retrieval latency metrics
-   LLM latency/cost tracking
-   failure monitoring
-   retrieval quality dashboards

### Evaluation

I would build a small evaluation dataset containing expected answers and
relevant source documents, then measure:

-   retrieval accuracy
-   answer faithfulness
-   citation accuracy
-   hallucination rate
-   latency
-   web-fallback frequency

------------------------------------------------------------------------

## 26. Thought Process

The architecture was designed around the main failure mode of a basic
RAG system: **retrieval can be wrong even when the vector similarity
score looks reasonable**.

Instead of assuming that the top-k vector results are correct, the
workflow separates retrieval from relevance verification.

The reasoning is:

``` text
Vector search finds candidates
        ↓
LLM checks whether candidates actually answer the question
        ↓
Relevant context is passed to generation
```

When retrieval fails, the system does not immediately answer from
insufficient context. It attempts query reformulation first:

``` text
Original query
      ↓
Retrieval
      ↓
No relevant context
      ↓
Query rewriting
      ↓
Retrieval again
```

Only after local retrieval remains unsuccessful does the system use
external web search.

The addition of answer verification provides another safety layer after
generation:

``` text
Retrieved context
      ↓
Generated answer
      ↓
Supported by context?
      ↓
Final answer
```

This creates a self-corrective workflow rather than a simple:

``` text
Question → Vector DB → LLM
```

pipeline.

------------------------------------------------------------------------

## 27. Limitations

This project is designed as a focused take-home implementation rather
than a production RAG platform.

Known limitations include:

-   In-memory conversation state
-   Local JSON feedback storage
-   Local ChromaDB
-   No authentication
-   No rate limiting
-   No distributed deployment
-   Limited corpus size
-   Basic HTML extraction for URL ingestion
-   Web fallback depends on an external API
-   LLM grading and verification introduce additional latency and API
    usage

These are intentional tradeoffs given the scope and time constraints of
the assignment.

------------------------------------------------------------------------

## 28. Summary

The final system combines:

``` text
FastAPI
   +
LangGraph
   +
ChromaDB
   +
Sentence Transformers
   +
Google Gemini
   +
Tavily
   +
Conversation Memory
   +
Streamlit
```

The core workflow is self-corrective, with document grading, query
rewriting, retry limits, grounded generation, and answer verification.

The implementation also provides the required FastAPI endpoints for
querying, ingestion, document listing, and feedback, while preserving
the local document corpus and supporting incremental API ingestion.
