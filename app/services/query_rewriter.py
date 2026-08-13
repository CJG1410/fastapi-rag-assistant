from pydantic import BaseModel

from app.services.gemini import GeminiService


class RewrittenQuery(BaseModel):
    """Structured result from the query rewriting step."""

    query: str
    reason: str


class QueryRewriter:
    """Rewrite user queries to improve document retrieval."""

    def __init__(self):
        self.gemini = GeminiService()

    def rewrite(
        self,
        original_query: str,
        failed_documents: list[str],
    ) -> RewrittenQuery:
        """Rewrite a query based on failed retrieval results."""

        documents_text = "\n\n--- DOCUMENT ---\n\n".join(
            failed_documents
        )

        prompt = f"""
You are a query rewriting component in a technical
documentation RAG system.

The user asked a question, but the retrieved documents
were judged to be irrelevant.

Your task is to rewrite the user's question into a clearer,
more precise search query that is more likely to retrieve
useful technical documentation.

USER QUESTION:
{original_query}

RETRIEVED DOCUMENTS THAT WERE NOT RELEVANT:
{documents_text}

Rules:
1. Preserve the user's actual intent.
2. Do not invent technical facts.
3. Make the query concise and specific.
4. Use important technical terms from the user's question.
5. If the question is outside the available documentation,
   make the limitation clear through the rewritten query.
6. Return only the rewritten search query and a short reason.

Return:
- query: the rewritten search query
- reason: why this reformulation should improve retrieval
"""

        response = self.gemini.client.models.generate_content(
            model=self.gemini.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": (
                    RewrittenQuery.model_json_schema()
                ),
            },
        )

        return RewrittenQuery.model_validate_json(
            response.text
        )