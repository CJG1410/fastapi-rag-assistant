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
        conversation_history: list[dict] | None = None,
    ) -> RewrittenQuery:
        """
        Rewrite a query based on failed retrieval results
        and optional conversation history.
        """

        # -------------------------------------------------
        # Failed retrieval documents
        # -------------------------------------------------

        documents_text = "\n\n--- DOCUMENT ---\n\n".join(
            failed_documents
        )

        # -------------------------------------------------
        # Conversation history
        # -------------------------------------------------

        if conversation_history:

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

                history_parts.append(
                    f"{role.upper()}: {content}"
                )

            conversation_text = "\n".join(
                history_parts
            )

        else:

            conversation_text = (
                "No previous conversation is available."
            )

        # -------------------------------------------------
        # Prompt
        # -------------------------------------------------

        prompt = f"""
You are a query rewriting component in a technical
documentation RAG system.

The user asked a question, but the retrieved documents
were judged to be irrelevant.

Your task is to rewrite the user's question into a clearer,
more precise search query that is more likely to retrieve
useful technical documentation.

You must also use the previous conversation when it helps
resolve references such as:

- "it"
- "this"
- "that"
- "they"
- "the above"
- "how do I do that?"
- "how do I configure it?"

Do not change the user's actual technical intent.

USER QUESTION:
{original_query}

PREVIOUS CONVERSATION:
{conversation_text}

RETRIEVED DOCUMENTS THAT WERE NOT RELEVANT:
{documents_text}

Rules:

1. Preserve the user's actual intent.

2. Use previous conversation context when necessary to
   resolve ambiguous references.

3. If the current question is already clear, do not
   unnecessarily add information from the conversation.

4. Do not invent technical facts.

5. Make the query concise and specific.

6. Use important technical terms from the user's question
   and relevant previous conversation context.

7. The rewritten query should be suitable for semantic
   retrieval from technical documentation.

8. If the question is outside the available documentation,
   make the rewritten query more technically specific,
   but do not invent unsupported details.

9. Return only the rewritten search query and a short reason.

Return:

- query: the rewritten search query
- reason: why this reformulation should improve retrieval
"""

        # -------------------------------------------------
        # Gemini structured output
        # -------------------------------------------------

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