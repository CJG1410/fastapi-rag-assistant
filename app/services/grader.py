from pydantic import BaseModel

from app.services.gemini import GeminiService


class RelevanceGrade(BaseModel):
    """Structured result from the document relevance grader."""

    relevant: bool
    reason: str


class DocumentGrader:
    """Use Gemini to determine whether a document is relevant."""

    def __init__(self):
        self.gemini = GeminiService()

    def grade(
        self,
        query: str,
        document: str,
    ) -> RelevanceGrade:
        """Grade a document against a user query."""

        prompt = f"""
You are a relevance grader for a technical documentation
question-answering system.

Your task is to determine whether the provided document contains
information that is useful for answering the user's question.

USER QUESTION:
{query}

DOCUMENT:
{document}

Evaluate the document based ONLY on whether it contains information
that can help answer the user's question.

A document is relevant if:
- It directly answers the question, or
- It contains technical information needed to answer the question.

A document is not relevant if:
- It discusses a different technical topic.
- It is only loosely related to the question.
- It cannot provide useful information for answering the question.

Return:
- relevant: true if the document is useful
- relevant: false otherwise
- reason: a short explanation of your decision
"""

        response = self.gemini.client.models.generate_content(
            model=self.gemini.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": RelevanceGrade.model_json_schema(),
            },
        )

        return RelevanceGrade.model_validate_json(
            response.text
        )