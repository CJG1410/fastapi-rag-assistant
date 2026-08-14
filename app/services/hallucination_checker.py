from pydantic import BaseModel, Field

from app.services.gemini import GeminiService


class HallucinationCheck(BaseModel):
    supported: bool = Field(
        description="Whether the answer is supported by the provided context."
    )

    reason: str = Field(
        description="Short explanation for the decision."
    )


class HallucinationChecker:
    """Verify that a generated answer is grounded in retrieved context."""

    def __init__(self):
        self.gemini = GeminiService()

    def check(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> HallucinationCheck:

        prompt = f"""
You are a strict RAG answer verification system.

Your task is to determine whether the generated answer
is supported by the provided context.

Do NOT judge whether the answer is generally correct
according to your own knowledge.

Only judge whether the claims in the answer are supported
by the supplied context.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Return:

supported:
true if the answer is sufficiently supported by the context.
false if the answer contains unsupported, invented, or
contradictory information.

reason:
Give a short explanation.

Be conservative. If important claims cannot be supported
by the context, mark the answer as unsupported.
"""

        response = self.gemini.client.models.generate_content(
            model=self.gemini.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": HallucinationCheck,
            },
        )

        return HallucinationCheck.model_validate_json(
            response.text
        )