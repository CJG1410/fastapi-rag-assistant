from google import genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiService:
    """Service wrapper for Google Gemini."""

    def __init__(self):
        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        self.model = GEMINI_MODEL

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate a text response from Gemini."""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text