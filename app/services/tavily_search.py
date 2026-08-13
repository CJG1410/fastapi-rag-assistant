from tavily import TavilyClient

from app.config import TAVILY_API_KEY


class TavilySearchService:
    """Service wrapper for Tavily web search."""

    def __init__(self):
        if not TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY is not set. "
                "Add it to your .env file."
            )

        self.client = TavilyClient(
            api_key=TAVILY_API_KEY
        )

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        """Search the web using Tavily."""

        response = self.client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
        )

        results = response.get(
            "results",
            [],
        )

        return results