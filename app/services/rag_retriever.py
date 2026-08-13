from app.services.query_rewriter import QueryRewriter
from app.services.retriever import Retriever
from app.services.grader import DocumentGrader


MAX_RETRIES = 2
TOP_K = 5
DISTANCE_THRESHOLD = 1.0


class RAGRetriever:
    """Self-corrective retrieval with grading and query rewriting."""

    def __init__(self):
        self.retriever = Retriever()
        self.grader = DocumentGrader()
        self.query_rewriter = QueryRewriter()

    def retrieve_with_retry(
        self,
        query: str,
    ) -> dict:
        """
        Retrieve relevant documents.

        If retrieved documents are not relevant, rewrite the
        query and retry up to MAX_RETRIES times.
        """

        current_query = query

        for attempt in range(MAX_RETRIES + 1):

            print(
                f"\nRetrieval attempt "
                f"{attempt + 1}/{MAX_RETRIES + 1}"
            )

            print(f"Query: {current_query}")

            results = self.retriever.retrieve(
                query=current_query,
                top_k=TOP_K,
            )

            # --------------------------------------------------
            # Distance quality gate
            # --------------------------------------------------

            if not results:
                relevant_results = []

            else:
                candidate_results = [
                    result
                    for result in results
                    if result["distance"] <= DISTANCE_THRESHOLD
                ]

                # --------------------------------------------------
                # Gemini document grading
                # --------------------------------------------------

                relevant_results = []

                for result in candidate_results:

                    grade = self.grader.grade(
                        query=current_query,
                        document=result["content"],
                    )

                    if grade.relevant:
                        result["grade_reason"] = grade.reason
                        relevant_results.append(result)

            # --------------------------------------------------
            # Relevant documents found
            # --------------------------------------------------

            if relevant_results:

                return {
                    "original_query": query,
                    "final_query": current_query,
                    "results": relevant_results,
                    "retry_count": attempt,
                    "success": True,
                }

            # --------------------------------------------------
            # Retry limit reached
            # --------------------------------------------------

            if attempt >= MAX_RETRIES:
                break

            # --------------------------------------------------
            # Query rewriting
            # --------------------------------------------------

            failed_documents = [
                result["content"]
                for result in results
            ]

            rewritten = self.query_rewriter.rewrite(
                original_query=current_query,
                failed_documents=failed_documents,
            )

            current_query = rewritten.query

            print(
                f"Rewritten query: {current_query}"
            )

        # ------------------------------------------------------
        # No relevant documents after retries
        # ------------------------------------------------------

        return {
            "original_query": query,
            "final_query": current_query,
            "results": [],
            "retry_count": MAX_RETRIES,
            "success": False,
        }