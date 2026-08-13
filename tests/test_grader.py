from app.services.grader import DocumentGrader


def main():
    print("Initializing Document Grader...")

    grader = DocumentGrader()

    query = "How do I use Depends for dependency injection in FastAPI?"

    relevant_document = """
FastAPI has a powerful Dependency Injection system.

You can use Depends to declare dependencies in your
path operation functions.

For example:

from fastapi import Depends, FastAPI

app = FastAPI()

async def common_parameters(
    q: str | None = None,
    skip: int = 0,
    limit: int = 100
):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(
    commons: dict = Depends(common_parameters)
):
    return commons
"""

    irrelevant_document = """
FastAPI response models are used to define the shape of
data returned by an API.

The response_model parameter can be used with path operation
decorators such as @app.get().
"""

    print("\n" + "=" * 70)
    print("TEST 1 — RELEVANT DOCUMENT")
    print("=" * 70)

    result = grader.grade(
        query=query,
        document=relevant_document,
    )

    print(f"\nRelevant: {result.relevant}")
    print(f"Reason:   {result.reason}")

    print("\n" + "=" * 70)
    print("TEST 2 — IRRELEVANT DOCUMENT")
    print("=" * 70)

    result = grader.grade(
        query=query,
        document=irrelevant_document,
    )

    print(f"\nRelevant: {result.relevant}")
    print(f"Reason:   {result.reason}")


if __name__ == "__main__":
    main()