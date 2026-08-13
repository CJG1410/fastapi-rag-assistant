from app.services.grader import DocumentGrader
from app.services.retriever import Retriever


def main():
    print("Initializing services...")

    retriever = Retriever()
    grader = DocumentGrader()

    query = "How do I configure PostgreSQL replication and database clustering?"

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    for index, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print("\n" + "-" * 80)
        print(f"RESULT {index}")
        print("-" * 80)

        print(f"Distance: {result['distance']:.4f}")
        print(f"Source:   {metadata.get('source')}")
        print(f"Title:    {metadata.get('title')}")
        print(f"Type:     {metadata.get('content_type')}")

        grade = grader.grade(
            query=query,
            document=result["content"],
        )

        print(f"Relevant: {grade.relevant}")
        print(f"Reason:   {grade.reason}")


if __name__ == "__main__":
    main()