import requests


BASE_URL = "http://127.0.0.1:8000"


def main():

    print("=" * 80)
    print("FASTAPI API TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Health check
    # --------------------------------------------------

    print("\nTesting /health...")

    response = requests.get(
        f"{BASE_URL}/health"
    )

    print(
        f"Status: {response.status_code}"
    )

    print(
        response.json()
    )

    # --------------------------------------------------
    # Ask endpoint
    # --------------------------------------------------

    print("\nTesting /ask...")

    payload = {
        "question": (
            "How do I configure PostgreSQL replication "
            "and database clustering?"
        )
    }

    response = requests.post(
        f"{BASE_URL}/ask",
        json=payload,
    )

    print(
        f"Status: {response.status_code}"
    )

    print("\nResponse:")

    print(
        response.json()
    )


if __name__ == "__main__":
    main()