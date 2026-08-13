from app.services.gemini import GeminiService


def main():
    print("Initializing Gemini...")

    gemini = GeminiService()

    print("Sending test request...")

    response = gemini.generate(
        "Reply with exactly: Gemini connection successful."
    )

    print("\nGemini response:")
    print(response)


if __name__ == "__main__":
    main()