from collections import defaultdict


class ConversationMemory:
    """
    Simple in-memory conversation store.

    Each session_id maintains its own ordered
    conversation history.
    """

    def __init__(self):
        self.sessions = defaultdict(list)

    def add_user_message(
        self,
        session_id: str,
        message: str,
    ):
        self.sessions[session_id].append(
            {
                "role": "user",
                "content": message,
            }
        )

    def add_assistant_message(
        self,
        session_id: str,
        message: str,
    ):
        self.sessions[session_id].append(
            {
                "role": "assistant",
                "content": message,
            }
        )

    def get_history(
        self,
        session_id: str,
    ) -> list[dict]:

        return self.sessions.get(
            session_id,
            [],
        )

    def clear_session(
        self,
        session_id: str,
    ):
        self.sessions.pop(
            session_id,
            None,
        )