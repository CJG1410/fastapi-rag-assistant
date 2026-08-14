import uuid

import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FastAPI Technical Documentation Assistant",
    page_icon="📚",
    layout="wide",
)


# =========================================================
# SESSION STATE
# =========================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# HEADER
# =========================================================

st.title("📚 FastAPI Technical Documentation Assistant")

st.caption(
    "Self-corrective RAG assistant powered by "
    "LangGraph, ChromaDB, Gemini, and Tavily."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("System")

    st.write(
        "Ask questions about the technical "
        "documentation loaded into the RAG system."
    )

    st.divider()

    st.subheader("Session")

    st.code(
        st.session_state.session_id,
        language="text",
    )

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        try:

            response = requests.delete(
                f"{API_URL}/sessions/"
                f"{st.session_state.session_id}",
                timeout=10,
            )

            if response.status_code == 200:

                st.session_state.messages = []

                st.session_state.session_id = (
                    str(uuid.uuid4())
                )

                st.success(
                    "Conversation cleared."
                )

                st.rerun()

            else:

                st.error(
                    "Failed to clear the conversation."
                )

        except requests.RequestException:

            st.error(
                "Could not connect to the FastAPI server."
            )

    st.divider()

    st.subheader("Backend")

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if health_response.status_code == 200:

            st.success(
                "FastAPI backend connected"
            )

        else:

            st.error(
                "FastAPI backend unavailable"
            )

    except requests.RequestException:

        st.error(
            "FastAPI backend unavailable"
        )

    st.divider()

    st.caption(
        "FastAPI RAG Assistant"
    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # -------------------------------------------------
        # Display metadata for assistant messages
        # -------------------------------------------------

        if message["role"] == "assistant":

            metadata = message.get(
                "metadata"
            )

            if metadata:

                st.divider()

                # Verification status

                if metadata.get(
                    "answer_supported"
                ):

                    st.success(
                        "✓ Answer verified against "
                        "retrieved context"
                    )

                elif metadata.get(
                    "hallucination_checked"
                ):

                    st.warning(
                        "⚠ Answer could not be fully "
                        "verified"
                    )

                # Web search status

                if metadata.get(
                    "web_search_used"
                ):

                    st.info(
                        "🌐 Web search was used"
                    )

                else:

                    st.caption(
                        "📚 Answer generated from "
                        "local documentation"
                    )

                # Sources

                sources = metadata.get(
                    "sources",
                    [],
                )

                if sources:

                    with st.expander(
                        f"Sources ({len(sources)})"
                    ):

                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):

                            title = source.get(
                                "title",
                                "Untitled source",
                            )

                            url = source.get(
                                "url",
                                "",
                            )

                            source_type = source.get(
                                "type",
                                "unknown",
                            )

                            st.markdown(
                                f"**{index}. {title}**"
                            )

                            st.caption(
                                f"Type: {source_type}"
                            )

                            if url:

                                st.markdown(
                                    f"[Open source]({url})"
                                )

                            st.divider()


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a technical documentation question..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # -----------------------------------------------------
    # Generate assistant response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documentation and generating answer..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "session_id": (
                            st.session_state.session_id
                        ),
                        "question": question,
                    },
                    timeout=120,
                )

                # -----------------------------------------
                # API error
                # -----------------------------------------

                if response.status_code != 200:

                    try:

                        error_data = response.json()

                    except ValueError:

                        error_data = response.text

                    st.error(
                        f"API error "
                        f"({response.status_code}): "
                        f"{error_data}"
                    )

                else:

                    data = response.json()

                    answer = data.get(
                        "answer",
                        "No answer was returned.",
                    )

                    st.markdown(answer)

                    # -------------------------------------
                    # Metadata
                    # -------------------------------------

                    metadata = {
                        "sources": data.get(
                            "sources",
                            [],
                        ),
                        "source_type": data.get(
                            "source_type",
                            "",
                        ),
                        "web_search_used": data.get(
                            "web_search_used",
                            False,
                        ),
                        "retry_count": data.get(
                            "retry_count",
                            0,
                        ),
                        "hallucination_checked": data.get(
                            "hallucination_checked",
                            False,
                        ),
                        "answer_supported": data.get(
                            "answer_supported",
                            False,
                        ),
                        "verification_reason": data.get(
                            "verification_reason",
                            "",
                        ),
                    }

                    # -------------------------------------
                    # Verification
                    # -------------------------------------

                    if metadata[
                        "answer_supported"
                    ]:

                        st.success(
                            "✓ Answer verified against "
                            "retrieved context"
                        )

                    elif metadata[
                        "hallucination_checked"
                    ]:

                        st.warning(
                            "⚠ Answer could not be fully "
                            "verified"
                        )

                    # -------------------------------------
                    # Source type
                    # -------------------------------------

                    if metadata[
                        "web_search_used"
                    ]:

                        st.info(
                            "🌐 Web search was used "
                            "because local documentation "
                            "was insufficient."
                        )

                    else:

                        st.caption(
                            "📚 Answer generated from "
                            "local documentation."
                        )

                    # -------------------------------------
                    # Sources
                    # -------------------------------------

                    sources = metadata[
                        "sources"
                    ]

                    if sources:

                        with st.expander(
                            f"Sources ({len(sources)})"
                        ):

                            for index, source in enumerate(
                                sources,
                                start=1,
                            ):

                                title = source.get(
                                    "title",
                                    "Untitled source",
                                )

                                url = source.get(
                                    "url",
                                    "",
                                )

                                source_type = source.get(
                                    "type",
                                    "unknown",
                                )

                                st.markdown(
                                    f"**{index}. {title}**"
                                )

                                st.caption(
                                    f"Type: {source_type}"
                                )

                                if url:

                                    st.markdown(
                                        f"[Open source]"
                                        f"({url})"
                                    )

                                st.divider()

                    # -------------------------------------
                    # Save assistant message
                    # -------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "metadata": metadata,
                        }
                    )

            except requests.Timeout:

                st.error(
                    "The request timed out. "
                    "The RAG pipeline may still be "
                    "processing or the backend is unavailable."
                )

            except requests.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI "
                    "backend. Make sure the FastAPI server "
                    "is running."
                )

            except requests.RequestException as exc:

                st.error(
                    f"Request failed: {exc}"
                )

            except Exception as exc:

                st.error(
                    f"Unexpected error: {exc}"
                )