import os

import streamlit as st

from src.ingest import create_vectorstore, VECTORSTORE_PATH
from src.retriever import answer_question


st.set_page_config(
    page_title="Pharma Dictionary Bot",
    page_icon="💊"
)


# --------------------------------------------------
# Initialize ChromaDB
# --------------------------------------------------

@st.cache_resource
def initialize_database():

    if not os.path.exists(VECTORSTORE_PATH):

        with st.spinner(
            "Preparing the Pharmacy Dictionary..."
        ):
            create_vectorstore()

    return True


initialize_database()


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("💊 Pharma Dictionary Bot")

st.caption(
    "Ask questions about pharmaceutical terminology "
    "from the Pharmacy Dictionary."
)


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Show sources directly below the answer
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.markdown("### 📚 Sources")

            for page in message["sources"]:

                st.markdown(
                    f"- **Pharmacy Dictionary** — "
                    f"Page **{page}**"
                )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

user_question = st.chat_input(
    "Ask about a pharmaceutical term..."
)


if user_question:

    # --------------------------------------------------
    # Display user question
    # --------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_question)

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })


    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the Pharmacy Dictionary..."
        ):

            answer, sources = answer_question(
                user_question
            )

        # Display answer
        st.markdown(answer)


        # --------------------------------------------------
        # Display sources directly below answer
        # --------------------------------------------------

        if sources:

            st.markdown("##### 📚 Sources")

            for page in sources:

                st.markdown(
                    f"- **Pharmacy Dictionary** — "
                    f"Page **{page}**"
                )


    # --------------------------------------------------
    # Save conversation
    # --------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
