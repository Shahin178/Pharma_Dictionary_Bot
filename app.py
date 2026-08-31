import os

import streamlit as st

from src.ingest import create_vectorstore, VECTORSTORE_PATH
from src.retriever import answer_question


st.set_page_config(
    page_title="Pharma Dictionary Bot",
    page_icon="💊"
)


@st.cache_resource
def initialize_database():

    if not os.path.exists(VECTORSTORE_PATH):

        with st.spinner(
            "Preparing the Pharmacy Dictionary..."
        ):
            create_vectorstore()

    return True


initialize_database()


st.title("💊 Pharma Dictionary Bot")

st.caption(
    "Ask questions about pharmaceutical terminology "
    "from the Pharmacy Dictionary."
)


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
user_question = st.chat_input(
    "Ask about a pharmaceutical term..."
)


if user_question:

    with st.chat_message("user"):
        st.markdown(user_question)

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })


    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the Pharmacy Dictionary..."
        ):

            answer = answer_question(user_question)

        st.markdown(answer)


    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


st.divider()

st.caption(
    "⚠️ This chatbot is for educational and reference purposes only. "
    "It should not be used as a substitute for professional "
    "medical or pharmaceutical advice."
)