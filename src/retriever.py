import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from src.prompt import prompt


load_dotenv()


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

VECTORSTORE_PATH = os.path.join(
    ROOT_DIR,
    "doc_vectorstore"
)


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# Answer Question
# --------------------------------------------------

def answer_question(user_question):

    # --------------------------------------------------
    # Load ChromaDB
    # --------------------------------------------------

    vectordb = Chroma(
        persist_directory=VECTORSTORE_PATH,
        collection_name="pharma_dictionary",
        embedding_function=embedding
    )


    # --------------------------------------------------
    # Similarity Search
    # --------------------------------------------------

    results = vectordb.similarity_search_with_score(
        user_question,
        k=6
    )


    # --------------------------------------------------
    # Filter relevant chunks
    # --------------------------------------------------

    documents = []

    for document, score in results:

        print(
            f"Page: {document.metadata.get('page')}, "
            f"Score: {score}"
        )

        # Lower Chroma distance = more similar
        if score < 0.8:
            documents.append(document)


    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    if not documents and results:

        # Use the best matching document
        documents = [results[0][0]]


    # --------------------------------------------------
    # Create context
    # --------------------------------------------------

    context_parts = []

    for document in documents:

        page = document.metadata.get("page")

        page_number = (
            page + 1
            if page is not None
            else "Unknown"
        )

        context_parts.append(
            f"[Page {page_number}]\n"
            f"{document.page_content}"
        )


    context = "\n\n".join(context_parts)


    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    messages = prompt.invoke({
        "context": context,
        "question": user_question
    })

    response = llm.invoke(messages)


    # --------------------------------------------------
    # Extract source pages
    # --------------------------------------------------

    sources = []

    for document in documents:

        page = document.metadata.get("page")

        if page is not None:

            page_number = page + 1

            if page_number not in sources:
                sources.append(page_number)


    return response.content, sources

