import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from src.prompt import prompt


load_dotenv()


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


VECTORSTORE_PATH = os.path.join(
    ROOT_DIR,
    "doc_vectorstore"
)


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


def answer_question(user_question):

    # Load ChromaDB
    vectordb = Chroma(
        persist_directory=VECTORSTORE_PATH,
        collection_name="pharma_dictionary",
        embedding_function=embedding
    )

    # Create retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4}
    )

    # Retrieve relevant documents
    documents = retriever.invoke(user_question)

    # Combine retrieved chunks
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create prompt
    messages = prompt.invoke({
        "context": context,
        "question": user_question
    })

    # Generate answer
    response = llm.invoke(messages)

    return response.content