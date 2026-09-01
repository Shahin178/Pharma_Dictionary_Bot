import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


PDF_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "pharmacy_dictionary.pdf"
)


VECTORSTORE_PATH = os.path.join(
    ROOT_DIR,
    "doc_vectorstore"
)


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vectorstore():

    print("Loading Pharmacy Dictionary...")

    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    # Add readable source name
    for document in documents:
        document.metadata["source"] = "Pharmacy Dictionary"

    # Split document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    texts = text_splitter.split_documents(documents)

    print(f"Created {len(texts)} chunks")

    # Create ChromaDB
    Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=VECTORSTORE_PATH,
        collection_name="pharma_dictionary"
    )

    print("ChromaDB created successfully!")