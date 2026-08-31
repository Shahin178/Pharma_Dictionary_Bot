import os

from langchain_community.document_loaders import UnstructuredPDFLoader
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

    loader = UnstructuredPDFLoader(PDF_PATH)

    documents = loader.load()

    print(f"Loaded {len(documents)} document(s)")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    texts = text_splitter.split_documents(documents)

    print(f"Created {len(texts)} chunks")

    Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=VECTORSTORE_PATH,
        collection_name="pharma_dictionary"
    )

    print("ChromaDB created successfully!")