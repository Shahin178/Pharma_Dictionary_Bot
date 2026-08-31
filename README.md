# 💊 Pharma Dictionary RAG Chatbot

An AI-powered **Pharmaceutical Terminology Reference Chatbot** built using **Retrieval-Augmented Generation (RAG)**, **LangChain**, **ChromaDB**, **Hugging Face Embeddings**, and **Groq LLM**.

The chatbot allows users to ask questions about pharmaceutical terminology and provides answers grounded in the **Pharmacy Dictionary PDF** rather than relying solely on the language model's general knowledge.

---

## 📌 Project Overview

Pharmaceutical terminology can be complex and difficult to navigate, especially for students, researchers, pharmacists, and healthcare professionals.

This project solves that problem by creating an interactive chatbot that allows users to ask questions such as:

- What is bioavailability?
- What is enteric coating?
- Define pharmacokinetics.
- What is an excipient?
- What is the difference between a tablet and a capsule?

The chatbot retrieves relevant information from the **Pharmacy Dictionary** and provides a concise, context-based response.

> ⚠️ This application is intended for educational and reference purposes only. It should not be used as a substitute for professional medical or pharmaceutical advice.

---

# 🎯 Objectives

- Build a practical **RAG-based chatbot** using LangChain.
- Process a pharmaceutical dictionary PDF into searchable document chunks.
- Generate vector embeddings using Hugging Face.
- Store embeddings in **ChromaDB**.
- Retrieve relevant document chunks based on the user's question.
- Generate grounded answers using a **Groq-hosted LLM**.
- Provide an interactive chatbot interface using **Streamlit**.
- Deploy the application as a web application.

---

# 🏗️ RAG Architecture

```text
                    Pharmacy Dictionary PDF
                              │
                              ▼
                    PDF Document Loader
                              │
                              ▼
                     Text Extraction / OCR
                              │
                              ▼
                  Recursive Character Splitter
                              │
                              ▼
                    Document Chunks
                              │
                              ▼
                 Hugging Face Embeddings
                              │
                              ▼
                         ChromaDB
                    (Vector Database)
                              │
                              │
                    User asks a question
                              │
                              ▼
                    Query Embedding
                              │
                              ▼
                     Similarity Search
                              │
                              ▼
                     Relevant Chunks
                              │
                              ▼
                         Prompt
                              │
                              ▼
                        Groq LLM
                              │
                              ▼
                     Generated Answer
                              │
                              ▼
                       Streamlit UI
```

---

# 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with Large Language Models.

Instead of asking the LLM to answer a question entirely from its pretrained knowledge, the application first retrieves relevant information from a custom knowledge base.

The process is:

```text
User Question
      ↓
Retrieve relevant documents
      ↓
Add documents to prompt
      ↓
Send context + question to LLM
      ↓
Generate grounded answer
```

This helps reduce hallucinations and allows the chatbot to answer questions using information from the Pharmacy Dictionary.

---

# 🛠️ Technologies Used

| Technology            | Purpose                            |
| --------------------- | ---------------------------------- |
| Python                | Application development            |
| Streamlit             | Web interface                      |
| LangChain             | RAG orchestration                  |
| Unstructured          | PDF processing                     |
| Poppler               | PDF processing/rendering           |
| Tesseract OCR         | Extracting text from scanned PDFs  |
| Hugging Face          | Text embeddings                    |
| Sentence Transformers | `all-MiniLM-L6-v2` embedding model |
| ChromaDB              | Vector database                    |
| Groq                  | LLM inference                      |
| python-dotenv         | Environment variable management    |

---

# 📂 Project Structure

```text
pharma-rag-chatbot/
│
├── data/
│   └── pharmacy_dictionary.pdf
│
├── src/
│   ├── __init__.py
│   ├── ingest.py
│   ├── retriever.py
│   └── prompt.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### `data/`

Contains the Pharmacy Dictionary PDF used as the knowledge base.

### `src/ingest.py`

Responsible for:

- Loading the PDF
- Extracting document content
- Splitting the document into chunks
- Creating embeddings
- Storing embeddings in ChromaDB

### `src/retriever.py`

Responsible for:

- Loading ChromaDB
- Retrieving relevant document chunks
- Sending retrieved context to the LLM
- Generating the final response

### `src/prompt.py`

Contains the prompt template and instructions given to the LLM.

### `app.py`

Contains the Streamlit chatbot interface.

---

# 🔄 Application Workflow

## 1. Document Ingestion

The Pharmacy Dictionary PDF is loaded using:

```python
UnstructuredPDFLoader
```

For scanned documents, PDF processing can involve **Poppler and Tesseract OCR**.

---

## 2. Text Chunking

The extracted text is divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

Current configuration:

```python
chunk_size=1000
chunk_overlap=150
```

Chunking allows the retrieval system to find specific sections of the document instead of processing the entire PDF for every question.

---

## 3. Embeddings

Each document chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These vectors represent the semantic meaning of the text.

---

## 4. Vector Database

The embeddings are stored in:

```text
ChromaDB
```

This allows the application to perform semantic similarity searches.

---

## 5. Retrieval

When the user asks a question, the question is converted into an embedding and compared with the document embeddings.

The application retrieves the top relevant chunks:

```python
search_kwargs={"k": 4}
```

---

## 6. Prompt Construction

The retrieved documents are inserted into the prompt together with the user's question.

The prompt instructs the model to:

- Use only the provided context.
- Avoid making up pharmaceutical information.
- Say when information cannot be found.
- Avoid diagnosis or treatment recommendations.

---

## 7. LLM Response

The retrieved context and question are sent to the Groq LLM.

The application then displays the generated response in the Streamlit chatbot.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Shahin178/Pharma_Dictionary_Bot
```

Navigate into the project:

```bash
cd Pharma_Dictionary_Bot
```

---

# 🐍 2. Create a Conda Environment

```bash
conda create -n pdfqa python=3.11
```

Activate it:

```bash
conda activate pdfqa
```

---

# 📦 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

# 📄 4. Install Poppler

If using Conda:

```bash
conda install -c conda-forge poppler
```

Verify the installation:

```bash
pdfinfo -v
```

---

# 🔤 5. Install Tesseract OCR

Using Conda:

```bash
conda install -c conda-forge tesseract
```

Verify:

```bash
tesseract --version
```

---

# 🔑 6. Configure Groq API Key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

The `.env` file should **never be committed to GitHub**.

Add it to `.gitignore`:

```text
.env
```

---

# 🚀 Running the Application

From the project root:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 💬 Example Questions

Try asking:

```text
What is bioavailability?
```

```text
What is pharmacokinetics?
```

```text
Define enteric coating.
```

```text
What is an excipient?
```

```text
What is the purpose of a tablet coating?
```

```text
What is the difference between pharmacokinetics and pharmacodynamics?
```

---

# 🔐 Hallucination Control

The chatbot uses a controlled prompt that instructs the LLM to answer only from retrieved information.

If the information cannot be found in the Pharmacy Dictionary, the chatbot responds:

```text
I couldn't find this information in the Pharmacy Dictionary.
```

This helps prevent the model from presenting unsupported pharmaceutical information as fact.

---

# 🧪 Example RAG Flow

For the question:

```text
What is bioavailability?
```

The system performs:

```text
Question
   ↓
Embedding
   ↓
ChromaDB similarity search
   ↓
Top 4 relevant chunks
   ↓
Context + Question
   ↓
Groq LLM
   ↓
Answer
```

---

# 📋 Requirements

The main Python dependencies are:

```text
streamlit
python-dotenv
langchain-community
langchain-text-splitters
langchain-huggingface
chromadb
langchain-chroma
langchain-groq
sentence-transformers
unstructured
unstructured[pdf]
nltk
```

---

# 🔮 Future Improvements

Possible improvements include:

- Add source/page references to responses.
- Display retrieved document sections.
- Add conversation memory.
- Add streaming LLM responses.
- Improve PDF chunking based on dictionary terms.
- Add hybrid search using keyword + semantic search.
- Add reranking for retrieved documents.
- Add multiple pharmaceutical reference documents.
- Add document metadata filtering.
- Add authentication.
- Add evaluation metrics for RAG retrieval and answer quality.
- Add automated ingestion for new reference documents.

---

# 📊 Key Skills Demonstrated

This project demonstrates practical experience with:

- Python
- LangChain
- Retrieval-Augmented Generation
- Large Language Models
- Prompt Engineering
- Vector Databases
- ChromaDB
- Semantic Search
- Hugging Face Embeddings
- Sentence Transformers
- PDF Processing
- OCR
- Tesseract
- Poppler
- Streamlit
- API Integration
- Environment Variables
- Git/GitHub
- Cloud Deployment

---

# ⚠️ Disclaimer

This chatbot is designed for **educational and pharmaceutical terminology reference purposes only**.

It does not provide medical diagnosis, prescriptions, dosage recommendations, treatment decisions, or professional medical advice.

Always consult a qualified healthcare or pharmaceutical professional for medical decisions.

---

# 👩‍💻 Author

**Shahin Bano**

GitHub: `https://github.com/Shahin178`

---

## ⭐ If you found this project useful

Feel free to ⭐ the repository and use the project as a reference for learning about **RAG, LangChain, vector databases, and LLM applications**.
