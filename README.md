# 🧠 RAG-Powered Team Knowledge Assistant

> **An intelligent document question-answering system built using Retrieval-Augmented Generation (RAG), allowing users to ask natural-language questions and receive accurate, context-grounded answers from uploaded PDF documents.**

---

## 🚀 Project Overview

The **RAG-Powered Team Knowledge Assistant** is an AI-based document intelligence system that allows users to interact with their knowledge base using natural language.

Instead of manually searching through lengthy PDF documents, users can simply ask a question. The system retrieves the most relevant information from the uploaded documents and provides an AI-generated answer based only on the retrieved context.

The project combines:

* 📄 PDF Document Ingestion
* ✂️ Intelligent Text Chunking
* 🔢 Vector Embeddings
* 🗄️ ChromaDB Vector Database
* 🔍 Semantic Retrieval
* 🤖 Large Language Model (LLM)
* 🧠 Retrieval-Augmented Generation (RAG)
* 💬 Streamlit Chat Interface

---

## 🎯 Problem Statement

Organizations and teams often store large amounts of information in PDF documents, research papers, reports, manuals, and internal documentation.

Finding specific information manually can be:

* Time-consuming
* Difficult for large documents
* Inefficient
* Prone to missing relevant information

Traditional keyword search may also fail when the user's question and the document use different wording.

### 💡 Solution

This project provides an AI-powered knowledge assistant that understands the meaning of a user's question, retrieves semantically relevant document sections, and generates a grounded answer using the retrieved context.

---

## ✨ Key Features

### 📚 Document-Based Question Answering

Ask questions about the uploaded PDF documents using natural language.

### 🔍 Semantic Search

The system uses embeddings to understand the semantic meaning of queries rather than relying only on exact keywords.

### 🧠 Retrieval-Augmented Generation

Relevant document chunks are retrieved first and then provided to the LLM as context.

### 🛡️ Context-Grounded Answers

The system is instructed to answer using the available document context and avoid unsupported information.

### 👀 Retrieved Chunk Visibility

Users can inspect the retrieved chunks used to generate an answer.

### 💬 Modern Chat Interface

A clean Streamlit interface provides an interactive ChatGPT-style experience.

### 📄 PDF Support

The system automatically loads PDF documents from the configured data directory.

---

# 🏗️ System Architecture

```text
                  ┌─────────────────────┐
                  │     PDF Documents   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    PDF Ingestion    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Text Chunking     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Embedding Model     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │      ChromaDB       │
                  │   Vector Database   │
                  └──────────┬──────────┘
                             │
              User Question │
                             ▼
                  ┌─────────────────────┐
                  │ Query Embedding     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Semantic Retrieval  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Retrieved Context   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │       LLM           │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Grounded Answer     │
                  └─────────────────────┘
```

---

# 🔄 RAG Pipeline

The complete pipeline works in the following stages:

### 1️⃣ PDF Ingestion

PDF files are loaded from the project's data directory.

```text
data/
├── document1.pdf
├── document2.pdf
└── document3.pdf
```

The PDF text is extracted and converted into documents that can be processed by the RAG pipeline.

---

### 2️⃣ Text Chunking

Large documents are divided into smaller chunks.

```text
PDF Document
     ↓
Extracted Text
     ↓
Text Chunks
     ↓
Embedding
```

Chunking allows the retrieval system to identify specific sections relevant to a user's question.

---

### 3️⃣ Embeddings

Each text chunk is converted into a numerical vector representation using an embedding model.

```text
Text
 ↓
Embedding Model
 ↓
Vector Representation
```

This allows the system to compare the semantic similarity between a user's question and document content.

---

### 4️⃣ ChromaDB

The generated embeddings are stored in **ChromaDB**, a vector database designed for similarity search.

ChromaDB stores:

* Document chunks
* Vector embeddings
* Metadata

---

### 5️⃣ Query Processing

When a user asks a question:

```text
User Question
      ↓
Query Embedding
      ↓
Similarity Search
      ↓
Relevant Chunks
```

The system searches the vector database for the most relevant document chunks.

---

### 6️⃣ Context Retrieval

The top relevant chunks are selected and passed to the language model as context.

```text
Question
+
Retrieved Context
        ↓
       LLM
```

---

### 7️⃣ Answer Generation

The LLM generates a natural-language response based on the retrieved context.

If the required information cannot be found in the documents, the system can respond that the answer is not available in the provided documents instead of unnecessarily generating unsupported information.

---

# 🛠️ Technology Stack

| Technology                     | Purpose                              |
| ------------------------------ | ------------------------------------ |
| Python                         | Core programming language            |
| Streamlit                      | User interface                       |
| LlamaIndex                     | RAG pipeline and document processing |
| ChromaDB                       | Vector database                      |
| Hugging Face / Embedding Model | Text embeddings                      |
| Groq / LLM                     | Answer generation                    |
| PyMuPDF / PDF Loader           | PDF processing                       |
| dotenv                         | Environment variable management      |

---


# 📁 Project Structure

```text

RAG_PRO/
├── data/                  # Source PDF documents
├── chroma_db/             # Local vector database (generated)
├── app.py                 # Streamlit application
├── build_index.py         # Build the document index
├── generate_pdf.py        # PDF generation utility
├── query_rag.py           # Query the knowledge base
├── rag_answer.py          # RAG answer generation
├── test_pdf.py            # PDF-related test script
├── test_pymupdf.py        # PyMuPDF test script
├── RAG_Capstone_Project_Report.pdf
├── README.md
└── .gitignore
```

The `chroma_db/` directory is generated when the vector database is created. The `.venv/`, `.env`, cache directories, and other local development files should not be committed to the repository.

> The exact file structure may vary depending on the implementation.

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG_PRO
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=your_model_name
```

### ⚠️ Security

Never commit your actual API key to GitHub.

Add the following to `.gitignore`:

```text
.env
.venv/
__pycache__/
*.pyc
chroma_db/
```

---

# 📄 Add Documents

Place your PDF files inside the `data` directory.

```text
data/
├── research_paper.pdf
├── technical_document.pdf
└── project_report.pdf
```

---

# ▶️ Run the Application

Make sure Python and the required project dependencies are installed.

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open the URL shown in your terminal to interact with the knowledge assistant.

Press `Ctrl+C` in the terminal to stop the application.

---

# 🧪 Automated Testing

The repository includes PDF-related test scripts:

- `test_pdf.py`
- `test_pymupdf.py`

Run an individual script using Python:

```bash
python test_pdf.py
```

```bash
python test_pymupdf.py
```

These scripts can help verify PDF-processing behavior in the local environment. Install the required dependencies before running them.

---

# 💬 Example Questions

Users can ask questions such as:

```text
What is the main objective of the document?
```

```text
How does the proposed system work?
```

```text
What are the key challenges discussed?
```

```text
What methodology was used?
```

```text
What are the major findings?
```

```text
What are the future improvements mentioned in the document?
```

---

# 🧪 Example RAG Flow

### User Question

```text
How does the system handle context overflow?
```

### Retrieval

The vector database searches for document chunks semantically related to context overflow.

### Retrieved Context

```text
Relevant document chunks
        ↓
Context construction
```

### LLM

```text
Question + Retrieved Context
              ↓
             LLM
```

### Final Answer

```text
The system manages context overflow by monitoring
the available context window and moving or retrieving
information according to its memory-management strategy.
```

---

# 📊 Evaluation

The system can be evaluated using:

### 1. Retrieval Accuracy

Measures whether the relevant document chunks are retrieved.

### 2. Answer Relevance

Measures whether the generated answer directly addresses the user's question.

### 3. Groundedness

Checks whether the answer is supported by the retrieved document context.

### 4. Response Quality

Evaluates clarity, completeness, and usefulness of the generated response.

---

# ⚙️ Challenges & Solutions

| Challenge                   | Solution                        |
| --------------------------- | ------------------------------- |
| Extracting text from PDFs   | Used PDF document loaders       |
| Large document size         | Applied text chunking           |
| Keyword search limitations  | Used semantic embeddings        |
| Efficient similarity search | Used ChromaDB                   |
| Hallucinated answers        | Used context-grounded prompting |
| API configuration           | Used environment variables      |
| User interaction            | Built Streamlit interface       |
| Inspecting retrieval        | Added retrieved chunk display   |

---

# 🔒 Hallucination Reduction

One of the important goals of the system is to reduce unsupported AI responses.

The RAG pipeline follows:

```text
User Question
      ↓
Retrieve Relevant Documents
      ↓
Build Context
      ↓
Send Context + Question to LLM
      ↓
Generate Grounded Answer
```

The LLM is instructed to prioritize the retrieved context rather than relying on unrelated knowledge.

---

# 🌟 Advantages

* ✅ Faster information discovery
* ✅ Natural-language interaction
* ✅ Semantic document search
* ✅ Reduces manual document reading
* ✅ Context-grounded responses
* ✅ Supports multiple PDF documents
* ✅ Easy-to-use web interface
* ✅ Extensible architecture

---

# 🔮 Future Scope

The project can be extended with:

* 📑 Support for DOCX, TXT, CSV and web pages
* 👥 Multi-user knowledge bases
* 🔐 Authentication and authorization
* 🗂️ Document management dashboard
* 🧾 Source citations in answers
* 📊 Retrieval and answer evaluation dashboard
* 💾 Conversation history
* 🌐 Cloud deployment
* 🔄 Automatic document indexing
* 🎙️ Voice-based questions
* 🌍 Multilingual document Q&A
* 🧠 Advanced reranking models
* 📈 Analytics and usage monitoring

---

# 🎓 Project Objectives

The major objectives of this project are:

1. Build an intelligent document question-answering system.
2. Implement a complete Retrieval-Augmented Generation pipeline.
3. Convert document text into meaningful vector representations.
4. Store and retrieve document embeddings efficiently.
5. Generate context-aware answers using an LLM.
6. Provide an intuitive user interface for interacting with documents.
7. Reduce unsupported responses by grounding answers in retrieved content.

---

# 📌 Conclusion

The **RAG-Powered Team Knowledge Assistant** demonstrates how Large Language Models can be combined with vector databases and semantic retrieval to build a practical document intelligence system.

Instead of asking an LLM to answer questions from its general knowledge alone, the system first retrieves relevant information from the user's documents and then uses that information to generate a grounded response.

This architecture provides a strong foundation for building enterprise knowledge assistants, research assistants, internal documentation tools, and intelligent document search systems.

---

# 👩‍💻 Author

**Hemanjali Jonnalagadda**

### Project

**RAG-Powered Team Knowledge Assistant**

### Domain

**Artificial Intelligence | Generative AI | Retrieval-Augmented Generation | NLP**

---

## ⭐ If you found this project useful

Give the repository a ⭐ on GitHub!
