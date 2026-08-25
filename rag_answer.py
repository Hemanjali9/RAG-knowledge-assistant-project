import os
import chromadb
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from groq import Groq


# =====================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

if not GROQ_MODEL:
    raise ValueError("GROQ_MODEL not found in .env")

print("Groq configuration loaded!")
print("Model:", GROQ_MODEL)


# =====================================================
# 2. LOAD EMBEDDING MODEL
# =====================================================

print("\n1. Loading embedding model...")

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.embed_model = embed_model

print("Embedding model loaded!")


# =====================================================
# 3. CONNECT TO CHROMADB
# =====================================================

print("\n2. Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "capstone_knowledge_base"
)

print(f"ChromaDB contains {collection.count()} vectors.")


# =====================================================
# 4. CREATE VECTOR INDEX
# =====================================================

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

retriever = index.as_retriever(
    similarity_top_k=5
)

print("Retriever ready!")


# =====================================================
# 5. CONNECT TO GROQ
# =====================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)

print("Groq client ready!")


# =====================================================
# 6. GET USER QUESTION
# =====================================================

question = input(
    "\nEnter your question: "
)


# =====================================================
# 7. RETRIEVE RELEVANT CHUNKS
# =====================================================

print("\nSearching knowledge base...")

results = retriever.retrieve(question)

print(f"Retrieved {len(results)} relevant chunks.")


# =====================================================
# 8. BUILD CONTEXT
# =====================================================

context_parts = []

for i, result in enumerate(results):

    context_parts.append(
        f"""
SOURCE {i + 1}
File: {result.node.metadata.get("file_name", "Unknown")}
Page: {result.node.metadata.get("page_number", "Unknown")}

{result.node.text}
"""
    )

context = "\n\n".join(context_parts)


# =====================================================
# 9. CREATE RAG PROMPT
# =====================================================

prompt = f"""
You are a helpful research assistant.

Answer the user's question using ONLY the information
provided in the retrieved context below.

If the answer cannot be found in the context,
say clearly that the information is not available
in the provided documents.

User Question:
{question}

Retrieved Context:
{context}

Instructions:
- Give a clear and accurate answer.
- Do not invent information.
- Explain the answer in simple language.
- Mention the relevant source file and page when possible.
"""


# =====================================================
# 10. SEND TO GROQ
# =====================================================

print("\nGenerating answer using Groq...")

response = groq_client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a precise document-based RAG assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.2
)


# =====================================================
# 11. DISPLAY FINAL ANSWER
# =====================================================

answer = response.choices[0].message.content

print("\n")
print("=" * 60)
print("FINAL RAG ANSWER")
print("=" * 60)

print(answer)

print("=" * 60)
print("RAG QUERY COMPLETE")
print("=" * 60)