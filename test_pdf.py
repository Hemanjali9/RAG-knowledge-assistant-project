import os
import chromadb
import tiktoken
import pymupdf

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    Settings
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


print("===== RAG INGESTION STARTED =====")


# =====================================================
# 1. EMBEDDING MODEL
# =====================================================

print("\n1. Loading embedding model...")

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.embed_model = embed_model

print("Embedding model loaded successfully!")


# =====================================================
# 2. CHUNKING
# =====================================================

print("\n2. Configuring SentenceSplitter...")

tokenizer = tiktoken.get_encoding("cl100k_base").encode

splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=128,
    tokenizer=tokenizer
)

Settings.node_parser = splitter

print("SentenceSplitter configured!")


# =====================================================
# 3. LOAD PDFs USING PyMuPDF
# =====================================================

print("\n3. Loading PDFs from ./data ...")

documents = []

pdf_files = [
    file
    for file in os.listdir("data")
    if file.lower().endswith(".pdf")
]

print(f"Found {len(pdf_files)} PDF files")


for filename in pdf_files:

    pdf_path = os.path.join("data", filename)

    print(f"Processing: {filename}")

    pdf = pymupdf.open(pdf_path)

    print(f"Pages: {len(pdf)}")

    for page_number, page in enumerate(pdf):

        text = page.get_text("text")

        if text.strip():

            document = Document(
                text=text,
                metadata={
                    "file_name": filename,
                    "page_number": page_number + 1
                }
            )

            documents.append(document)

    pdf.close()


print(f"\nLoaded {len(documents)} document pages.")


# =====================================================
# 4. TEST EXTRACTED TEXT
# =====================================================

print("\n===== PDF TEXT TEST =====")

for i, doc in enumerate(documents[:3]):

    print(f"\n--- Document {i + 1} ---")

    print(
        "File:",
        doc.metadata["file_name"]
    )

    print(
        "Page:",
        doc.metadata["page_number"]
    )

    print(doc.text[:1000])


print("\n===== END PDF TEXT TEST =====")


# =====================================================
# 5. CREATE CHUNKS
# =====================================================

print("\n4. Splitting documents into chunks...")

nodes = splitter.get_nodes_from_documents(documents)

print(f"Total chunks created: {len(nodes)}")


# =====================================================
# 6. CREATE CHROMADB
# =====================================================

print("\n5. Creating ChromaDB...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    "capstone_knowledge_base"
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# =====================================================
# 7. CREATE VECTOR INDEX
# =====================================================

print("\n6. Creating vector index and embeddings...")

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model
)


# =====================================================
# 8. SUCCESS
# =====================================================

print("\n======================================")
print("SUCCESS!")
print("======================================")
print("PDF extraction : SUCCESS")
print("Chunking        : SUCCESS")
print("Embeddings      : SUCCESS")
print("ChromaDB        : SUCCESS")
print(f"Total PDFs      : {len(pdf_files)}")
print(f"Total pages     : {len(documents)}")
print(f"Total chunks    : {len(nodes)}")
print("Database        : ./chroma_db")
print("======================================")