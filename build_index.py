import os
import fitz
import chromadb
import tiktoken

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    Settings
)

from chunking_utils import create_text_splitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# =========================================================
# 1. EMBEDDING MODEL
# =========================================================

print("1. Loading embedding model...")

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.embed_model = embed_model


# =========================================================
# 2. CHUNKING
# =========================================================

print("2. Configuring SentenceSplitter...")

tokenizer = tiktoken.get_encoding("cl100k_base").encode

splitter = create_text_splitter(
    chunk_size=1200,
    chunk_overlap=250
)
Settings.node_parser = splitter


# =========================================================
# 3. LOAD PDFs WITH PYMUPDF
# =========================================================

print("3. Loading PDFs using PyMuPDF...")

data_dir = "data"

documents = []

pdf_files = sorted(
    [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]
)

print(f"Found {len(pdf_files)} PDF files.")


for pdf_file in pdf_files:

    pdf_path = os.path.join(data_dir, pdf_file)

    print(f"\nReading: {pdf_file}")

    pdf = fitz.open(pdf_path)

    print(f"Pages: {len(pdf)}")

    for page_number, page in enumerate(pdf):

        text = page.get_text("text").strip()

        if not text:
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "file_name": pdf_file,
                    "page_number": page_number + 1
                }
            )
        )

    pdf.close()


print("\n========================================")
print(f"Total pages with text: {len(documents)}")
print("========================================")


# =========================================================
# 4. VERIFY A-MEM EXTRACTION
# =========================================================

print("\n===== A-MEM EXTRACTION TEST =====")

amem_docs = [
    doc
    for doc in documents
    if doc.metadata.get("file_name") == "07_Xu_2025_AMem.pdf"
]


print(f"A-MEM pages loaded: {len(amem_docs)}")


if amem_docs:

    print("\nFirst A-MEM page text:\n")
    print(amem_docs[0].text[:3000])

    # Search for Zettelkasten
    zettelkasten_found = False

    for doc in amem_docs:

        if "zettelkasten" in doc.text.lower():

            zettelkasten_found = True

            print("\n========================================")
            print("ZETTELKASTEN FOUND!")
            print("========================================")

            print(
                f"Page: {doc.metadata.get('page_number')}"
            )

            index_position = doc.text.lower().find(
                "zettelkasten"
            )

            start = max(0, index_position - 500)
            end = min(
                len(doc.text),
                index_position + 1500
            )

            print(
                doc.text[start:end]
            )

            print("========================================")

            break

    if not zettelkasten_found:

        print(
            "\nWARNING: 'Zettelkasten' was not found "
            "in extracted A-MEM text."
        )

else:

    print(
        "ERROR: 07_Xu_2025_AMem.pdf was not loaded."
    )


print("\n===== END A-MEM TEST =====")


# =========================================================
# 5. CREATE CHUNKS
# =========================================================

print("\n4. Creating chunks...")

nodes = splitter.get_nodes_from_documents(
    documents
)

print(
    f"Total chunks created: {len(nodes)}"
)


# =========================================================
# 6. CHECK A-MEM CHUNKS
# =========================================================

amem_chunks = [
    node
    for node in nodes
    if node.metadata.get("file_name")
    == "07_Xu_2025_AMem.pdf"
]

print(
    f"A-MEM chunks created: {len(amem_chunks)}"
)


# =========================================================
# 7. CREATE FRESH CHROMADB
# =========================================================

print("\n5. Creating fresh ChromaDB...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection_name = "capstone_knowledge_base"


try:

    client.delete_collection(
        collection_name
    )

    print("Old collection deleted.")

except Exception:

    print(
        "No existing collection found."
    )


collection = client.create_collection(
    collection_name
)

print("New collection created.")


# =========================================================
# 8. VECTOR STORE
# =========================================================

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# =========================================================
# 9. BUILD VECTOR INDEX
# =========================================================

print("\n6. Creating vector index...")

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model
)


# =========================================================
# 10. FINAL RESULT
# =========================================================

print("\n")
print("========================================")
print("       RAG INDEX BUILD SUCCESSFUL")
print("========================================")
print(
    f"PDF files          : {len(pdf_files)}"
)
print(
    f"Text pages         : {len(documents)}"
)
print(
    f"Total chunks       : {len(nodes)}"
)
print(
    f"A-MEM chunks       : {len(amem_chunks)}"
)
print(
    f"Chroma collection  : {collection_name}"
)
print("========================================")

print("\nYou can now run:")
print("streamlit run app.py")