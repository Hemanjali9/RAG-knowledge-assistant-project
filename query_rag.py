import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# 1. Load embedding model
print("1. Loading embedding model...")

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.embed_model = embed_model

print("Embedding model loaded!")


# 2. Connect to existing ChromaDB
print("\n2. Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "capstone_knowledge_base"
)

print(f"ChromaDB contains {collection.count()} vectors.")


# 3. Connect ChromaDB with LlamaIndex
vector_store = ChromaVectorStore(
    chroma_collection=collection
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)


# 4. Create retriever
print("\n3. Creating retriever...")

retriever = index.as_retriever(
    similarity_top_k=5
)

print("Retriever ready!")


# 5. Ask a question
question = input(
    "\nEnter your question: "
)


# 6. Retrieve relevant chunks
print("\n4. Searching relevant chunks...")

results = retriever.retrieve(question)


# 7. Display results
print("\n===== RETRIEVED CHUNKS =====")

for i, result in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print("Score:", result.score)

    print("Text:")
    print(result.node.text[:1000])

    print("Metadata:")
    print(result.node.metadata)


print("\n===== RETRIEVAL COMPLETE =====")