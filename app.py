import os
import streamlit as st
import chromadb
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore


# =========================================================
# 1. LOAD ENVIRONMENT
# =========================================================

load_dotenv(override=True)


# =========================================================
# 2. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Team RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 3. SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# 4. SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🧠 Team RAG Assistant")

    st.caption(
        "Intelligent Knowledge Assistant"
    )

    st.divider()

    st.success(
        "RAG System Online"
    )

    st.info(
        "📚 Vector Database\n\nChromaDB"
    )

    st.info(
        "⚡ Language Model\n\nGroq LLM"
    )

    st.divider()

    st.subheader("💬 Conversation")

    st.write(
        f"Questions asked: {len(st.session_state.history)}"
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.history = []

        st.rerun()

    st.divider()

    st.caption(
        "LlamaIndex • ChromaDB • HuggingFace • Groq"
    )


# =========================================================
# 5. MAIN HEADER
# =========================================================

st.title("🧠 Team Knowledge Assistant")

st.write(
    "Ask questions about your documents and get "
    "accurate, context-grounded answers."
)

st.caption(
    "✨ RAG-Powered Knowledge System"
)

st.divider()


# =========================================================
# 6. LOAD QUERY ENGINE
# =========================================================

@st.cache_resource
def load_query_engine():

    # -----------------------------------------------------
    # EMBEDDING MODEL
    # -----------------------------------------------------

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Settings.embed_model = embed_model


    # -----------------------------------------------------
    # GROQ LLM
    # -----------------------------------------------------

    api_key = os.getenv("GROQ_API_KEY")

    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing from .env"
        )


    llm = OpenAILike(
        model=model,
        api_base="https://api.groq.com/openai/v1",
        api_key=api_key,
        is_chat_model=True,
        is_function_calling_model=False,
    )

    Settings.llm = llm


    # -----------------------------------------------------
    # CHROMADB
    # -----------------------------------------------------


    CHROMA_DB_PATH = os.getenv(
        "CHROMA_DB_PATH",
        "./chroma_db"
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    collection = client.get_or_create_collection(
        "capstone_knowledge_base"
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )


    # -----------------------------------------------------
    # VECTOR INDEX
    # -----------------------------------------------------

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )


    # -----------------------------------------------------
    # RAG PROMPT
    # -----------------------------------------------------

    qa_template = PromptTemplate(
        """
You are a precise research assistant.

Answer the user's question using ONLY the provided context.

Rules:

1. First determine whether the context contains enough information.
2. If the answer is present, answer directly and confidently.
3. Preserve mathematical formulas exactly when they appear in the context.
4. Do not invent, reconstruct, or guess formulas.
5. If multiple sources provide information, combine them carefully.
6. Explain technical concepts clearly in simple language.
7. For mathematical questions, show the formula first and then explain each term.
8. For comparison questions, use a concise table when useful.
9. If the context does not contain the answer, say:

"The answer is not available in the provided documents."

10. Do not mention the retrieval process unless asked.

Context:
{context_str}

Question:
{query_str}

Answer:
"""
    )


    # -----------------------------------------------------
    # QUERY ENGINE
    # -----------------------------------------------------

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        text_qa_template=qa_template
    )

    return query_engine


# =========================================================
# 7. INITIALIZE RAG
# =========================================================

if os.getenv("RAG_TEST_MODE") == "1":
    # Skip model downloads and external API setup during tests.
    query_engine = None
else:
    try:
        query_engine = load_query_engine()
    except Exception as e:
        query_engine = None
        st.error(f"Failed to initialize RAG system: {e}")


# =========================================================
# 8. WELCOME SCREEN
# =========================================================

if not st.session_state.history:

    st.subheader(
        "💡 What would you like to know?"
    )

    st.write(
        "Ask a question about the documents in "
        "your knowledge base."
    )

    st.write("")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.info(
            """
            📚 **Document Grounded**

            Answers are generated using
            information from your documents.
            """
        )


    with col2:

        st.info(
            """
            🔍 **Smart Retrieval**

            Relevant document chunks are
            retrieved before generating answers.
            """
        )


    with col3:

        st.info(
            """
            ⚡ **Fast AI Answers**

            Powered by Groq for fast,
            context-aware responses.
            """
        )


    st.divider()


# =========================================================
# 9. PREVIOUS CHAT HISTORY
# =========================================================

for question_text, answer_text, sources in st.session_state.history:

    # USER
    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.write(question_text)


    # ASSISTANT
    with st.chat_message(
        "assistant",
        avatar="🧠"
    ):

        st.subheader("✨ Answer")

        st.markdown(
            answer_text
        )


        if sources:

            st.markdown(
                "#### 📚 Sources"
            )

            for source in sources:

                st.write(
                    f"📄 {source}"
                )


# =========================================================
# 10. CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask anything about your knowledge base..."
)


if question:

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.write(question)


    # -----------------------------------------------------
    # ASSISTANT MESSAGE
    # -----------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🧠"
    ):

        with st.spinner(
            "🔎 Searching your knowledge base..."
        ):

            try:

                # -------------------------------------------------
                # QUERY RAG
                # -------------------------------------------------

                response = query_engine.query(
                    question
                )


                # -------------------------------------------------
                # ANSWER
                # -------------------------------------------------

                answer = str(response)


                # Convert LaTeX delimiters

                answer = answer.replace(
                    "\\[",
                    "$$"
                )

                answer = answer.replace(
                    "\\]",
                    "$$"
                )

                answer = answer.replace(
                    "\\(",
                    "$"
                )

                answer = answer.replace(
                    "\\)",
                    "$"
                )


                st.subheader(
                    "✨ Answer"
                )

                st.markdown(
                    answer
                )


                # -------------------------------------------------
                # RETRIEVED CHUNKS
                # -------------------------------------------------

                with st.expander(
                    "🔍 View Retrieved Chunks"
                ):

                    st.caption(
                        f"{len(response.source_nodes)} "
                        "relevant chunks retrieved"
                    )


                    for i, node in enumerate(
                        response.source_nodes
                    ):

                        st.markdown(
                            f"#### Chunk {i + 1}"
                        )

                        st.write(
                            node.get_content()
                        )

                        if i < len(
                            response.source_nodes
                        ) - 1:

                            st.divider()


                # -------------------------------------------------
                # SOURCES
                # -------------------------------------------------

                sources = sorted(
                    {
                        n.metadata.get(
                            "file_name",
                            "unknown"
                        )
                        for n in response.source_nodes
                    }
                )


                if sources:

                    st.markdown(
                        "#### 📚 Sources"
                    )

                    for source in sources:

                        st.write(
                            f"📄 {source}"
                        )


                # -------------------------------------------------
                # SAVE HISTORY
                # -------------------------------------------------

                st.session_state.history.append(
                    (
                        question,
                        answer,
                        sources
                    )
                )


            except Exception as e:

                st.error(
                    f"Error while generating the answer: {e}"
                )


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.caption(
    "Team RAG Knowledge Assistant • "
    "Context-Grounded AI • "
    "LlamaIndex + ChromaDB + Groq"
)