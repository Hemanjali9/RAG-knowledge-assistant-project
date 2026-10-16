
import os
from unittest.mock import Mock, patch

import app
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms.mock import MockLLM


def test_load_query_engine_uses_expected_configuration():
    test_env = {
        "GROQ_API_KEY": "test-key",
        "GROQ_MODEL": "test-model",
        "CHROMA_DB_PATH": "./test_chroma_db",
        "RAG_TEST_MODE": "1",
    }

    with patch.dict(os.environ, test_env):
        fake_embedding = MockEmbedding(embed_dim=8)
        fake_llm = MockLLM()

        with (
            patch.object(
                app, "HuggingFaceEmbedding", return_value=fake_embedding
            ) as mock_embedding,
            patch.object(
                app, "OpenAILike", return_value=fake_llm
            ) as mock_llm_class,
            patch.object(
                app.chromadb, "PersistentClient"
            ) as mock_client_class,
            patch.object(app, "ChromaVectorStore") as mock_store_class,
            patch.object(
                app.VectorStoreIndex, "from_vector_store"
            ) as mock_index_factory,
        ):
            mock_collection = Mock()
            mock_client = mock_client_class.return_value
            mock_client.get_or_create_collection.return_value = mock_collection

            mock_store = mock_store_class.return_value
            mock_index = mock_index_factory.return_value
            expected_engine = Mock()
            mock_index.as_query_engine.return_value = expected_engine

            # Run the actual function without Streamlit's resource cache.
            engine = app.load_query_engine.__wrapped__()

    assert engine is expected_engine

    mock_embedding.assert_called_once_with(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    mock_llm_class.assert_called_once_with(
        model="test-model",
        api_base="https://api.groq.com/openai/v1",
        api_key="test-key",
        is_chat_model=True,
        is_function_calling_model=False,
    )

    mock_client_class.assert_called_once_with(
        path="./test_chroma_db"
    )
    mock_client.get_or_create_collection.assert_called_once_with(
        "capstone_knowledge_base"
    )
    mock_store_class.assert_called_once_with(
        chroma_collection=mock_collection
    )
    mock_index_factory.assert_called_once_with(
        mock_store,
        embed_model=fake_embedding,
    )

    mock_index.as_query_engine.assert_called_once()
    query_kwargs = mock_index.as_query_engine.call_args.kwargs

    assert query_kwargs["llm"] is fake_llm
    assert query_kwargs["similarity_top_k"] == 5

    prompt = query_kwargs["text_qa_template"].template.lower()
    assert (
        "the answer is not available in the provided documents"
        in prompt
    )
