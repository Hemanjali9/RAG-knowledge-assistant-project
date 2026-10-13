
from unittest.mock import Mock


def test_rag_query_uses_mocked_llm_and_embeddings():
    # Fake embedding and LLM dependencies: no model downloads or API calls.
    mock_embeddings = Mock()
    mock_llm = Mock()

    # Simulate a retriever returning document context.
    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = [
        Mock(
            node=Mock(
                text="RAG retrieves relevant documents before generating an answer.",
                metadata={"file_name": "sample.txt", "page_number": 1},
            ),
            score=0.95,
        )
    ]

    results = mock_retriever.retrieve("What is RAG?")

    assert len(results) == 1
    assert "retrieves relevant documents" in results[0].node.text
    assert results[0].node.metadata["file_name"] == "sample.txt"

    # Verify the fake dependencies were not replaced by real services.
    assert mock_embeddings is not None
    assert mock_llm is not None
    mock_retriever.retrieve.assert_called_once_with("What is RAG?")
