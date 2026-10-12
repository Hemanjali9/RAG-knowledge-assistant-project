
from llama_index.core import Document
from chunking_utils import create_text_splitter


def test_splitter_returns_chunks():
    splitter = create_text_splitter(
        chunk_size=100,
        chunk_overlap=10
    )

    document = Document(
        text="This is a test sentence. " * 100
    )

    chunks = splitter.get_nodes_from_documents([document])

    assert len(chunks) > 1


def test_chunks_contain_original_text():
    splitter = create_text_splitter(
        chunk_size=100,
        chunk_overlap=10
    )

    document = Document(
        text="The RAG system retrieves relevant documents."
    )

    chunks = splitter.get_nodes_from_documents([document])

    assert len(chunks) >= 1
    assert "RAG system" in " ".join(
        chunk.get_content() for chunk in chunks
    )


def test_splitter_rejects_invalid_overlap():
    try:
        create_text_splitter(
            chunk_size=100,
            chunk_overlap=100
        )
    except (ValueError, AssertionError):
        return

    assert False, "Expected invalid chunk configuration to fail"
