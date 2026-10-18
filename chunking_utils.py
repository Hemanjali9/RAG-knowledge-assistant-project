

import tiktoken
from llama_index.core.node_parser import SentenceSplitter


def create_text_splitter(
    chunk_size=1200,
    chunk_overlap=250
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    tokenizer = tiktoken.get_encoding("cl100k_base").encode

    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer
    )
