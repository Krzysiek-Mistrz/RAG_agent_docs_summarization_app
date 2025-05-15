from os import path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from src.docs_loader import load_and_split_docs

def create_vectorstore(docs, api_key: str, index_path: str = "faiss_index") -> FAISS:
    """
    Embed documents and save a FAISS index to disk.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def get_vectorstore(
    api_key: str,
    index_path: str = "faiss_index",
    docs_path: str = "./docs"
) -> FAISS:
    """
    Load a FAISS index from disk or create a new one if missing.

    Args:
        api_key: OpenAI API key.
        index_path: Local path to save/load the FAISS index.
        docs_path: Directory containing PDF docs to load & split.

    Raises:
        FileNotFoundError: if docs_path does not exist.
        ValueError: if no documents are found under docs_path.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    if path.exists(index_path):
        return FAISS.load_local(index_path, embeddings)

    if not path.isdir(docs_path):
        raise FileNotFoundError(f"No docs directory found at '{docs_path}'")

    docs = load_and_split_docs(docs_path)
    if not docs:
        raise ValueError(f"No documents found in '{docs_path}'")

    return create_vectorstore(docs, api_key, index_path)
