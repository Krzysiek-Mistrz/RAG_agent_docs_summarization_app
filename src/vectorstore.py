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

def get_vectorstore(api_key: str, index_path: str = "faiss_index") -> FAISS:
    """
    Load a FAISS index from disk or create a new one if missing.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    if path.exists(index_path):
        return FAISS.load_local(index_path, embeddings)
    docs = load_and_split_docs("./docs")
    return create_vectorstore(docs, api_key, index_path)
