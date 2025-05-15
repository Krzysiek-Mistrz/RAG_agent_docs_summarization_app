import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from config import get_api_key
from docs_loader import load_and_split_docs

API_KEY = get_api_key()

def create_vectorstore(docs, index_path: str = "faiss_index") -> FAISS:
    """
    Embed documents and save a FAISS index to disk.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def get_vectorstore(path: str = "faiss_index") -> FAISS:
    """
    Load a FAISS index from disk or create a new one if missing.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings)
    docs = load_and_split_docs("./docs")
    return create_vectorstore(docs, path)