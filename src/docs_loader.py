from langchain.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_and_split_docs(path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Load all PDF documents under 'path' and split into text chunks.
    """
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)