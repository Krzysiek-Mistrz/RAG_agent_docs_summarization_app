from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

def build_qa_chain(vectorstore, api_key: str):
    """
    Create a RetrievalQA chain with map-reduce summarization.
    
    Args:
        vectorstore: FAISS vectorstore for retrieval.
        api_key: OpenAI API key.
        
    Returns:
        A RetrievalQA chain instance.
    """
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"map_prompt": None, "combine_prompt": None}
    )
    return qa_chain
