from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from config import get_api_key

API_KEY = get_api_key()

def build_qa_chain(vectorstore):
    """
    Create a RetrievalQA chain with map-reduce summarization.
    """
    llm = OpenAI(temperature=0, openai_api_key=API_KEY)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"map_prompt": None, "combine_prompt": None}
    )
    return qa_chain