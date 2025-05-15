from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from config import get_api_key

API_KEY = get_api_key()

def initialize_summarization_agent(qa_chain):
    """
    Wrap the RetrievalQA chain into an agent for interactive queries.
    """
    tools = [
        {
            "name": "DocumentRetriever",
            "func": qa_chain.run,
            "description": "Summarize or answer queries based on loaded documents."
        }
    ]
    agent = initialize_agent(
        tools,
        llm=OpenAI(temperature=0, openai_api_key=API_KEY),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent