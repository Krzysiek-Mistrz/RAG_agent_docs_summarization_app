from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import OpenAI

def initialize_summarization_agent(qa_chain, api_key):
    """
    Wrap the RetrievalQA chain into an agent for interactive queries.
    
    Args:
        qa_chain: A RetrievalQA-based chain for document Q&A.
        api_key: OpenAI API key to instantiate the LLM.
        
    Returns:
        A LangChain agent that uses ZERO_SHOT_REACT_DESCRIPTION.
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
        llm=OpenAI(temperature=0, openai_api_key=api_key),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent
