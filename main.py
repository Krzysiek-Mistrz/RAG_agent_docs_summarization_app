from src.vectorstore import get_vectorstore
from src.qa_chain import build_qa_chain
from src.agent import initialize_summarization_agent

# Get API key from user
api_key = input("Wprowadź swój klucz OpenAI API: ")

def main():
    """
    Entry point: builds vectorstore, QA chain, initializes agent, and loops for user input.
    """
    # Load or build vectorstore
    vectorstore = get_vectorstore(api_key, "faiss_index")
    
    # Build QA summarization chain
    qa_chain = build_qa_chain(vectorstore, api_key)
    
    # Initialize agent
    agent = initialize_summarization_agent(qa_chain, api_key)

    # Example interaction
    print("Agent ready! Ask me to summarize or answer questions about your documents.")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        result = agent.run(query)
        print(f"Agent: {result}\n")

if __name__ == "__main__":
    main()
