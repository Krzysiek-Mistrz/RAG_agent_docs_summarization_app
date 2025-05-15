from src.vectorstore import get_vectorstore
from src.qa_chain import build_qa_chain
from src.agent import initialize_summarization_agent

api_key = input("Enter Ur OpenAI API: ")

def main():
    """
    Entry point: builds vectorstore, QA chain, initializes agent, and loops for user input.
    """
    vectorstore = get_vectorstore(api_key, "faiss_index")
    qa_chain = build_qa_chain(vectorstore, api_key)
    agent = initialize_summarization_agent(qa_chain, api_key)
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
