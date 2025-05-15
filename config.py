import os

def get_api_key() -> str:
    """
    Retrieve OpenAI API key from environment or prompt the user.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        key = input("Enter your OpenAI API key: ").strip()
    if not key:
        raise ValueError("OpenAI API key is required.")
    return key