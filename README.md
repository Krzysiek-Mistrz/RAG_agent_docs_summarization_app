# RAG Agent Docs Summarization App

A Retrieval-Augmented Generation (RAG) application that loads PDF documents, builds a FAISS vector index of embeddings, and provides an interactive agent for summarization and Q&A.

## Features

- Load and split PDF documents into chunks
- Embed chunks using OpenAI embeddings and store in FAISS
- RetrievalQA chain using map-reduce
- Interactive chat agent powered by LangChain
- Modular code structure

## Requirements

- Python 3.8+
- Poetry or pip
- OpenAI API key

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-org/rag-summarization-app.git
   cd rag-summarization-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or with Poetry:
   ```bash
   poetry install
   ```

3. Set your API key:
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

## Project Structure

```
.
├── README.md
├── config.py            # API key loader
├── main.py              # Application entrypoint
├── docs/                # Place your PDF files here
└── src/         # Generated FAISS index directory
     ├── __init__.py
     ├── docs_loader.py       # Document loading & splitting
     ├── vectorstore.py       # FAISS index creation/loading
     ├── qa_chain.py          # RetrievalQA chain builder
     └── agent.py             # LangChain agent initializer
```

## Usage

Run the app:
```bash
python main.py
```

Type questions or “summarize X” and the agent will respond based on your PDF docs.  
Type `exit` or `quit` to end the session.

## License

MIT