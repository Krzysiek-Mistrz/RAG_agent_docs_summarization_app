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

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.py            # API key loader
├── main.py              # Application entrypoint
├── docs/                # Place your PDF files here
└── src/         # Generated FAISS index directory
     ├── __init__.py
     ├── docs_loader.py
     ├── vectorstore.py 
     ├── qa_chain.py
     └── agent.py
```

## Usage

Run the app:
```bash
python main.py
```

Type questions or “summarize X” and the agent will respond based on your PDF docs.  
Type `exit` or `quit` to end the session.

## License

GNU GPL V3 @ Krzychu 2025