# MongoDB + LangGraph Agent Demo

This project demonstrates how to use MongoDB as a comprehensive memory provider for LangGraph agents, enabling persistent conversations, semantic tool discovery, and intelligent document retrieval. The agent showcases MongoDB's role as both conversation memory and knowledge base.

## What This Demo Shows

**🧠 MongoDB as Memory Provider:** LangGraph's MongoDBSaver checkpointer stores conversation state, enabling seamless continuation of multi-turn dialogues across sessions with full context preservation.

**🔍 Semantic Memory Retrieval:** Vector search across conversation history and policy documents allows the agent to maintain context and retrieve relevant information from previous interactions.

**🛠️ Dynamic Tool Discovery:** Tools are stored in MongoDB with embeddings, enabling semantic matching between user queries and available capabilities without hardcoded routing.

**📚 Context-Aware Document Intelligence:** Policy documents are processed with contextualized embeddings, enabling sophisticated retrieval-augmented generation for customer service scenarios.

**💾 Unified Data Architecture:** MongoDB serves as the single source of truth for conversations, tools, documents, and operational data, demonstrating enterprise-ready memory management.

**Key Components:**

**MongoDB as Memory Provider:** Serves as the unified storage layer for conversation checkpoints, tool definitions, document embeddings, and operational data using LangGraph's MongoDBSaver.

**VoyageAI Context-Aware Embeddings:** Uses voyage-context-3 model for contextualized embeddings of both tools and documents, enabling sophisticated semantic search and retrieval-augmented generation.

**LangGraph State Management:** Provides the framework for building conversational state machines with persistent memory, enabling complex multi-turn interactions.

**Vector Search Architecture:** MongoDB Atlas vector search indexes enable semantic retrieval across conversations, tools, and documents for context-aware responses.

**Dynamic Memory Integration:** Conversation context is automatically incorporated into tool selection and document retrieval, creating a cohesive memory-aware experience.

## Prerequisites

- Python 3.10+
- A MongoDB deployment (Atlas recommended) and connection string
- API keys set as environment variables:

```bash
export MONGODB_URI="your_mongodb_connection_string"
export OPENAI_API_KEY="your_openai_api_key"
# Optional, for reranking and embeddings:
export VOYAGE_API_KEY="your_voyage_api_key"
```

Install dependencies using `uv` (recommended) or `pip`:

```bash
uv sync
# or
pip install -e .
```

## Quick Start

### 1) Run Full Setup

Creates tools, vector indexes, ingests policy documents, and inserts dummy orders.

```bash
python setup_agent_database.py --full
```

You can also run individual steps, e.g.:

```bash
python setup_agent_database.py --tools --indexes
python setup_agent_database.py --policies
python setup_agent_database.py --dummy-data
python setup_agent_database.py --status
```

### 2) Start the Interactive Agent CLI

```bash
python langgraph_mongodb_toolbox_agent.py
```

You will see a menu:

- 1) Type a custom query and thread ID
- 2) Choose from predefined test queries
- q) Quit

While chatting in a thread:

- Type `back` to return to the main menu
- Type `quit` (or press Ctrl+C/D) to exit

Conversations are persisted by thread ID using MongoDB-backed checkpoints.

## Project Structure

- `langgraph_mongodb_toolbox_agent.py` — Agent implementation, interactive CLI, tool registry integration, and LangGraph execution.
- `setup_agent_database.py` — One-shot CLI to set up tools, vector indexes, policy documents, and dummy orders.
- `reset_demo.py` — Optional helper to reset demo data (if present).

## Notes on Vector Search and Embeddings

- The project uses MongoDB vector search indexes (created via `setup_agent_database.py`).
- Tools and policy document chunks store `embedding` vectors. Ensure your MongoDB deployment supports vector search.
- VoyageAI is used for contextualized embeddings and optional reranking. Set `VOYAGE_API_KEY` for best results.

## Troubleshooting

- Connection issues: verify `MONGODB_URI` and network access.
- Missing variables: ensure `MONGODB_URI` and `OPENAI_API_KEY` are exported.
- Index creation delays: the setup script waits ~30 seconds after creating indexes.
- Agent initialization: the main agent script initializes the agent builder automatically when needed.

## License

This repository is for demo purposes. Adapt as needed for your environment.


