# LangGraph MongoDB Toolbox Agent


# MongoDB + LangGraph Agent Demo
This project demonstrates a LangGraph-based agent that discovers and uses tools stored in MongoDB. It includes a full setup script and an interactive CLI for chatting with the agent, selecting predefined test queries, and persisting conversations by thread.

**Key Components:**

**MongoDB Integration:** Stores tool definitions and conversation checkpoints, vector embeddings and operational data.
**VoyageAI Embeddings:** Used for vectorizing tool descriptions to enable semantic search for tool selection.
**LangGraph:** Provides the framework for building the agent's state machine and managing conversation flow.
**Dynamic Tool Selection:** The agent uses vector search to identify and utilize the most relevant tools for a given user query.
**Conversation Persistence: **Conversation history is stored in MongoDB using LangGraph's MongoDBSaver checkpointer.

**Workflow:**

* **Initialization:** Connects to MongoDB and initializes the MongoDBManager and ToolRegistry.
* **Tool Registration:** Registers available tools (functions) with the ToolRegistry, storing their definitions and embeddings in MongoDB.
* **Agent Creation:** When a user query is received, the LangGraphAgentBuilder performs a vector search on the tool descriptions in MongoDB to find the most relevant tools.
* **Graph Execution:** A LangGraph is dynamically constructed with the selected tools and the LLM. The user's query is processed through the graph.
* **Tool Calling:** If the LLM determines a tool is needed, the graph routes the execution to the tool node, which invokes the appropriate tool function.
* **Response Generation:** The agent incorporates the tool's output into its response to the user.
* **Conversation Persistence:**The entire conversation state is saved in MongoDB, allowing for continuation of the conversation.


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


