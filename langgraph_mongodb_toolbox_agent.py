## Install required libraries
# !uv pip install --quiet --upgrade pymongo langchain langchain-openai langgraph langgraph-checkpoint-mongodb voyageai requests python-dotenv PyPDF2 pytz

# Import required libraries
from pymongo import MongoClient
from langchain.agents import tool
from typing import List, Dict, Any, Callable, Optional
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.mongodb import MongoDBSaver
import voyageai
import inspect
import random
from datetime import datetime, timezone
import os
from functools import wraps
from typing import get_type_hints
from dataclasses import dataclass
from contextlib import contextmanager
import pytz

# Get API keys from environment variables or Colab secrets
try:
    from google.colab import userdata
    MONGODB_URI = userdata.get('MDB_DEMO_URI')
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    VOYAGE_API_KEY = userdata.get('VOYAGE_API_KEY')
except ImportError:
    # For local testing, use environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-openai-api-key')
    VOYAGE_API_KEY = os.environ.get('VOYAGE_API_KEY', 'your-voyage-api-key')

# Configuration constants
GPT_MODEL = "gpt-4o"
DB_NAME = "langgraph_agent_demo"
COLLECTION_NAMES = {
    "tools": "tools",
    "orders": "orders", 
    "returns": "returns",
    "policies": "policies"
}
VECTOR_INDEX_CONFIG = {
    "name": "vector_index",
    "dimensions": 1024,
    "similarity": "cosine"
}

@dataclass
class MongoDBConfig:
    """Configuration class for MongoDB connection and collections."""
    uri: str
    db_name: str
    collection_names: Dict[str, str]
    
    def __post_init__(self):
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.collections = {
            name: self.db[collection_name] 
            for name, collection_name in self.collection_names.items()
        }

class MongoDBManager:
    """Manager class for MongoDB operations."""
    
    def __init__(self, config: MongoDBConfig):
        self.config = config
        self._voyage_client = None
    
    @property
    def voyage_client(self):
        """Lazy initialization of VoyageAI client."""
        if self._voyage_client is None:
            self._voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        return self._voyage_client
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using VoyageAI."""
        text = text.replace("\n", " ")
        # Use contextualized query embedding to match contextualized doc vectors
        result = self.voyage_client.contextualized_embed(
            inputs=[[text]],
            model="voyage-context-3",
            input_type="query",
            output_dimension=VECTOR_INDEX_CONFIG["dimensions"],
        )
        res0 = result.results[0] if getattr(result, "results", None) else None
        if not res0 or not res0.embeddings:
            raise ValueError("Voyage contextualized_embed returned no embeddings for query")
        return res0.embeddings[0]
    
    def get_collection(self, name: str):
        """Get a collection by name."""
        return self.config.collections.get(name)
    
    @property
    def client(self):
        """Get the MongoDB client."""
        return self.config.client
    
    @contextmanager
    def get_connection(self):
        """Context manager for MongoDB connection."""
        try:
            yield self.config.client
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            raise

class ToolRegistry:
    """Registry for managing tool registration and discovery."""
    
    def __init__(self, mongo_manager: MongoDBManager):
        self.mongo_manager = mongo_manager
        self.decorated_tools_registry = {}
    
    def register_tool(self, func: Callable, vector_store_collection=None):
        """Register a tool in MongoDB and local registry."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Generate tool definition
        tool_def = self._generate_tool_definition(func)
        
        # Store in MongoDB if collection provided
        if vector_store_collection is not None:
            self._store_tool_in_mongodb(tool_def, vector_store_collection)
        
        # Register in local registry
        self.decorated_tools_registry[func.__name__] = func
        
        return wrapper
    
    def _generate_tool_definition(self, func: Callable) -> Dict[str, Any]:
        """Generate tool definition from function."""
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        type_hints = get_type_hints(func)
        
        tool_def = {
            "name": func.__name__,
            "description": docstring.strip(),
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
        
        for param_name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            
            param_type = type_hints.get(param_name, type(None))
            json_type = self._get_json_type(param_type)
            
            tool_def["parameters"]["properties"][param_name] = {
                "type": json_type,
                "description": f"Parameter {param_name}",
            }
            
            if param.default == inspect.Parameter.empty:
                tool_def["parameters"]["required"].append(param_name)
        
        tool_def["parameters"]["additionalProperties"] = False
        return tool_def
    
    def _get_json_type(self, param_type) -> str:
        """Get JSON type from Python type."""
        if param_type in (int, float):
            return "number"
        elif param_type is bool:
            return "boolean"
        return "string"
    
    def _store_tool_in_mongodb(self, tool_def: Dict[str, Any], collection):
        """Store tool definition in MongoDB."""
        existing_doc = collection.find_one({"metadata.name": tool_def["name"]})
        if not existing_doc:
            embedding = self.mongo_manager.generate_embedding(tool_def["description"])
            document = {
                "text": tool_def["description"],
                "metadata": tool_def,
                "embedding": embedding
            }
            collection.insert_one(document)
            print(f"✅ Registered new tool '{tool_def['name']}' in MongoDB.")
        # Silently skip if tool already exists
    
    def vector_search_tools(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform vector search to find relevant tools."""
        query_embedding = self.mongo_manager.generate_embedding(user_query)
        tools_collection = self.mongo_manager.get_collection("tools")
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_CONFIG["name"],
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 150,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        
        results = list(tools_collection.aggregate(pipeline))
        return [result["metadata"] for result in results if "metadata" in result]
    
    def populate_tools(self, search_results: List[Dict[str, Any]]) -> List[Callable]:
        """Populate tools from search results."""
        tools = []
        for result in search_results:
            tool_name = result["name"]
            function_obj = self.decorated_tools_registry.get(tool_name)
            if function_obj:
                tools.append(function_obj)
        return tools

# Define the graph state type
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

class LangGraphAgentBuilder:
    """Builder class for creating LangGraph agents."""
    
    def __init__(self, mongo_manager: MongoDBManager, tool_registry: ToolRegistry):
        self.mongo_manager = mongo_manager
        self.tool_registry = tool_registry
    
    def create_langgraph_tools(self, tool_functions: List[Callable]) -> List[Any]:
        """Convert function objects to LangGraph tools."""
        from langchain.tools import StructuredTool
        
        langgraph_tools = []
        for func in tool_functions:
            tool_func = StructuredTool.from_function(
                func=func,
                name=func.__name__,
                description=func.__doc__ or f"Tool: {func.__name__}"
            )
            langgraph_tools.append(tool_func)
        return langgraph_tools
    
    def create_dynamic_agent(self, user_query: str, thread_id: str = "default"):
        """Create a dynamic LangGraph agent based on user query."""
        # Create a temporary app to check for existing context
        temp_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model=GPT_MODEL)
        temp_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        temp_app = self._init_graph(temp_prompt | temp_llm, {})
        
        # Get conversation context from existing checkpoint
        conversation_context = self._get_context_from_checkpoint(temp_app, thread_id)
        
        # Create enhanced query that includes conversation context
        enhanced_query = self._create_enhanced_query(user_query, conversation_context)
        
        # Debug: Show what context and enhanced query we're using
        if conversation_context:
            print(f"📚 Context found: {len(conversation_context)} messages")
            print(f"🔍 Enhanced query: {enhanced_query[:150]}...")
        else:
            print(f"📚 No context found for thread {thread_id}")
            print(f"🔍 Using original query: {user_query}")
        
        # Retrieve relevant tools using enhanced query
        tools_metadata = self.tool_registry.vector_search_tools(enhanced_query, top_k=3)
        tool_functions = self.tool_registry.populate_tools(tools_metadata)
        
        # Print selected tools and context info
        self._print_selected_tools(tool_functions)
        if conversation_context:
            print(f"📚 Using conversation context: {len(conversation_context)} previous messages")
        
        # Create LangGraph tools
        langgraph_tools = self.create_langgraph_tools(tool_functions)
        
        # Initialize LLM and create graph
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model=GPT_MODEL)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with access to various tools. Use the available tools to answer the user's question. Think step-by-step and use the most appropriate tools for the task."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Bind tools and create graph
        bind_tools = llm.bind_tools(langgraph_tools)
        llm_with_tools = prompt | bind_tools
        tools_by_name = {tool.name: tool for tool in langgraph_tools}
        
        # Initialize and execute graph
        app = self._init_graph(llm_with_tools, tools_by_name)
        self._execute_graph(app, thread_id, user_query)
    
    def _get_context_from_checkpoint(self, app, thread_id: str) -> List[str]:
        """Get conversation context from an existing checkpoint."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = app.get_state(config)
            
            if checkpoint and hasattr(checkpoint, 'values') and checkpoint.values:
                messages = checkpoint.values.get("messages", [])
                if messages:
                    # Extract relevant context from previous messages
                    context_messages = []
                    for message in messages[-4:]:  # Last 4 messages for context
                        if hasattr(message, 'content') and message.content:
                            # Clean and truncate message content
                            content = str(message.content).strip()
                            if len(content) > 200:
                                content = content[:200] + "..."
                            context_messages.append(content)
                    return context_messages
        except Exception as e:
            # If we can't get context, that's okay - we'll proceed without it
            pass
        return []
    
    def _create_enhanced_query(self, user_query: str, conversation_context: List[str]) -> str:
        """Create an enhanced query that includes conversation context for better tool selection."""
        if not conversation_context:
            return user_query
        
        # Combine current query with relevant context
        context_summary = " ".join(conversation_context[-2:])  # Use last 2 context messages
        enhanced_query = f"{user_query} [Context: {context_summary}]"
        
        # Truncate if too long to avoid embedding issues
        if len(enhanced_query) > 1000:
            enhanced_query = enhanced_query[:1000] + "..."
        
        return enhanced_query
    
    def _print_selected_tools(self, tools: List[Callable]) -> None:
        """Print selected tools."""
        tool_names = [tool.__name__ for tool in tools]
        print("Selected tools:")
        import pprint
        pprint.pprint(tool_names)
        print("---------")
    
    def _print_agent_step(self, step_count: int, value: Any) -> None:
        """Print agent step in a readable format."""
        print(f"\n📝 Step {step_count}: Agent Thinking")
        print("-" * 30)
        
        if value and value.get("messages"):
            last_message = value["messages"][-1]
            
            # Check if this is a tool call
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                print("🤔 Agent decided to use tools:")
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call.get("name", "Unknown")
                    args = tool_call.get("args", {})
                    print(f"   🔧 {tool_name}({args})")
            else:
                # Regular response
                content = getattr(last_message, 'content', 'No content')
                if content and content.strip():
                    # Truncate long content for readability
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"💭 {content}")
                else:
                    print("💭 Agent is thinking...")
        else:
            print("💭 Agent is processing...")
    
    def _print_tool_step(self, step_count: int, value: Any) -> None:
        """Print tool execution step in a readable format."""
        print(f"\n⚙️  Step {step_count}: Tool Execution")
        print("-" * 30)
        
        if value and value.get("messages"):
            for message in value["messages"]:
                if hasattr(message, 'content'):
                    content = message.content
                    # Truncate very long tool outputs
                    if len(content) > 300:
                        content = content[:300] + "..."
                    
                    # Format the output nicely
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if i == 0:
                            print(f"📊 Result: {line}")
                        else:
                            print(f"      {line}")
                else:
                    print("📊 Tool executed successfully")
        else:
            print("📊 Tools executed")
    
    def _init_graph(self, llm_with_tools, tools_by_name: Dict[str, Callable]):
        """Initialize the LangGraph."""
        graph = StateGraph(GraphState)
        
        graph.add_node("agent", lambda state: self._agent_node(state, llm_with_tools))
        graph.add_node("tools", lambda state: self._tool_node(state, tools_by_name))
        
        graph.add_edge(START, "agent")
        graph.add_edge("tools", "agent")
        graph.add_conditional_edges("agent", self._route_tools, {"tools": "tools", END: END})
        
        checkpointer = MongoDBSaver(self.mongo_manager.config.client)
        return graph.compile(checkpointer=checkpointer)
    
    def _execute_graph(self, app, thread_id: str, user_input: str) -> None:
        """Execute the graph with conversation persistence."""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Check if this is a new conversation or continuing an existing one
        try:
            # Try to get existing checkpoint
            checkpoint = app.get_state(config)
            if checkpoint and checkpoint.get("messages"):
                print(f"📝 Continuing conversation in thread '{thread_id}' ({len(checkpoint['messages'])} previous messages)")
            else:
                print(f"🆕 Starting new conversation in thread '{thread_id}'")
        except Exception as e:
            # No existing checkpoint, start new conversation
            print(f"🆕 Starting new conversation in thread '{thread_id}'")
        
        # Always start with the new user message - LangGraph will merge with existing state
        input_data = {"messages": [("user", user_input)]}
        
        # Execute the graph with conversation persistence
        final_value = None
        step_count = 0
        
        print("\n🤖 Agent Execution Steps:")
        print("=" * 50)
        
        for output in app.stream(input_data, config):
            for key, value in output.items():
                step_count += 1
                final_value = value
                
                if key == "agent":
                    self._print_agent_step(step_count, value)
                elif key == "tools":
                    self._print_tool_step(step_count, value)
        
        print("=" * 50)
        print("🎯 Final Answer:")
        print("-" * 30)
        if final_value and final_value.get("messages"):
            last_message = final_value["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content:
                print(last_message.content)
            else:
                print("No content in final message")
        else:
            print("No final answer generated.")
        print("-" * 30)
    
    def _agent_node(self, state: GraphState, llm_with_tools) -> GraphState:
        """Agent node that processes messages."""
        messages = state["messages"]
        result = llm_with_tools.invoke(messages)
        return {"messages": [result]}
    

    
    def _tool_node(self, state: GraphState, tools_by_name: Dict[str, Callable]) -> GraphState:
        """Tool node that executes tool calls."""
        result = []
        tool_calls = state["messages"][-1].tool_calls
        
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        
        return {"messages": result}
    
    def _route_tools(self, state: GraphState):
        """Route to tool node if needed."""
        messages = state.get("messages", [])
        
        if len(messages) > 0:
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state: {state}")
        
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        
        return END

# Initialize global instances
mongo_config = MongoDBConfig(MONGODB_URI, DB_NAME, COLLECTION_NAMES)
mongo_manager = MongoDBManager(mongo_config)
tool_registry = ToolRegistry(mongo_manager)

# Tool definitions with improved organization
def get_current_time(timezone: str = "Australia/Melbourne") -> str:
    """Get the current time for a specified timezone. Defaults to Melbourne, Australia timezone. Returns both time and date in the specified timezone."""
    from datetime import timezone as dt_timezone
    import pytz
    
    # Get current UTC time
    utc_now = datetime.now(dt_timezone.utc)
    
    # Convert to specified timezone (defaults to Melbourne)
    try:
        tz = pytz.timezone(timezone)
        local_time = utc_now.astimezone(tz)
        current_time = local_time.strftime("%H:%M:%S")
        date_str = local_time.strftime("%Y-%m-%d")
        return f"The current time in {timezone} is {current_time} on {date_str}."
    except pytz.exceptions.UnknownTimeZoneError:
        # Fallback to UTC if timezone is invalid
        current_time = utc_now.strftime("%H:%M:%S")
        return f"Invalid timezone '{timezone}'. The current time in UTC is {current_time}."

def lookup_order_number(order_id: int) -> str:
    """Lookup the details of a specific order number using its order ID."""
    orders_collection = mongo_manager.get_collection("orders")
    order = orders_collection.find_one({"order_id": order_id})
    
    if order:
        order_details = [
            f"Order ID: {order.get('order_id')}",
            f"Order Date: {order.get('order_date').strftime('%Y-%m-%d %H:%M:%S')}",
            f"Status: {order.get('status')}",
            f"Total Amount: ${order.get('total_amount')}",
            f"Shipping Address: {order.get('shipping_address')}",
            f"Payment Method: {order.get('payment_method')}",
            "Items:"
        ]
        
        for item in order.get("items", []):
            order_details.append(f"- {item.get('name')}: ${item.get('price')}")
        
        return "\n".join(order_details)
    else:
        return f"Order with ID {order_id} not found."

def search_return_policy_documents(customer_query: str) -> str:
    """Search and retrieve relevant return policy information from stored policy documents using vector search and VoyageAI rerank-2.5."""
    
    # Generate embedding for the customer query
    query_embedding = mongo_manager.generate_embedding(customer_query)
    policies_collection = mongo_manager.get_collection("policies")
    
    # Perform vector search to find candidate policy documents
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_CONFIG["name"],
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 200,
                "limit": 10,
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    
    candidates = list(policies_collection.aggregate(pipeline))
    
    if not candidates:
        return "No relevant return policy information found. Please contact customer service for assistance."
    
    # Prepare documents for reranking
    documents = [candidate["text"] for candidate in candidates]
    
    # Perform reranking using VoyageAI rerank-2.5
    rerank_results = mongo_manager.voyage_client.rerank(
        query=customer_query,
        documents=documents,
        model="rerank-2.5",
        top_k=min(5, len(documents))  # Return top 5 reranked results
    )
    
    # Map reranked results back to original candidates with metadata
    reranked_candidates = []
    for rerank_result in rerank_results.results:
        # Find the original candidate by matching text
        for candidate in candidates:
            if candidate["text"] == rerank_result.document:
                reranked_candidates.append({
                    **candidate,
                    "rerank_score": rerank_result.relevance_score
                })
                break
    
    # Use reranked results
    results = reranked_candidates
    search_method = "vector search + VoyageAI rerank-2.5"
    
    # Format the results
    response_parts = [f"📋 Return Policy Information ({search_method}):"]
    response_parts.append("=" * 50)
    
    for i, result in enumerate(results, 1):
        score = result.get("rerank_score", result.get("score", 0))
        text = result.get("text", "No text available")
        metadata = result.get("metadata", {})

        file_name = metadata.get("file_name")
        chunk_index = metadata.get("chunk_index")

        response_parts.append(f"\n{i}. Relevance Score: {score:.3f}")
        response_parts.append(f"   Content: {text[:300]}{'...' if len(text) > 300 else ''}")
        response_parts.append(f"   File: {file_name}")
        response_parts.append(f"   Chunk Index: {chunk_index}")

        
        response_parts.append("-" * 30)
    
    response_parts.append(f"\n💡 Found {len(results)} relevant policy document(s) for your query: '{customer_query}'")
    
    return "\n".join(response_parts)

def create_return_request(order_id: int, reason: str) -> str:
    """Create a return request entry for a given order ID."""
    orders_collection = mongo_manager.get_collection("orders")
    returns_collection = mongo_manager.get_collection("returns")
    
    order = orders_collection.find_one({"order_id": order_id})
    
    if order:
        return_data = {
            "return_id": returns_collection.count_documents({}) + 1,
            "order_id": order_id,
            "return_date": datetime.now(pytz.timezone("Australia/Melbourne")),
            "reason": reason,
            "status": "Pending",
            "items": order.get("items", []),
        }
        
        returns_collection.insert_one(return_data)
        return f"Return request created successfully for order {order_id} with reason: {reason}."
    else:
        return f"Order with ID {order_id} not found. Could not create return."

# Additional tools (simplified for brevity)
def greet_user(name: str) -> str:
    """Greet the user by name."""
    return f"Hello, {name}! Nice to meet you."

def calculate_square_root(number: float) -> str:
    """Calculate the square root of a given number."""
    if number < 0:
        return "Cannot calculate the square root of a negative number."
    return f"The square root of {number} is {number**0.5}."

def repeat_phrase(phrase: str, times: int = 1) -> str:
    """Repeat a given phrase a specified number of times."""
    if times <= 0:
        return "Please specify a positive number of times to repeat."
    return (phrase + " ") * times

def roll_dice(number_of_dice: int = 1, sides: int = 6) -> str:
    """Roll dice and return results."""
    if number_of_dice <= 0 or sides <= 0:
        return "Please specify a positive number of dice and sides."
    rolls = [random.randint(1, sides) for _ in range(number_of_dice)]
    total = sum(rolls)
    return f"You rolled {number_of_dice} dice with {sides} sides each. Results: {rolls}. Total: {total}."

def flip_coin(number_of_flips: int = 1) -> str:
    """Flip a coin and return results."""
    if number_of_flips <= 0:
        return "Please specify a positive number of flips."
    results = [random.choice(["Heads", "Tails"]) for _ in range(number_of_flips)]
    return f"You flipped the coin {number_of_flips} times. Results: {results}."

def generate_random_password(length: int = 12) -> str:
    """Generate a random password."""
    if length <= 0:
        return "Please specify a positive password length."
    import string
    characters = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.choice(characters) for i in range(length))
    return f"Here is a random password: {password}"

def register_all_tools():
    """Register all tools in MongoDB."""
    tools_collection = mongo_manager.get_collection("tools")
    
    # List of all tool functions
    tool_functions = [
        get_current_time,
        lookup_order_number,
        search_return_policy_documents,
        create_return_request,
        greet_user,
        calculate_square_root,
        repeat_phrase,
        roll_dice,
        flip_coin,
        generate_random_password
    ]
    
    # Register each tool
    for tool_func in tool_functions:
        tool_registry.register_tool(tool_func, tools_collection)
    
    # Also populate the local registry for the populate_tools function
    populate_local_tool_registry()

def populate_local_tool_registry():
    """Populate the local tool registry without re-registering in MongoDB."""
    tool_functions = [
        get_current_time,
        lookup_order_number,
        search_return_policy_documents,
        create_return_request,
        greet_user,
        calculate_square_root,
        repeat_phrase,
        roll_dice,
        flip_coin,
        generate_random_password
    ]
    
    for tool_func in tool_functions:
        tool_registry.decorated_tools_registry[tool_func.__name__] = tool_func

# Initialize agent builder (will be re-initialized in main if needed)
agent_builder = None

def main():
    """Main function to demonstrate the LangGraph MongoDB toolbox agent."""
    
    # Check if tools are registered, if not, register them
    tools_collection = mongo_manager.get_collection("tools")
    if tools_collection.count_documents({}) == 0:
        print("No tools found in MongoDB. Registering tools...")
        register_all_tools()
        print("Tools registered successfully!")
    else:
        print(f"Found {tools_collection.count_documents({})} tools in MongoDB. Loading local registry...")
        # Only populate the local registry, don't re-register tools
        populate_local_tool_registry()
    
    # Initialize agent builder after tools are registered
    global agent_builder
    agent_builder = LangGraphAgentBuilder(mongo_manager, tool_registry)
    
    print("Running example queries to demonstrate the agent...")
    print("="*60)
    
    # Example usage - demonstrating conversation persistence and vector search
    print("🧪 Testing conversation persistence and vector search capabilities...")
    test_queries = [
        ("I would like to return the laptop that I recently purchased ar part of order 101. I have had it for 20 days since delivery. Please organise a return of just the laptop and tell me what my refund will be. thanks.", "thread_1"),
        # ("What is the return policy for electronics? I need to know the time limit.", "customer_conversation"),
        # ("I just purchased a laptop on order 101 today. How long do I have before I am unable to return it?", "customer_conversation"),
        # ("It is unused. I would like to send the laptop back. Can you please create refund?", "customer_conversation"),
    ]
    
    for i, (query, thread_id) in enumerate(test_queries, 1):
        print(f"\n{i}. Testing: {query}")
        print("-" * 40)
        try:
            agent_builder.create_dynamic_agent(query, thread_id)
        except Exception as e:
            print(f"Error running query: {e}")
        print("-" * 40)
    
    print("\n" + "="*60)
    print("Demo complete! Use the setup script to initialize the system:")
    print("  python setup_agent_database.py --help")

if __name__ == "__main__":
    main()
