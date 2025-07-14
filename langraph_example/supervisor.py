import os
import requests
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- 1. Define Tools ---
# The delegate_task tool is now a real network call.

@tool
def get_client_profile(client_id: str) -> dict:
    """
    Fetches the profile for a given client ID.
    """
    print(f"--- TOOL CALLED: get_client_profile for client {client_id} ---")
    if client_id == "C-12345":
        return {"name": "John Doe", "age": 45, "risk_tolerance": "moderate", "income": 150000}
    return {"error": "Client not found."}

@tool
def get_portfolio(client_id: str) -> dict:
    """
    Fetches the investment portfolio for a given client ID.
    """
    print(f"--- TOOL CALLED: get_portfolio for client {client_id} ---")
    if client_id == "C-12345":
        return {"stocks": 150000, "bonds": 75000, "cash": 25000, "total": 250000}
    return {"error": "Portfolio not found."}

# This tool is a placeholder for the LLM. The actual logic is in our custom tool_node.
@tool
def delegate_task(target_agent: str, task: str, form_number: str, client_id: str) -> str:
    """
    Delegates a specific task to a specialized agent.
    
    Args:
        target_agent: The name of the agent to delegate to (e.g., 'PDF-Form-Filler-Agent').
        task: A description of the task to be performed.
        form_number: The form number if applicable.
        client_id: The client ID for the task.
        
    Returns:
        A confirmation or result from the delegated agent.
    """
    # The actual implementation is in the custom tool_node, which has state access.
    # This definition is just for the LLM's reference.
    return "Delegation logic is handled by the graph's tool node."

# --- 2. Define Agent State (with explicit fields) ---
# We now have dedicated fields for important data, making it easier to access.

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    client_id: Optional[str]
    client_profile: Optional[dict]
    portfolio: Optional[dict]
    client_context: Optional[dict]

# --- 3. Define Graph Nodes ---

def agent_node(state: AgentState, llm):
    """The main reasoning brain of the agent."""
    print("--- NODE: agent_node ---")
    return {"messages": [llm.invoke(state["messages"])]}

def custom_tool_node(state: AgentState):
    """
    Custom tool node that can access the full state.
    This is necessary for the delegation tool to access client_profile.
    """
    print("--- NODE: custom_tool_node ---")
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    
    for call in tool_calls:
        tool_name = call["name"]
        args = call["args"]
        print(f"--- EXECUTING TOOL: {tool_name} with args {args} ---")
        
        if tool_name == "get_client_profile":
            output = get_client_profile.invoke(args)
            # Update the state with the fetched profile
            state['client_profile'] = output
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        
        elif tool_name == "get_portfolio":
            output = get_portfolio.invoke(args)
            # Update the state with the fetched portfolio
            state['portfolio'] = output
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

        elif tool_name == "delegate_task":
            # A2A Delegation Logic
            target_agent = args.get("target_agent")
            if target_agent == "PDF-Form-Filler-Agent":
                # The endpoint for our specialized agent
                url = "http://localhost:8000/invoke"
                payload = {
                    "task": args.get("task"),
                    "client_id": args.get("client_id"),
                    "form_number": args.get("form_number"),
                    "client_profile": state.get("client_profile") # Accessing state!
                }
                try:
                    response = requests.post(url, json=payload)
                    response.raise_for_status() # Raise an exception for bad status codes
                    output = response.json()
                except requests.exceptions.RequestException as e:
                    output = f"Error delegating to PDF agent: {e}"
                
                tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

    return {"messages": tool_messages, "client_profile": state.get('client_profile'), "portfolio": state.get('portfolio')}


def responsible_ai_filter_node(state: AgentState):
    """A placeholder for our RAI validation logic."""
    print("--- NODE: responsible_ai_filter_node ---")
    print("--- RAI FILTER: Validation Passed (mock) ---")
    return {}

# --- 4. Define Conditional Edges ---

def should_continue(state: AgentState) -> str:
    """Determines the next step after the main agent node has run."""
    print("--- EDGE: should_continue ---")
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "use_tools"
    else:
        return "validate_output"

# --- 5. Build and Compile the Graph ---

tools = [get_client_profile, get_portfolio, delegate_task]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

workflow = StateGraph(AgentState)

workflow.add_node("agent", lambda state: agent_node(state, llm_with_tools))
workflow.add_node("tools", custom_tool_node) # <-- Use our custom node
workflow.add_node("responsible_ai_filter", responsible_ai_filter_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"use_tools": "tools", "validate_output": "responsible_ai_filter"},
)

workflow.add_edge("tools", "agent")
workflow.add_edge("responsible_ai_filter", END)

app = workflow.compile()

# --- 6. Run the Agent ---

def run_agent(query: str):
    """Function to interact with our compiled agent graph."""
    inputs = {"messages": [HumanMessage(content=query)]}
    print(f"\n--- Running Agent for query: '{query}' ---\n")
    for event in app.stream(inputs, stream_mode="values"):
        last_message = event["messages"][-1]
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            print("--- AGENT RESPONSE ---")
            print(last_message.content)
            print("----------------------\n")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
    else:
        # Make sure to install requests: pip install requests
        # And run the pdf_filler_agent_v1.py in a separate terminal first.
        
        # This query will now make a real A2A call to the other agent.
        run_agent("First, get the profile for client C-12345, then fill out money transfer form 1234 for them.")
