import os
import requests
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- 1. Define Tools ---
# The delegate_task tool now contains the real network call logic.
# It expects all necessary data to be passed in its arguments.

@tool
def get_client_profile(client_id: str) -> dict:
    """Fetches the profile for a given client ID."""
    print(f"--- TOOL CALLED: get_client_profile for client {client_id} ---")
    if client_id == "C-12345":
        return {"name": "John Doe", "age": 45, "risk_tolerance": "moderate", "income": 150000}
    return {"error": "Client not found."}

@tool
def get_portfolio(client_id: str) -> dict:
    """Fetches the investment portfolio for a given client ID."""
    print(f"--- TOOL CALLED: get_portfolio for client {client_id} ---")
    if client_id == "C-12345":
        return {"stocks": 150000, "bonds": 75000, "cash": 25000, "total": 250000}
    return {"error": "Portfolio not found."}

@tool
def delegate_task(target_agent: str, task: str, form_number: str, client_id: str, client_profile: dict) -> str:
    """
    Delegates a specific task to a specialized agent.
    This tool requires the client_profile to be passed in the arguments.
    """
    print(f"--- TOOL CALLED: delegate_task to {target_agent} ---")
    if target_agent == "PDF-Form-Filler-Agent":
        url = "http://localhost:8000/invoke"
        payload = {
            "task": task,
            "client_id": client_id,
            "form_number": form_number,
            "client_profile": client_profile
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return str(response.json())
        except requests.exceptions.RequestException as e:
            return f"Error delegating to PDF agent: {e}"
    return f"Error: Unknown agent '{target_agent}'."

# --- 2. Define Agent State (with explicit fields) ---
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
    # First, save the client_id if the LLM provides it
    # This is a simple example of how the agent can update state directly
    # based on its reasoning.
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for call in last_message.tool_calls:
            if call['name'] == 'get_client_profile' or call['name'] == 'get_portfolio':
                state['client_id'] = call['args']['client_id']

    return {"messages": [llm.invoke(state["messages"])]}

def argument_injector_node(state: AgentState):
    """
    This node inspects the pending tool calls and injects additional
    arguments from the agent's state where needed.
    """
    print("--- NODE: argument_injector_node ---")
    last_message = state["messages"][-1]
    
    # Ensure there are tool calls to process
    if not last_message.tool_calls:
        return {}

    for call in last_message.tool_calls:
        if call["name"] == "delegate_task":
            print("--- INJECTOR: Found delegate_task. Injecting client_profile. ---")
            if state.get("client_profile"):
                call["args"]["client_profile"] = state["client_profile"]
            else:
                # Handle case where profile isn't fetched yet
                print("--- INJECTOR WARNING: client_profile not found in state. ---")
                call["args"]["client_profile"] = {"error": "Profile not available in state."}
    
    return {"messages": [last_message]}


def state_updater_node(state: AgentState):
    """
    This node updates the main state with the results from simple tool calls
    before they are passed back to the LLM.
    """
    print("--- NODE: state_updater_node ---")
    last_message = state["messages"][-1] # This will be a ToolMessage from the ToolNode
    
    # This is a simplified example. In a real system, you'd parse the tool
    # output more robustly.
    if "get_client_profile" in last_message.name:
        # Assuming the content is a string representation of a dict
        import json
        try:
            profile_data = json.loads(last_message.content.replace("'", "\""))
            return {"client_profile": profile_data}
        except json.JSONDecodeError:
            return {} # Ignore if parsing fails
            
    return {}


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
        return "inject_arguments"
    else:
        return "validate_output"

# --- 5. Build and Compile the Graph ---

tools = [get_client_profile, get_portfolio, delegate_task]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

workflow = StateGraph(AgentState)

workflow.add_node("agent", lambda state: agent_node(state, llm_with_tools))
workflow.add_node("argument_injector", argument_injector_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("state_updater", state_updater_node)
workflow.add_node("responsible_ai_filter", responsible_ai_filter_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"inject_arguments": "argument_injector", "validate_output": "responsible_ai_filter"},
)

workflow.add_edge("argument_injector", "tools")
workflow.add_edge("tools", "state_updater")
workflow.add_edge("state_updater", "agent")
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
        run_agent("First, get the profile for client C-12345, then fill out money transfer form 1234 for them.")
