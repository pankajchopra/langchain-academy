import os
import requests
import time
from typing import TypedDict, Annotated, List, Optional, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- 1. Define Tools for Asynchronous A2A Communication ---

@tool
def get_client_profile(client_id: str) -> dict:
    """Fetches the profile for a given client ID."""
    print(f"--- TOOL CALLED: get_client_profile for client {client_id} ---")
    if client_id == "C-12345":
        return {"name": "John Doe", "age": 45, "risk_tolerance": "moderate", "income": 150000}
    return {"error": "Client not found."}

@tool
def submit_form_filling_task(form_number: str, client_id: str, client_profile: dict) -> dict:
    """
    Submits a form filling task to the PDF Form Filler Agent.
    This starts the task and returns a task_id for status checking.
    """
    print(f"--- TOOL CALLED: submit_form_filling_task for form {form_number} ---")
    url = "http://localhost:8000/submit_task"
    payload = {
        "task": f"Fill form {form_number}",
        "client_id": client_id,
        "form_number": form_number,
        "client_profile": client_profile
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json() # Should return {"status": "ACCEPTED", "task_id": "..."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to submit task: {e}"}

@tool
def check_task_status(task_id: str) -> dict:
    """
    Checks the status of a previously submitted asynchronous task.
    """
    print(f"--- TOOL CALLED: check_task_status for task {task_id} ---")
    url = f"http://localhost:8000/task_status/{task_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json() # Returns {"task_id": "...", "status": "...", "result": "..."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to check status: {e}"}


# --- 2. Define Agent State (with task tracking) ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    client_profile: Optional[dict]
    # We will store the ID of the pending task here
    pending_task_id: Optional[str]

# --- 3. Define Graph Nodes ---
# We need a custom node to manage state updates from our tools

def agent_node(state: AgentState, llm):
    """The main reasoning brain of the agent."""
    print("--- NODE: agent_node ---")
    return {"messages": [llm.invoke(state["messages"])]}

def custom_tool_node(state: AgentState):
    """
    Custom tool node to handle state updates after tool execution.
    """
    print("--- NODE: custom_tool_node ---")
    # We use the standard ToolNode to execute the tools first
    tool_node = ToolNode([get_client_profile, submit_form_filling_task, check_task_status])
    tool_result_state = tool_node.invoke(state)
    
    # Now, we process the results to update our state
    new_messages = tool_result_state['messages']
    
    for msg in new_messages:
        if msg.name == "get_client_profile":
            import json
            try:
                # The output from a tool is a string, so we parse it
                profile_data = json.loads(msg.content.replace("'", "\""))
                state['client_profile'] = profile_data
            except json.JSONDecodeError:
                pass # Ignore parsing errors
        elif msg.name == "submit_form_filling_task":
            import json
            try:
                task_data = json.loads(msg.content.replace("'", "\""))
                if task_data.get("task_id"):
                    state['pending_task_id'] = task_data['task_id']
            except json.JSONDecodeError:
                pass
                
    return {"messages": new_messages, "client_profile": state.get('client_profile'), "pending_task_id": state.get('pending_task_id')}


def responsible_ai_filter_node(state: AgentState):
    """A placeholder for our RAI validation logic."""
    print("--- NODE: responsible_ai_filter_node ---")
    print("--- RAI FILTER: Validation Passed (mock) ---")
    return {}

# --- 4. Define Conditional Edges ---
from langgraph.prebuilt import tools_condition

# --- 5. Build and Compile the Graph ---

tools = [get_client_profile, submit_form_filling_task, check_task_status]
llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0) # Using a specific model version for consistency
llm_with_tools = llm.bind_tools(tools)

workflow = StateGraph(AgentState)

workflow.add_node("agent", lambda state: agent_node(state, llm_with_tools))
workflow.add_node("tools", custom_tool_node)
workflow.add_node("responsible_ai_filter", responsible_ai_filter_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "responsible_ai_filter"})
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
        # This prompt is designed to test the full asynchronous flow.
        # The agent should first get the profile, then submit the task,
        # and then loop, checking the status until it's complete.
        run_agent(
            "First, get the profile for client C-12345. "
            "Then, using that profile, submit a task to fill out money transfer form 1234. "
            "After submitting, check the status of the task until it is complete and tell me the final result."
        )
