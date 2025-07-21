import os
import requests
import json
import sseclient
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- 1. Define Tools for SSE-based A2A Communication ---

@tool
def get_client_profile(client_id: str) -> dict:
    """Fetches the profile for a given client ID."""
    print(f"--- TOOL CALLED: get_client_profile for client {client_id} ---")
    if client_id == "C-12345":
        return {"name": "John Doe", "age": 45, "risk_tolerance": "moderate", "income": 150000}
    return {"error": "Client not found."}

@tool
def delegate_form_filling_task_sse(form_number: str, client_id: str, client_profile: dict) -> str:
    """
    Delegates a form filling task to the PDF Form Filler Agent via an SSE stream.
    This tool will connect to the stream and wait for the final completion event.
    """
    print(f"--- TOOL CALLED: delegate_form_filling_task_sse for form {form_number} ---")
    url = "http://localhost:8000/stream_task"
    payload = {
        "task": f"Fill form {form_number}",
        "client_id": client_id,
        "form_number": form_number,
        "client_profile": client_profile
    }
    headers = {'Accept': 'text/event-stream'}

    try:
        # The requests library can handle streaming responses
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
        
        # Use sseclient-py to parse the event stream
        client = sseclient.SSEClient(response)
        
        print("--- SSE: Connection opened. Waiting for events... ---")
        for event in client.events():
            print(f"--- SSE EVENT RECEIVED: event='{event.event}', data='{event.data}' ---")
            
            # We are only interested in the final event for this workflow
            if event.event == 'task_completed':
                print("--- SSE: Final event received. Task complete. ---")
                return f"Task completed successfully. Final result: {event.data}"

        return "Stream ended without a completion event."

    except requests.exceptions.RequestException as e:
        return f"Error connecting to SSE agent: {e}"
    except Exception as e:
        return f"An unexpected error occurred while processing the stream: {e}"

# --- 2. Define Agent State (with explicit fields) ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    client_profile: Optional[dict]

# --- 3. Define Graph Nodes ---

def agent_node(state: AgentState, llm):
    """The main reasoning brain of the agent."""
    print("--- NODE: agent_node ---")
    return {"messages": [llm.invoke(state["messages"])]}

def custom_tool_node(state: AgentState):
    """
    Custom tool node to handle state updates after tool execution.
    Specifically, it saves the client_profile to the state.
    """
    print("--- NODE: custom_tool_node ---")
    tool_node = ToolNode([get_client_profile, delegate_form_filling_task_sse])
    tool_result_state = tool_node.invoke(state)
    
    new_messages = tool_result_state['messages']
    
    # Update state with the client profile if it was fetched
    for msg in new_messages:
        if msg.name == "get_client_profile":
            try:
                profile_data = json.loads(msg.content.replace("'", "\""))
                state['client_profile'] = profile_data
            except (json.JSONDecodeError, AttributeError):
                pass
                
    return {"messages": new_messages, "client_profile": state.get('client_profile')}

def responsible_ai_filter_node(state: AgentState):
    """A placeholder for our RAI validation logic."""
    print("--- NODE: responsible_ai_filter_node ---")
    print("--- RAI FILTER: Validation Passed (mock) ---")
    return {}

# --- 4. Define Conditional Edges ---
from langgraph.prebuilt import tools_condition

# --- 5. Build and Compile the Graph ---

tools = [get_client_profile, delegate_form_filling_task_sse]
llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)
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
        # This prompt tests the full SSE flow. The agent should fetch the profile
        # and then delegate the task, waiting on the stream for the final result.
        run_agent(
            "First, get the profile for client C-12345. "
            "Then, using that profile, delegate the task to fill out money transfer form 1234."
        )
