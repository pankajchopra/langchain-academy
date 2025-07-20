import uvicorn
import uuid
import threading
import asyncio
import json
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from typing import TypedDict, Annotated, List, Dict
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

# --- This is a self-contained Specialized Agent ---
# It runs as its own web server and streams task updates via SSE.

# --- In-Memory Task Storage ---
# In a production system, this would be a more robust, persistent store like Redis.
task_storage: Dict[str, Dict] = {}

# --- 1. Define the Agent's Specialized Tools ---

@tool
def fetch_pdf_template(form_number: str) -> dict:
    """Fetches the path to a blank PDF form template."""
    print(f"--- PDF AGENT TOOL: Fetching template for form {form_number} ---")
    if form_number == "1234":
        return {"pdf_path": "/templates/money_transfer_1234.pdf"}
    return {"error": "PDF template not found."}

@tool
def fill_and_generate_link(client_data: dict, pdf_path: str) -> dict:
    """Fills a PDF with client data and generates a secure download link."""
    print(f"--- PDF AGENT TOOL: Filling form {pdf_path} for {client_data.get('name')} ---")
    import time
    time.sleep(5) # Simulate a long-running PDF generation task
    download_link = f"https://secure-downloads.example.com/filled_forms/{uuid.uuid4()}.pdf"
    return {"download_link": download_link}

# --- 2. Define the Agent's State ---

class FormFillerState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    client_data: dict
    form_number: str

# --- 3. Build the Agent's Graph ---

def create_pdf_filler_graph():
    """Creates the LangGraph agent for the PDF filler."""
    tools = [fetch_pdf_template, fill_and_generate_link]
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: FormFillerState):
        print("--- PDF AGENT NODE: agent_node ---")
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    tool_node = ToolNode(tools)
    
    workflow = StateGraph(FormFillerState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

pdf_filler_app = create_pdf_filler_graph()

# --- 4. Define the Background Task Execution Logic ---

def run_agent_in_background(task_id: str, initial_state: dict):
    """This function runs the agent's graph in a separate thread."""
    try:
        print(f"--- PDF AGENT: Starting background task {task_id} ---")
        task_storage[task_id]['status'] = 'IN_PROGRESS'
        
        final_state = pdf_filler_app.invoke(initial_state)
        
        result = final_state["messages"][-1].content
        task_storage[task_id]['status'] = 'COMPLETED'
        task_storage[task_id]['result'] = result
        print(f"--- PDF AGENT: Background task {task_id} completed ---")

    except Exception as e:
        print(f"--- PDF AGENT ERROR: Task {task_id} failed: {e} ---")
        task_storage[task_id]['status'] = 'FAILED'
        task_storage[task_id]['result'] = str(e)

# --- 5. Set up the FastAPI Web Server with an SSE Endpoint ---

app = FastAPI(title="PDF Form Filler Agent (SSE)", version="3.0.0")

class StreamTaskRequest(BaseModel):
    task: str
    client_id: str
    form_number: str
    client_profile: dict

async def event_generator(request: Request, task_id: str):
    """
    This async generator function is the core of the SSE endpoint.
    It yields events back to the client as they become available.
    """
    # First, yield an immediate confirmation event
    yield {
        "event": "task_accepted",
        "data": json.dumps({"task_id": task_id, "status": "PENDING"})
    }
    
    # Now, we wait for the background thread to complete the task
    while True:
        # Check if the client has disconnected
        if await request.is_disconnected():
            print(f"--- SSE: Client disconnected from task {task_id} ---")
            break
            
        task_status = task_storage.get(task_id, {}).get('status')
        
        if task_status in ["COMPLETED", "FAILED"]:
            final_result = task_storage.get(task_id, {}).get('result')
            print(f"--- SSE: Sending final event for task {task_id} ---")
            yield {
                "event": "task_completed",
                "data": json.dumps({"task_id": task_id, "status": task_status, "result": final_result})
            }
            # Clean up the completed task from storage
            del task_storage[task_id]
            break
        
        # Wait for a short period before checking the status again
        await asyncio.sleep(1)

@app.post("/stream_task")
async def stream_task(request: StreamTaskRequest, raw_request: Request):
    """
    Accepts a task, starts it in the background, and returns an
    EventSourceResponse to stream updates back to the client.
    """
    task_id = f"task_{uuid.uuid4()}"
    print(f"--- PDF AGENT: Task {task_id} submitted for streaming. ---")
    
    task_storage[task_id] = {"status": "PENDING", "result": None}

    initial_state = {
        "messages": [AIMessage(content=f"Fill form {request.form_number} for client {request.client_id}.")],
        "client_data": request.client_profile,
        "form_number": request.form_number,
    }

    # Start the agent execution in a background thread
    thread = threading.Thread(target=run_agent_in_background, args=(task_id, initial_state))
    thread.start()

    return EventSourceResponse(event_generator(raw_request, task_id))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
