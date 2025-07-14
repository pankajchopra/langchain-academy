import uvicorn
from fastapi import FastAPI
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

# --- This is a self-contained Specialized Agent ---
# It runs as its own web server and listens for A2A requests.

# --- 1. Define the Agent's Specialized Tools (with Dummy Data) ---

@tool
def fetch_pdf_template(form_number: str) -> dict:
    """
    Fetches the path to a blank PDF form template.
    
    Args:
        form_number: The number identifying the PDF form.
        
    Returns:
        A dictionary containing the path to the blank PDF.
    """
    print(f"--- PDF AGENT TOOL: Fetching template for form {form_number} ---")
    if form_number == "1234":
        return {"pdf_path": "/templates/money_transfer_1234.pdf"}
    return {"error": "PDF template not found."}

@tool
def fill_and_generate_link(client_data: dict, pdf_path: str) -> dict:
    """
    Fills a PDF with client data and generates a secure download link.
    
    Args:
        client_data: A dictionary with the client's information.
        pdf_path: The path to the blank PDF template.
        
    Returns:
        A dictionary containing the download link.
    """
    print(f"--- PDF AGENT TOOL: Filling form {pdf_path} for {client_data.get('name')} ---")
    # In a real system, this would use a PDF library to fill the form
    # and upload it to a secure storage to generate a link.
    import uuid
    download_link = f"https://secure-downloads.example.com/filled_forms/{uuid.uuid4()}.pdf"
    return {"download_link": download_link}

# --- 2. Define the Agent's State ---

class FormFillerState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    client_data: dict
    form_number: str
    download_link: Optional[str]

# --- 3. Build the Agent's Graph ---

def create_pdf_filler_graph():
    """Creates the LangGraph agent for the PDF filler."""
    tools = [fetch_pdf_template, fill_and_generate_link]
    llm = ChatOpenAI(model="gpt-4o-mini")
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

# --- 4. Set up the FastAPI Web Server ---

# This creates the web server application
app = FastAPI(
    title="PDF Form Filler Agent",
    description="A specialized agent for filling PDF forms via A2A protocol.",
    version="1.0.0",
)

# This defines the data model for the incoming A2A request
from pydantic import BaseModel

class A2ARequest(BaseModel):
    task: str
    client_id: str
    form_number: str
    client_profile: dict # The main agent will pass the data it already has

@app.get("/invoke")
async def invoke_agent():
    """
    GET method is not supported for /invoke. Please use POST with a valid A2ARequest payload.
    """
    return {"error": "GET method is not supported for /invoke. Use POST with a JSON payload."}


@app.get("/")
async def root():
    """
    Root endpoint providing information about the PDF Form Filler Agent API.
    """
    return {
        "message": "Welcome to the PDF Form Filler Agent API.",
        "description": "This API allows you to fill PDF forms via an agent-to-agent (A2A) protocol.",
        "endpoints": {
            "/invoke (POST)": "Submit a JSON payload to fill a PDF form and receive a download link."
        },
        "version": "1.0.0"
    }
    

@app.post("/invoke")
async def invoke_agent(request: A2ARequest):
    """
    The main endpoint for A2A communication.
    """
    print(f"--- PDF AGENT: Received A2A request for task '{request.task}' ---")
    
    # We construct the initial state for our agent from the request
    initial_state = {
        "messages": [
            AIMessage(
                content=f"Task received: {request.task}. "
                        f"Fill form {request.form_number} for client {request.client_id}. "
                        "I will first fetch the PDF template, then fill it and generate a link."
            )
        ],
        "client_data": request.client_profile,
        "form_number": request.form_number,
    }
    
    # Run the agent graph
    final_state = pdf_filler_app.invoke(initial_state)
    
    # Extract the final response
    final_response = final_state["messages"][-1].content
    
    return {"status": "SUCCESS", "response": final_response}


if __name__ == "__main__":
    # This command runs the web server when you execute the script.
    # It will be accessible at http://127.0.0.1:8000
    uvicorn.run("form_agent:app", host="127.0.0.1", port=8000, reload=True)


