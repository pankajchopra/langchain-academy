from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from pathlib import Path

# Directory where your files are stored
current_dir = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(current_dir, 'data')

# Create the data directory if it doesn't exist
Path(PDF_DIR).mkdir(exist_ok=True)

app = FastAPI(title='File Download Server')

# Serve static files from the 'data' directory
app.mount("/data", StaticFiles(directory=PDF_DIR), name="data")

# Create the PDF directory if it doesn’t exist

Path(PDF_DIR).mkdir(exist_ok=True)

# Dictionary of files (filename -> [display name, file type])
FILES = {
    'MCP x A2A Framework for Enhancing Interoperability of LLM-based Autonomous Agents2506.01804v2.pdf': ['MCP x A2A Framework for Enhancing Interoperability', 'pdf'],
    'RaptorlargeContextLLM2401.18059v1.pdf': ['RaptorlargeContextLLM', 'pdf'],
    'LangGraph vs ADK_Main difference between these two.xlsx': ['LangGraph vs ADK_Main difference between these two ', 'excel']
}

# MIME types for different file extensions
MIME_TYPES = {
    'pdf': 'application/pdf',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'xls': 'application/vnd.ms-excel'
}

@app.get("/", response_class=HTMLResponse)
async def home():
    """ Main page with download links """

    html_content = """  
    <!DOCTYPE html>
    <html>
    <head>
    <title>File Downloads</title>
    <style>
    body {
    font-family: Arial, sans-serif;
    max-width: 800px;
    margin: 50px auto;
    padding: 20px;
    background-color: #f5f5f5;
    }
    .container {
    background-color: white;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h1 {
    color: #333;
    text-align: center;
    margin-bottom: 30px;
    }
    .pdf-link {
    display: block;
    padding: 15px 20px;
    margin: 10px 0;
    background-color: #007bff;
    color: white;
    text-decoration: none;
    border-radius: 5px;
    transition: background-color 0.3s;
    }
    .pdf-link:hover {
    background-color: #0056b3;
    }
    .pdf-icon {
    margin-right: 10px;
    }
    </style>
    </head>
    <body>
    <div class="container">
    <h1>📁 File Downloads</h1>
    <div class="links">
    """


    # Add download links for each file
    for filename, (display_name, file_type) in FILES.items():
        icon = '📄' if file_type == 'pdf' else '📊'  # PDF or Excel icon
        html_content += f'''
                <a href="/data/{filename}" class="pdf-link">
                    <span class="pdf-icon">{icon}</span>
                    {display_name}
                </a> '''

    html_content += "</div> </div></body></html> "

    return html_content


@app.get("/data/{filename}")
async def download_file(filename: str):
    """
    Download a specific file (PDF or Excel)
    """
    # Check if the filename is in our allowed list
    if filename not in FILES:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = os.path.join(PDF_DIR, filename)

    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
    
    # Get file extension and determine MIME type
    file_ext = filename.split('.')[-1].lower()
    media_type = MIME_TYPES.get(file_ext, 'application/octet-stream')

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )


@app.get("/list")
async def list_files():
    """API endpoint to get list of available files"""
    return {"files": FILES}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")