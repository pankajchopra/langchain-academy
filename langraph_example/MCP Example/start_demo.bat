@echo off
REM Start the async LangGraph MCP demo

echo 🚀 Starting Async LangGraph MCP Demo...

REM Check if virtual environment exists
@REM if not exist "lc-academy-env" (
@REM     echo Creating virtual environment...
@REM     python -m venv lc-academy-env
@REM )

REM Activate virtual environment
echo Activating virtual environment...
call lc-academy-env\Scripts\activate.bat

REM Install dependencies if needed
pip show langgraph >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
)

REM Check for .env file
if not exist ".env" (
    echo ⚠️  Warning: .env file not found. Creating template...
    echo OPENAI_API_KEY=your_key_here> .env
    echo MCP_SERVER_PORT=8000>> .env
    echo Please edit .env file with your API keys
)

REM Run the demo
echo ▶️  Running demo...
python main.py

pause
