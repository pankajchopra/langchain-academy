#!/bin/bash
# Start the async LangGraph MCP demo

echo "Starting Async LangGraph MCP Demo..."

# Check if virtual environment exists
if [ ! -d "../../lc-academy-env" ]; then
    echo " Creating virtual environment..."
    python3 -m venv ../../lc-academy-env
fi

# Activate virtual environment
echo "Activating virtual environment..."
# source ../../lc-academy-env/bin/activate

# Install dependencies if needed
if ! pip show langgraph &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Creating template..."
    echo "OPENAI_API_KEY=your_key_here" > .env
    echo "MCP_SERVER_PORT=8000" >> .env
    echo "Please edit .env file with your API keys"
fi

# Run the demo
echo "▶️  Running demo..."
python main.py
