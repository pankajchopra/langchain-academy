# Async LangGraph MCP Integration - Requirements & Setup

## Requirements.txt for Async Implementation

```bash
# Core async-compatible LangGraph and LangChain packages
langgraph>=0.2.40
langchain>=0.3.9  
langchain-core>=0.3.21
langchain-community>=0.3.8

# MCP Integration - Latest versions with async support
langchain-mcp-adapters>=0.1.9
mcp>=1.1.0
fastmcp>=2.3.0

# LLM Providers (choose one or more)
langchain-openai>=0.2.6
openai>=1.52.0
# langchain-anthropic>=0.2.4
# langchain-google-genai>=2.1.9

# Async HTTP and networking
httpx>=0.25.0
aiohttp>=3.9.0
uvicorn>=0.24.0
fastapi>=0.104.0

# Async utilities
asyncio-mqtt>=0.11.0
aiodns>=3.0.0  # For async DNS resolution

# Development and testing
pytest>=7.4.0
pytest-asyncio>=0.21.0

# Logging and debugging  
rich>=13.7.0
loguru>=0.7.2

# Data validation
pydantic>=2.5.0
```

## File Structure

Your project should have this structure:

```
async_langgraph_mcp/
├── main.py                                    # Main async implementation
├── async-stdio-math-mcp-server.py             # Async math server (stdio)
├── async-streamable-http-weather-mcp-server.py # Async weather server (HTTP)
├── requirements.txt                            # Dependencies
└── README.md                                  # This file
```

## Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv async_langgraph_env

# Activate environment
# On Windows:
async_langgraph_env\Scripts\activate
# On macOS/Linux:
source async_langgraph_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file (or set environment variables):

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Alternative LLM providers
# ANTHROPIC_API_KEY=your_anthropic_key_here
# GOOGLE_API_KEY=your_google_key_here

# Optional: MCP Server Configuration
MCP_SERVER_PORT=8000
```

### 3. Running the Async Demo

```bash
# Run the complete async demonstration
python main.py
```

## Key Async Improvements

### 1. **Fully Async Tool Execution**
- All local tools use `async def` and `await`
- Non-blocking I/O throughout the pipeline
- Proper async delays instead of blocking operations

### 2. **Async Server Management**
- Servers started as async subprocesses
- Graceful async shutdown handling
- Proper resource cleanup with context managers

### 3. **Async MCP Client Operations**
- Non-blocking server connections
- Async tool retrieval from MCP servers
- Concurrent tool execution when possible

### 4. **Performance Benefits**
- Concurrent execution of independent operations
- Non-blocking I/O reduces wait times
- Better resource utilization
- Scalable to many simultaneous requests

## Technical Explanation

### Why Full Async Matters

1. **Non-blocking I/O**: All network calls, file operations, and subprocess communications are non-blocking
2. **Concurrency**: Multiple tools can execute simultaneously when they don't depend on each other
3. **Resource Efficiency**: CPU threads aren't blocked waiting for I/O operations
4. **Scalability**: Can handle many concurrent requests without linear resource growth

### Async Patterns Used

1. **async/await**: All functions that perform I/O are async
2. **asyncio.create_subprocess_exec()**: Non-blocking subprocess creation
3. **asyncio.gather()**: Concurrent execution of multiple async operations
4. **asyncio.create_task()**: Background task management
5. **Signal handlers**: Graceful async shutdown

### Example Async Flow

```python
# Traditional synchronous flow:
result1 = tool1()      # Blocks for 100ms
result2 = tool2()      # Blocks for 200ms  
result3 = tool3()      # Blocks for 150ms
# Total time: 450ms

# Async concurrent flow:
task1 = asyncio.create_task(tool1())  # Starts immediately
task2 = asyncio.create_task(tool2())  # Starts immediately  
task3 = asyncio.create_task(tool3())  # Starts immediately
results = await asyncio.gather(task1, task2, task3)
# Total time: 200ms (max of individual times)
```

## Error Handling

The async implementation includes comprehensive error handling:

- **Server startup failures**: Graceful fallback to local tools only
- **Network timeouts**: Configurable timeouts with proper cleanup
- **Signal handling**: Proper shutdown on CTRL+C or SIGTERM
- **Resource cleanup**: Ensures all async resources are properly closed

## Performance Monitoring

The demo includes timing metrics to show async performance benefits:

- Setup time measurement
- Individual tool execution times
- Total pipeline execution time
- Concurrent vs sequential execution comparisons

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with correct versions
2. **Server Connection Failures**: Check that MCP servers start successfully
3. **API Key Issues**: Verify environment variables are set correctly
4. **Port Conflicts**: Ensure port 8000 is available for the HTTP server

### Debug Mode

Add this to enable detailed async debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
asyncio.get_event_loop().set_debug(True)
```

## Next Steps

1. **Scale Testing**: Test with more concurrent requests
2. **Custom Tools**: Add your own async tools to the pipeline
3. **Error Recovery**: Implement retry logic for failed async operations
4. **Monitoring**: Add metrics collection for production use