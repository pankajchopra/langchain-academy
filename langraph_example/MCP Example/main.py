# ============================================================================
# Refactored main.py
# ============================================================================
"""
LangGraph MCP Integration Demo - Fully Async Version (Refactored)
===================================================

Demonstrates a unified approach to using various tools by separating
development and production workflows for better maintainability and scalability.
"""

import os
import asyncio
import subprocess
import sys
import aiohttp
from aiohttp import TCPConnector
from typing import List, Any, Optional
from pathlib import Path
from contextlib import asynccontextmanager

# Core LangGraph and LangChain imports
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
# MCP Integration imports
from langchain_mcp_adapters.client import MultiServerMCPClient

import logging

# ============================================================================
print("🚀 Starting Fully Async LangGraph MCP Integration Demo...")

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ============================================================================
# SECTION 1: ASYNC LOCAL TOOLS
# ============================================================================
print("\n📋 Section 1: Setting up ASYNC LOCAL TOOLS...")

@tool
async def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number using async iterative approach."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        if n > 20:
            await asyncio.sleep(0.01)
            
        for _ in range(2, n + 1):
            a, b = b, a + b
            if n > 100 and _ % 50 == 0:
                await asyncio.sleep(0.001)
        return b

@tool  
async def format_text_stylish(text: str, style: str = "bold") -> str:
    """Format text with different styles: bold, italic, underline, or uppercase."""
    await asyncio.sleep(0.001)
    
    styles = {
        "bold": f"**{text}**",
        "italic": f"_{text}_", 
        "underline": f"__{text}__",
        "uppercase": text.upper(),
        "lowercase": text.lower()
    }
    return styles.get(style, text)

# These are our ASYNC LOCAL TOOLS
local_tools = [calculate_fibonacci, format_text_stylish]
print(f"✅ Created {len(local_tools)} async local tools: {[tool.name for tool in local_tools]}")

# ============================================================================
# SECTION 2: MCP SERVER MANAGEMENT (DEVELOPMENT MODE ONLY)
# ============================================================================
class DevelopmentMCPServerManager:
    """Manages MCP servers as subprocesses for local development."""
    
    def __init__(self):
        self.processes = {}
        
    async def start_all(self):
        """Starts all local MCP servers."""
        print("🔧 Starting development MCP servers...")
        
        server_scripts = {
            "math_server": "stdio-math-mcp-server.py",
            "weather_server": "streamable-http-weather-mcp-server.py"
        }
        
        for name, script in server_scripts.items():
            if not Path(script).exists():
                print(f"⚠️  Warning: {script} not found, skipping {name} server.")
                continue
            
            print(f"   Starting {name} server...")
            try:
                if "stdio" in script:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, script,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                else: # Assuming HTTP server
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, script,
                        env={**dict(os.environ), 'MCP_SERVER_PORT': '8000'},
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                self.processes[name] = proc
                print(f"   ✅ Started {name} with PID: {proc.pid}")
            except Exception as e:
                print(f"❌ Error starting {name} server: {e}")
                
        # Give servers time to initialize
        await asyncio.sleep(5)
        
    async def shutdown_all(self):
        """Gracefully shuts down all managed servers."""
        print("🧹 Shutting down development MCP servers...")
        for name, proc in self.processes.items():
            if proc.returncode is None:
                print(f"   Terminating {name} server...")
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    print(f"   Force killing {name} server...")
                    proc.kill()
        print("✅ All development servers shut down.")

# ============================================================================
# SECTION 3: ASYNC MCP CLIENT SETUP
# ============================================================================
@asynccontextmanager
async def get_mcp_client():
    """Context manager for setting up and tearing down the MCP client."""
    client = None
    server_manager = None
    is_development = os.getenv('ENVIRONMENT', 'development').lower() == 'development'

    try:
        if is_development:
            print("\n⚙️  Running in DEVELOPMENT mode...")
            server_manager = DevelopmentMCPServerManager()
            await server_manager.start_all()
            
            # Use fixed config for local development
            server_config = {
                "math_server": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["stdio-math-mcp-server.py"]
                },
                "weather_server": {
                    "transport": "streamable_http",
                    "url": "http://localhost:8000/mcp"
                }
            }
            client = MultiServerMCPClient(server_config)

        else: # Production mode
            print("\n🚀 Running in PRODUCTION mode...")
            # production_deployment = AsyncProductionDeployment()
            server_config = {
                "math_server": {
                    "transport": "streamable_http", # Assuming production uses HTTP
                    "url": os.getenv('MCP_MATH_SERVER_URL')
                },
                "weather_server": {
                    "transport": "streamable_http",
                    "url": os.getenv('MCP_WEATHER_SERVER_URL')
                }
            }
            # client = await production_deployment.create_production_mcp_client(server_config)

        # yield client
    finally:
        if client:
            try:
                if hasattr(client, 'aclose'):
                    await client.aclose()
            except Exception as e:
                print(f"⚠️  Warning during MCP client cleanup: {e}")
        if server_manager:
            await server_manager.shutdown_all()

# ============================================================================
# SECTION 4: ASYNC UNIFIED TOOL REGISTRATION
# ============================================================================
async def create_async_unified_toolset(client: MultiServerMCPClient) -> List[Any]:
    """Combine local tools and MCP tools into a single unified tool list."""
    print("\n🔧 Section 4: Creating ASYNC UNIFIED TOOLSET...")
    
    mcp_tools = []
    try:
        mcp_tools = await client.get_tools()
        print(f"✅ Retrieved {len(mcp_tools)} tools from MCP servers.")
    except Exception as e:
        print(f"⚠️  Warning: Could not retrieve tools from MCP servers: {e}")
        print("   Continuing with local tools only.")
    
    all_tools = local_tools + mcp_tools
    
    print(f"\n🎯 ASYNC UNIFIED TOOLSET CREATED:")
    print(f"   - Local tools: {len(local_tools)} tools")
    print(f"   - MCP tools: {len(mcp_tools)} tools") 
    print(f"   - Total unified tools: {len(all_tools)} tools")
    
    print("\n🔍 All tools available to the agent:")
    for i, tool in enumerate(all_tools, 1):
        tool_source = "LOCAL" if tool in local_tools else "MCP"
        print(f"   {i}. {tool.name} ({tool_source}) - {tool.description}")
    
    return all_tools

# ============================================================================
# SECTION 5: ASYNC LANGGRAPH AGENT WITH TOOLNODE
# ============================================================================
async def create_async_agent_with_toolnode(tools: List[Any]):
    """Create fully async LangGraph agent using ToolNode for unified tool execution."""
    print("\n🤖 Section 5: Building ASYNC LANGGRAPH AGENT with ToolNode...")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print(f"✅ OPENAI_API_KEY available (last 4 chars: {openai_key[-4:]})")
    else:
        print("❌ OPENAI_API_KEY not found. Please set OPENAI_API_KEY in your .env file.")
        raise ValueError("OPENAI_API_KEY not found")

    model = init_chat_model(
            model="gpt-3.5-turbo",
            model_provider="openai",
            max_retries=1
        )
    
    model_with_tools = model.bind_tools(tools)
    print(f"🔗 Bound {len(tools)} tools to model")
    
    tool_node = ToolNode(tools)
    print("⚙️ Created async ToolNode for unified tool execution")
    
    async def should_continue(state: MessagesState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        await asyncio.sleep(0.001)
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
    
    async def call_model(state: MessagesState) -> dict:
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
    print("📊 Building async LangGraph workflow...")
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue)
    builder.add_edge("tools", "call_model")
    
    agent = builder.compile()
    print("✅ Async LangGraph agent compiled successfully!")
    
    return agent

# ============================================================================
# SECTION 6: ASYNC DEMONSTRATION SCENARIOS
# ============================================================================
async def run_async_demonstrations(agent):
    """Run various demonstrations showing async tool interoperability."""
    print("\n🎪 Section 6: Running ASYNC DEMONSTRATIONS...")
    
    test_cases = [
        {
            "name": "Async Local Tool Test - Fibonacci",
            "query": "Calculate the 15th Fibonacci number",
            "expected_tool": "calculate_fibonacci"
        },
        {
            "name": "Async Local Tool Test - Text Formatting", 
            "query": "Format the text 'Async LangGraph' in bold style",
            "expected_tool": "format_text_stylish"
        },
        {
            "name": "Async MCP Tool Test - Math Operations",
            "query": "What is 25 + 37, then multiply that result by 4?",
            "expected_tool": "add, multiply"
        },
        {
            "name": "Async MCP Tool Test - Weather Query",
            "query": "What's the weather like in Tokyo?",
            "expected_tool": "get_weather"
        },
        {
            "name": "Async Mixed Tools Test",
            "query": "Calculate the 12th Fibonacci number, format it in uppercase, and get weather for London",
            "expected_tool": "multiple tools"
        }
    ]
    
    async def run_single_test(i: int, test: dict):
        print(f"\n{'='*60}")
        print(f"🧪 ASYNC TEST {i}: {test['name']}")
        print(f"🔍 Query: {test['query']}")
        print(f"🎯 Expected tools: {test['expected_tool']}")
        print(f"{'='*60}")
        
        try:
            start_time = asyncio.get_event_loop().time()
            result = await agent.ainvoke({"messages": [HumanMessage(content=test['query'])]})
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            print("📤 ASYNC RESPONSE:")
            final_message = result["messages"][-1]
            print(f"   {final_message.content}")
            print(f"⏱️  Execution time: {duration:.3f} seconds")
            
            all_messages = result["messages"]
            tool_calls_found = [msg for msg in all_messages if hasattr(msg, 'tool_calls') and msg.tool_calls]
            
            if tool_calls_found:
                print("\n🔧 Async tool calls made:")
                for msg in tool_calls_found:
                    for tool_call in msg.tool_calls:
                        print(f"   - {tool_call['name']}: {tool_call.get('args', {})}")
        except Exception as e:
            print(f"❌ Error in async test {i}: {e}")
            
    for i, test in enumerate(test_cases, 1):
        await run_single_test(i, test)
        await asyncio.sleep(0.5)

# ============================================================================
# SECTION 7: PRODUCTION DEPLOYMENT WITH CONNECTION POOLING
# Uncomment later
# ============================================================================
# class AsyncProductionDeployment:
    """Handles production-grade deployment with connection pooling."""
    
    def __init__(self):
        self._session = None
        self._pool = None
        self.max_connections = 100
        self.connection_timeout = 30
        
    async def create_production_mcp_client(self, server_config: dict) -> MultiServerMCPClient:
        """Create MCP client with connection pooling for production use."""
        self._pool = TCPConnector(
            limit=self.max_connections,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        self._session = aiohttp.ClientSession(
            connector=self._pool,
            timeout=aiohttp.ClientTimeout(total=self.connection_timeout),
            headers={'Connection': 'keep-alive'}
        )
        
        # Check if production URLs are set
        for name, config in server_config.items():
            if not config.get('url'):
                print(f"❌ ERROR: URL for production server '{name}' is not set in .env file.")
                raise ValueError("Missing production server URL")
        
        client = MultiServerMCPClient(
            server_config,
            session=self._session,
            retry_config={
                'max_retries': 3,
                'backoff_factor': 1.5,
                'retry_on_status': [500, 502, 503, 504]
            }
        )
        print("✅ Production MCP client created with connection pooling")
        return client

# ============================================================================
# SECTION 8: MAIN ASYNC EXECUTION
# ============================================================================
async def async_main():
    """Main async execution function demonstrating complete MCP integration."""
    print("\n" + "="*80)
    print("🎯 FULLY ASYNC LANGGRAPH MCP INTEGRATION DEMONSTRATION")
    print("="*80)
    
    start_time = asyncio.get_event_loop().time()
    
    # Use the async context manager to handle client and server lifecycle
    async with get_mcp_client() as mcp_client:
        if mcp_client is None:
            print("❌ Failed to set up MCP client. Exiting.")
            return

        # Step 1: Create unified toolset asynchronously
        unified_tools = await create_async_unified_toolset(mcp_client)
        
        # Step 2: Create agent with async ToolNode
        agent = await create_async_agent_with_toolnode(unified_tools)
        
        # Step 3: Run async demonstrations
        await run_async_demonstrations(agent)
    
    total_time = asyncio.get_event_loop().time() - start_time
    print("\n" + "="*80)
    print("🎉 ASYNC DEMONSTRATION COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Total execution time: {total_time:.3f}s")
    print("\n✅ All resources have been properly cleaned up.")

def main():
    """Entry point that properly handles async execution."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()