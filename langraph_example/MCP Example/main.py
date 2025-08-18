"""
LangGraph MCP Integration Demo - Fully Async Version
===================================================

Demonstrates a unified approach to using:
1. Two local Python tools wrapped as LangGraph ToolNodes
2. Two tools behind a local MCP server (stdio transport)
3. Two tools behind a remote MCP server (streamable_http transport)

All operations are fully asynchronous for optimal performance.
"""

import gc
import os
import tracemalloc
import asyncio
import subprocess
import time
import signal
import sys
import aiohttp
from aiohttp import TCPConnector
from typing import List, Any, Optional
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
logging.basicConfig(level=logging.INFO)
# asyncio.get_event_loop().set_debug(True)
# ============================================================================
print("🚀 Starting Fully Async LangGraph MCP Integration Demo...")

# Load environment variables from .env file
load_dotenv()

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
        # Add small async delay to simulate computation for large numbers
        if n > 20:
            await asyncio.sleep(0.01)  # Non-blocking delay
            
        for _ in range(2, n + 1):
            a, b = b, a + b
            # Yield control periodically for very large numbers
            if n > 100 and _ % 50 == 0:
                await asyncio.sleep(0.001)
        return b

@tool  
async def format_text_stylish(text: str, style: str = "bold") -> str:
    """Format text with different styles: bold, italic, underline, or uppercase."""
    # Simulate async processing delay
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
# SECTION 2: ASYNC MCP SERVER MANAGEMENT
# ============================================================================
print("\n🔌 Section 2: Setting up ASYNC MCP SERVER MANAGEMENT...")

class AsyncMCPServerManager:
    """Manages MCP servers with full async lifecycle management."""
    
    def __init__(self):
        self.servers = {}
        self.tasks = {}
        self.shutdown_event = asyncio.Event()
        
    async def start_stdio_server(self, name: str, script_path: str) -> subprocess.Popen:
        """Start a stdio MCP server as subprocess."""
        print(f"🔧 Starting stdio server: {name}")
        
        # Start subprocess asynchronously
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give server time to initialize
        await asyncio.sleep(0.5)
        
        if process.returncode is not None:
            stderr = await process.stderr.read()
            raise RuntimeError(f"Server {name} failed to start: {stderr.decode()}")
            
        print(f"✅ Started stdio server {name} with PID: {process.pid}")
        self.servers[name] = process
        return process
        
    async def start_http_server(self, name: str, script_path: str, port: int = 8000) -> asyncio.Task:
        """Start an HTTP MCP server as async task."""
        print(f"🌐 Starting HTTP server: {name} on port {port}")
        
        async def run_server():
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**dict(os.environ), 'MCP_SERVER_PORT': str(port)} if 'os' in globals() else None
            )
            
            # Wait for server to be ready
            await asyncio.sleep(2)
            
            if process.returncode is not None:
                stderr = await process.stderr.read()
                raise RuntimeError(f"HTTP Server {name} failed to start: {stderr.decode()}")
                
            print(f"✅ Started HTTP server {name} on port {port}")
            self.servers[name] = process
            
            # Keep server running
            await process.wait()
            
        task = asyncio.create_task(run_server())
        self.tasks[name] = task
        return task
        
    async def shutdown_all(self):
        """Gracefully shutdown all managed servers."""
        print("🧹 Shutting down all MCP servers...")
        
        # Cancel HTTP server tasks
        for name, task in self.tasks.items():
            if not task.done():
                print(f"   Cancelling HTTP server: {name}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        # Terminate stdio servers
        for name, process in self.servers.items():
            if hasattr(process, 'terminate'):
                print(f"   Terminating stdio server: {name}")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    print(f"   Force killing server: {name}")
                    process.kill()
                    
        print("✅ All servers shutdown complete")

# Global server manager instance
server_manager = AsyncMCPServerManager()

# ============================================================================
# SECTION 3: ASYNC MCP CLIENT SETUP
# ============================================================================
print("\n🔌 Section 3: Setting up ASYNC MCP CLIENT...")

async def setup_async_mcp_client() -> tuple[MultiServerMCPClient, List[Any]]:
    """Initialize MCP client with full async server management."""
    
    # Start MCP servers asynchronously
    print("🚀 Starting MCP servers...")
    
    try:
        
        from pathlib import Path
        # Start HTTP weather server start application from this root directory
        if Path("stdio-math-mcp-server.py").exists():
            print("   Found stdio-math-mcp-server.py, starting HTTP server...")
        else:
            print("   WARNING: stdio-math-mcp-server.py not found, skipping HTTP server startup")
            return None, []

        
       
        # Check if the HTTP server script exists
        if Path("streamable-http-weather-mcp-server.py").exists():
            print("   Found streamable-http-weather-mcp-server.py, starting HTTP server...")
        else:
            print("   WARNING: streamable-http-weather-mcp-server.py not found, skipping HTTP server startup")
            return None, []
        async with asyncio.TaskGroup() as tg:
            # Start stdio math server
            stdio_task = asyncio.create_task(
                server_manager.start_stdio_server("math_server", "stdio-math-mcp-server.py")
            )
            # Start the HTTP server task
            http_task = asyncio.create_task(
                server_manager.start_http_server("weather_server", "streamable-http-weather-mcp-server.py", 8000)
            )
            http_task.add_done_callback(lambda t: print(f"HTTP server task completed: {t.result()}"))
        
        # Wait for both servers to start
        print("   Waiting for MCP servers to initialize...")
        # Allow some time for servers to start
        await asyncio.sleep(5)
        # Wait for servers to be ready
        await asyncio.gather(stdio_task, return_exceptions=True)
        await asyncio.sleep(2)  # Give HTTP server extra time
        
    except asyncio.CancelledError:
        print("⚠️  Server startup was cancelled")
        raise
    except Exception as e:
        print(f"⚠️  Error starting MCP servers: {str(e)}")
        print("   Attempting to continue with reduced functionality...")
        # Optionally re-raise if you want to halt execution
        # raise
    
    # Configuration for multiple MCP servers
    server_config = {
        # LOCAL MCP SERVER - stdio transport
        "math_server": {
            "command": "python",
            "args": ["stdio-math-mcp-server.py"],
            "transport": "stdio",
        },
        # REMOTE MCP SERVER - streamable_http transport 
        "weather_server": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
    
    print("📡 Initializing MultiServerMCPClient...")
    print("   - math_server: stdio transport (local subprocess)")
    print("   - weather_server: streamable_http transport (remote HTTP)")

    # Create the multi-server MCP client
    client = MultiServerMCPClient(server_config)
    
    # Get tools from all connected MCP servers
    print("🔍 Fetching tools from MCP servers...")
    try:
        mcp_tools = None
        await asyncio.sleep(2)  # Wait for servers to stabilize
        mcp_tools = await client.get_tools()
        print(f"   Raw MCP tools response: {mcp_tools}")  # Debug output
    
        if not mcp_tools:
            print("⚠️  Warning: No tools retrieved from MCP servers")
            print("   Checking individual server connections...")
            mcp_tools = await client.get_tools()
            # Try individual server connections
            for server_name, config in server_config.items():
                try:
                    print(f"   Testing connection to {server_name}...")
                    if config.get('transport') == 'stdio':
                        # Check if stdio server process is running
                        process = server_manager.servers.get(server_name)
                        if process and process.returncode is None:
                            print(f"   ✅ {server_name} process is running")
                        else:
                            print(f"   ❌ {server_name} process is not running")
                    else:
                        # Test HTTP server connection
                        url = config.get('url')
                        if url:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(f"{url}/health") as response:
                                    print(f"   {server_name} health check: {response.status}")
                except Exception as server_error:
                    print(f"   ❌ Error connecting to {server_name}: {server_error}")
        
                # Try restarting servers
                print("   Attempting server restart...")
                await server_manager.shutdown_all()
                await asyncio.sleep(2)
                await setup_async_mcp_client()
        
        print(f"✅ Retrieved {len(mcp_tools)} tools from MCP servers:")
        for tool in mcp_tools:
            print(f"   - {tool.name}: {tool.description}")
        if len(mcp_tools) <= 2:
            print(" only two tools available so shutting down and restarting...")
            # Shutdown servers after fetching tools
            await server_manager.shutdown_servers()

    except Exception as e:
        print(f"⚠️  Warning: Could not connect to MCP servers: {e}")
        print("   Continuing with local tools only...")
        mcp_tools = []
        
    
    return client, mcp_tools

# ============================================================================
# SECTION 4: ASYNC UNIFIED TOOL REGISTRATION
# ============================================================================
async def create_async_unified_toolset(existing_client: Optional[MultiServerMCPClient] = None) -> tuple[Optional[MultiServerMCPClient], List[Any]]:
    """Combine local tools and MCP tools into a single unified tool list."""
    print("\n🔧 Section 4: Creating ASYNC UNIFIED TOOLSET...")
    mcp_client, mcp_tools = None, None
    try:
        if existing_client:
            print("🔄 Using existing production MCP client")
            mcp_client = existing_client
            mcp_tools = await mcp_client.get_tools()
        else:
            # Get MCP tools asynchronously with new client
            print("🔄 Not found production MCP client so creating a new development")
            mcp_client, mcp_tools = await setup_async_mcp_client()
    except Exception as e:
        print(f"⚠️  Could not set up MCP client: {e}")
        print("   Continuing with local tools only...")
        mcp_client, mcp_tools = None, []
    
    # Combine all tools - this demonstrates the key insight:
    # Local tools and MCP tools are treated identically by LangGraph
    all_tools = local_tools + mcp_tools
    
    print(f"\n🎯 ASYNC UNIFIED TOOLSET CREATED:")
    print(f"   - Local tools: {len(local_tools)} tools")
    print(f"   - MCP tools: {len(mcp_tools)} tools") 
    print(f"   - Total unified tools: {len(all_tools)} tools")
    
    print("\n🔍 All tools available to the agent:")
    for i, tool in enumerate(all_tools, 1):
        tool_source = "LOCAL" if tool in local_tools else "MCP"
        print(f"   {i}. {tool.name} ({tool_source}) - {tool.description}")
    
    return mcp_client, all_tools

# ============================================================================
# SECTION 5: ASYNC LANGGRAPH AGENT WITH TOOLNODE
# ============================================================================
async def create_async_agent_with_toolnode(tools: List[Any]):
    """Create fully async LangGraph agent using ToolNode for unified tool execution."""
    print("\n🤖 Section 5: Building ASYNC LANGGRAPH AGENT with ToolNode...")
    
    # Initialize the LLM asynchronously
    print("🧠 Initializing LLM: OpenAI GPT-4.1 Preview...")
    
    # Use async-compatible model initialization
    # model = init_chat_model("gpt-5-nano-2025-08-07")
    # Check for OpenAI API key
    
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if openai_key:
        print(f"✅ OPENAI_API_KEY available (last 4 chars: {openai_key[-4:]})")
    else:
        print("❌ OPENAI_API_KEY not found")

    # Initialize the model
    model = init_chat_model(
            model="gpt-3.5-turbo",
            model_provider="openai",  # <-- specify provider
            max_retries=1
        )
    # Alternative async-friendly options:
    # model = init_chat_model("anthropic:claude-3-5-sonnet-latest")
    # model = init_chat_model("google:gemini-2.5-pro-latest")
    
    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)
    print(f"🔗 Bound {len(tools)} tools to model")
    
    # Create ToolNode - this provides execution parity for all tool types
    # All tools (local and MCP) are executed asynchronously
    tool_node = ToolNode(tools)
    print("⚙️ Created async ToolNode for unified tool execution")
    
    # Define async routing logic
    async def should_continue(state: MessagesState) -> str:
        """Determine whether to continue with tool calls or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Add small async delay for state processing
        await asyncio.sleep(0.001)
        
        # If the last message has tool calls, route to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
    
    # Define async model calling function
    async def call_model(state: MessagesState) -> dict:
        """Call the model with current messages asynchronously."""
        messages = state["messages"]
        
        # Async model invocation
        response = await model_with_tools.ainvoke(messages)
        
        return {"messages": [response]}
    
    # Build the async LangGraph workflow
    print("📊 Building async LangGraph workflow...")
    builder = StateGraph(MessagesState)
    
    # Add nodes (all async)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)  # ToolNode handles async execution
    
    # Add edges
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue)
    builder.add_edge("tools", "call_model")  # Loop back after tool execution
    
    # Compile the graph
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
    
    # Run all tests concurrently for maximum async efficiency
    async def run_single_test(i: int, test: dict):
        """Run a single test case asynchronously."""
        print(f"\n{'='*60}")
        print(f"🧪 ASYNC TEST {i}: {test['name']}")
        print(f"🔍 Query: {test['query']}")
        print(f"🎯 Expected tools: {test['expected_tool']}")
        print(f"{'='*60}")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Async agent invocation
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=test['query'])]
            })
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            # Display results
            print("📤 ASYNC RESPONSE:")
            final_message = result["messages"][-1]
            print(f"   {final_message.content}")
            print(f"⏱️  Execution time: {duration:.3f} seconds")
            
            # Show tool calls if any
            all_messages = result["messages"]
            tool_calls_found = []
            for msg in all_messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls_found.extend(msg.tool_calls)
            
            if tool_calls_found:
                print("\n🔧 Async tool calls made:")
                for tool_call in tool_calls_found:
                    print(f"   - {tool_call['name']}: {tool_call.get('args', {})}")
                    
        except Exception as e:
            print(f"❌ Error in async test {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return i
    
    # Option 1: Run tests sequentially (safer for demonstration)
    for i, test in enumerate(test_cases, 1):
        await run_single_test(i, test)
        await asyncio.sleep(0.5)  # Brief pause between tests
    
    # Option 2: Run tests concurrently (uncomment for maximum async performance)
    # tasks = [run_single_test(i, test) for i, test in enumerate(test_cases, 1)]
    # await asyncio.gather(*tasks, return_exceptions=True)

# ============================================================================
# SECTION 7: ASYNC CLEANUP AND SIGNAL HANDLING
# ============================================================================
async def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    if os.name == 'nt':  # Windows
        # Windows only supports SIGINT (Ctrl+C) and SIGBREAK (Ctrl+Break)
        def win_handler(signum, frame):
            print("\n🛑 Shutdown signal received...")
            server_manager.shutdown_event.set()
            
        import signal
        signal.signal(signal.SIGINT, win_handler)
        signal.signal(signal.SIGBREAK, win_handler)
    else:  # Unix-like systems
        def signal_handler():
            print("\n🛑 Shutdown signal received...")
            server_manager.shutdown_event.set()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(sig, signal_handler)

async def cleanup_resources(mcp_client: Optional[MultiServerMCPClient]):
    """Clean up all resources asynchronously."""
    print("\n🧹 Starting async cleanup...")
    
    # Shutdown MCP servers
    await server_manager.shutdown_all()
    
    # Close MCP client if exists
    if mcp_client:
        try:
            # Add proper async cleanup if client supports it
            if hasattr(mcp_client, 'aclose'):
                await mcp_client.aclose()
            elif hasattr(mcp_client, 'close'):
                mcp_client.close()
        except Exception as e:
            print(f"⚠️  Warning during MCP client cleanup: {e}")
            
    # Clean up production deployment resources if in production mode
    if os.getenv('ENVIRONMENT', 'development').lower() == 'production':
        try:
            await production_deployment.cleanup()
        except Exception as e:
            print(f"⚠️  Warning during production deployment cleanup: {e}")
    
    print("✅ Async cleanup complete")

# ============================================================================
# SECTION 8: PRODUCTION DEPLOYMENT WITH CONNECTION POOLING
# ============================================================================
class AsyncProductionDeployment:
    """Handles production-grade deployment with connection pooling."""
    
    def __init__(self):
        self._session = None
        self._pool = None
        self.max_connections = 100
        self.min_connections = 10
        self.connection_timeout = 30
        
    async def setup_connection_pool(self):
        """Initialize connection pool for production use."""
        # import aiohttp
        # from aiohttp import TCPConnector
        
        # Create connection pool with specified limits
        self._pool = TCPConnector(
            limit=self.max_connections,  # Max number of concurrent connections
            limit_per_host=20,  # Max concurrent connections per host
            keepalive_timeout=60,  # Keep connections alive for 60 seconds
            enable_cleanup_closed=True  # Automatically cleanup closed connections
        )
        
        # Create client session with connection pooling
        self._session = aiohttp.ClientSession(
            connector=self._pool,
            timeout=aiohttp.ClientTimeout(total=self.connection_timeout),
            headers={'Connection': 'keep-alive'}
        )
        
        print(f"✅ Connection pool initialized with {self.max_connections} max connections")
        return self._session
    
    async def get_client_session(self) -> aiohttp.ClientSession:
        """Get or create a client session with connection pooling."""
        if self._session is None or self._session.closed:
            await self.setup_connection_pool()
        return self._session
    
    async def create_production_mcp_client(self, server_config: dict) -> MultiServerMCPClient:
        """Create MCP client with connection pooling for production use."""
        session = await self.get_client_session()
        
        # Create MCP client with the pooled session
        client = MultiServerMCPClient(
            server_config,
            session=session,
            retry_config={
                'max_retries': 3,
                'backoff_factor': 1.5,
                'retry_on_status': [500, 502, 503, 504]
            }
        )
        
        print("✅ Production MCP client created with connection pooling")
        return client
    
    async def cleanup(self):
        """Cleanup connection pool and sessions."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._pool and not self._pool.closed:
            await self._pool.close()
        print("✅ Connection pool and sessions cleaned up")

# Global production deployment instance
production_deployment = AsyncProductionDeployment()

# ============================================================================
# SECTION 9: ASYNC RESOURCE MONITORING
# ============================================================================
class AsyncResourceMonitor:
    """Monitors system resources asynchronously."""
    
    def __init__(self):
        self.monitoring = False
        self._task = None
        self.metrics = {}
        
    async def get_system_metrics(self):
        """Get current system metrics."""
        import psutil
        
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu,
            'memory_percent': memory.percent,
            'memory_used': memory.used / (1024 * 1024 * 1024),  # Convert to GB
            'disk_percent': disk.percent,
            'disk_used': disk.used / (1024 * 1024 * 1024)  # Convert to GB
        }
    
    async def monitor_resources(self):
        """Continuously monitor system resources."""
        while self.monitoring:
            self.metrics = await self.get_system_metrics()
            print("\n📊 System Resources:")
            print(f"   CPU Usage: {self.metrics['cpu_percent']}%")
            print(f"   Memory: {self.metrics['memory_percent']}% ({self.metrics['memory_used']:.1f}GB)")
            print(f"   Disk: {self.metrics['disk_percent']}% ({self.metrics['disk_used']:.1f}GB)")
            await asyncio.sleep(5)  # Update every 5 seconds
    
    async def start(self):
        """Start resource monitoring."""
        try:
            import psutil
        except ImportError:
            print("Installing psutil for resource monitoring...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
            import psutil
            
        self.monitoring = True
        self._task = asyncio.create_task(self.monitor_resources())
        print("✅ Resource monitoring started")
    
    async def stop(self):
        """Stop resource monitoring."""
        if self.monitoring:
            self.monitoring = False
            if self._task:
                await self._task
            print("✅ Resource monitoring stopped")

# Global resource monitor instance
resource_monitor = AsyncResourceMonitor()

# ============================================================================
# SECTION 9: MAIN ASYNC EXECUTION
# ============================================================================
async def async_main():
    """Main async execution function demonstrating complete MCP integration."""
    print("\n" + "="*80)
    print("🎯 FULLY ASYNC LANGGRAPH MCP INTEGRATION DEMONSTRATION")
    print("="*80)
    
    mcp_client = None
    
    try:
        # Set up signal handlers
        await setup_signal_handlers()
        
        # Start resource monitoring
        await resource_monitor.start()
        
        # Initialize production deployment if in production mode
        is_production = os.getenv('ENVIRONMENT', 'development').lower() == 'production'
        if is_production:
            print("🚀 Starting in PRODUCTION mode with connection pooling")
            # Setup production client with connection pooling
            #MAKE SURE ADD MCP_SERVER_URL to .env file
            mcp_client = await production_deployment.create_production_mcp_client({
                'server_url': os.getenv('MCP_SERVER_URL', 'http://localhost:8000'),
                'timeout': 30
            })
        
        # Step 1: Create unified toolset asynchronously
        start_time = asyncio.get_event_loop().time()
        tracemalloc.start()
        mcp_client, unified_tools = await create_async_unified_toolset(existing_client=mcp_client)
        setup_time = asyncio.get_event_loop().time() - start_time
        print(f"⏱️  Async setup completed in {setup_time:.3f} seconds")
        
        # Step 2: Create agent with async ToolNode
        agent_start = asyncio.get_event_loop().time()
        agent = await create_async_agent_with_toolnode(unified_tools)
        agent_time = asyncio.get_event_loop().time() - agent_start
        print(f"⏱️  Async agent creation completed in {agent_time:.3f} seconds")
        
        # Step 3: Run async demonstrations
        demo_start = asyncio.get_event_loop().time()
        await run_async_demonstrations(agent)
        demo_time = asyncio.get_event_loop().time() - demo_start
        
        total_time = asyncio.get_event_loop().time() - start_time
        
        print("\n" + "="*80)
        print("🎉 ASYNC DEMONSTRATION COMPLETE!")
        print("="*80)
        print("\n📋 ASYNC PERFORMANCE METRICS:")
        print(f"   ⏱️  Setup time: {setup_time:.3f}s")
        print(f"   ⏱️  Agent creation: {agent_time:.3f}s") 
        print(f"   ⏱️  Demo execution: {demo_time:.3f}s")
        print(f"   ⏱️  Total time: {total_time:.3f}s")
        
        print("\n📋 KEY ASYNC TAKEAWAYS:")
        print("   ✅ All operations are fully asynchronous")
        print("   ✅ Local and MCP tools execute with async parity")
        print("   ✅ Server lifecycle managed asynchronously") 
        print("   ✅ Concurrent execution improves performance")
        print("   ✅ Graceful async shutdown handling")
        print("   ✅ Non-blocking I/O throughout the pipeline")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in async main execution: {str(e)}")
        tracemalloc.print_exc()
    finally:
        # Always clean up resources
        await resource_monitor.stop()
        await cleanup_resources(mcp_client)

# ============================================================================
# SECTION 9: ENTRY POINT WITH PROPER ASYNC HANDLING
# ============================================================================
def main():

    """Entry point that properly handles async execution."""
    print("🚀 Starting fully async demonstration...")
    
    try:
        # Python 3.7+ async entry point
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        asyncio.sleep(10)
        tracemalloc.print_exc()
        tracemalloc.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()