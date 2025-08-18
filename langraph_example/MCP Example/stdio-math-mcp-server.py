# ============================================================================
# FILE 1: async-stdio-math-mcp-server.py
# ============================================================================
"""
Async Math MCP Server - Local stdio Transport
==============================================

This is a fully async local MCP server that provides mathematical operations
via stdio transport with proper async handling.
"""

import asyncio
import sys
from mcp.server.fastmcp import FastMCP

# Create async MCP server instance
mcp = FastMCP("AsyncMathServer")

@mcp.tool()
async def add(a: int, b: int) -> int:
    """Add two numbers together and return the result asynchronously."""
    # Simulate async computation
    await asyncio.sleep(0.001)
    result = a + b
    print(f"[Async-MCP-Math] Adding {a} + {b} = {result}", flush=True)
    return result

@mcp.tool()  
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers together and return the result asynchronously."""
    # Simulate async computation
    await asyncio.sleep(0.001)
    result = a * b
    print(f"[Async-MCP-Math] Multiplying {a} * {b} = {result}", flush=True)
    return result

@mcp.tool()
async def subtract(a: int, b: int) -> int:
    """Subtract second number from first number and return the result asynchronously."""
    await asyncio.sleep(0.001)
    result = a - b
    print(f"[Async-MCP-Math] Subtracting {a} - {b} = {result}", flush=True)
    return result

@mcp.tool()
async def divide(a: int, b: int) -> float:
    """Divide first number by second number and return the result asynchronously."""
    await asyncio.sleep(0.001)
    if b == 0:
        raise ValueError("Cannot divide by zero")
    result = a / b
    print(f"[Async-MCP-Math] Dividing {a} / {b} = {result}", flush=True)
    return result

@mcp.tool()
async def power(a: int, b: int) -> int:
    """Calculate a to the power of b asynchronously."""
    # For large powers, add progressive delays
    if b > 10:
        await asyncio.sleep(0.01)
    result = a ** b
    print(f"[Async-MCP-Math] Power {a}^{b} = {result}", flush=True)
    return result

def cleanup_event_loop():
    """Clean up any existing event loops."""
    try:
        # Get the current event loop
        loop = asyncio.get_event_loop()
        
        # Close the loop if it's running
        if loop.is_running():
            print("[Async-MCP-Math] Closing existing event loop...", flush=True)
            loop.stop()
            loop.close()
            
        # Remove the loop from the current thread
        asyncio.set_event_loop(None)
        
    except Exception as e:
        print(f"[Async-MCP-Math] Warning during cleanup: {e}", flush=True)

# The 'main' function no longer needs to be async if mcp.run() is a blocking
# call that manages its own event loop. It just becomes the entry point for mcp.run().
def main():
    """Main async entry point for the MCP server."""
    print("[Async-MCP-Math] Starting Async Math MCP Server with stdio transport...", flush=True)
    try:
        # This assumes mcp.run() handles the entire asyncio event loop lifecycle.
        # It is a blocking call that keeps the server running until terminated.
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"[Async-MCP-Math] Error: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    # Clean up any existing event loops first
    print("[Async-MCP-Math] Ensuring clean event loop for startup...", flush=True)
    cleanup_event_loop()
    
    # Create and set a new event loop
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    
    try:
        # Try to get the running loop if it exists (e.g., in an interactive shell)
        loop = asyncio.get_running_loop()
        # If a loop is already running, run main as a task on that loop
        loop.create_task(main())
        print("[Async-MCP-Math] Main task scheduled on existing loop.", flush=True)
        # You might need to add a mechanism to keep the loop running
        # if this script is meant to be the primary loop controller.
        # For stdio transport, mcp.run() itself often blocks.
    except RuntimeError:
        # No running loop, so create and run a new one
        print("[Async-MCP-Math] No existing loop found. Starting new loop.", flush=True)
        asyncio.run(main()) # This creates and manages the loop
    finally:
        new_loop.close()