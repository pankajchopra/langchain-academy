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

async def main():
    """Main async entry point for the MCP server."""
    print("[Async-MCP-Math] Starting Async Math MCP Server with stdio transport...", flush=True)
    
    try:
        # Run with stdio transport - enables async subprocess communication
        await mcp.arun(transport="stdio")
    except Exception as e:
        print(f"[Async-MCP-Math] Error: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())