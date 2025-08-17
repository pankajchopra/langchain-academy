"""
Async Weather MCP Server - Remote HTTP Transport
===============================================

This is a fully async remote MCP server that provides weather information
via streamable_http transport with proper async handling.
"""

import asyncio
import random
import sys
from mcp.server.fastmcp import FastMCP

import os

# Create async MCP server instance
mcp = FastMCP("AsyncWeatherServer")

# Expanded mock weather data for demonstration
WEATHER_DATA = {
    "new york": "Sunny, 72°F (22°C), light breeze from the southwest",
    "san francisco": "Foggy, 58°F (14°C), high humidity with coastal winds", 
    "london": "Cloudy, 61°F (16°C), chance of light rain in the afternoon",
    "tokyo": "Partly cloudy, 75°F (24°C), mild humidity with gentle winds",
    "paris": "Overcast, 64°F (18°C), scattered showers expected",
    "sydney": "Warm and sunny, 78°F (26°C), clear skies with ocean breeze",
    "toronto": "Cool and crisp, 55°F (13°C), clear skies with light winds",
    "berlin": "Cold and windy, 48°F (9°C), overcast with possible snow flurries",
    "los angeles": "Sunny and warm, 78°F (26°C), clear skies with light winds",
    "chicago": "Windy, 52°F (11°C), partly cloudy with gusty conditions",
    "miami": "Hot and humid, 84°F (29°C), scattered thunderstorms possible",
    "seattle": "Drizzle, 58°F (14°C), overcast with light rain expected"
}

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get current weather information for the specified location asynchronously."""
    location_clean = location.lower().strip()
    
    print(f"[Async-MCP-Weather] Fetching weather for: {location}", flush=True)
    
    # Simulate async API call delay
    await asyncio.sleep(0.1)
    
    # Check if we have data for this location
    if location_clean in WEATHER_DATA:
        weather = WEATHER_DATA[location_clean]
        print(f"[Async-MCP-Weather] Found weather data for {location}", flush=True)
        return f"Weather in {location.title()}: {weather}"
    else:
        # Generate a random weather response for unknown locations
        conditions = ["Sunny", "Cloudy", "Partly cloudy", "Rainy", "Overcast"]
        temps = list(range(45, 85))
        condition = random.choice(conditions)
        temp_f = random.choice(temps)
        temp_c = int((temp_f - 32) * 5/9)
        
        weather = f"{condition}, {temp_f}°F ({temp_c}°C), typical weather patterns"
        print(f"[Async-MCP-Weather] Generated weather for unknown location {location}", flush=True)
        return f"Weather in {location.title()}: {weather}"

@mcp.tool()
async def get_forecast(location: str, days: int = 3) -> str:
    """Get weather forecast for the specified location over the next few days asynchronously."""
    location_clean = location.lower().strip()
    
    print(f"[Async-MCP-Weather] Fetching {days}-day forecast for: {location}", flush=True)
    
    # Simulate async processing time based on forecast length
    await asyncio.sleep(0.05 * min(days, 7))
    
    # Generate forecast data
    conditions = ["Sunny", "Partly cloudy", "Cloudy", "Light rain", "Overcast", "Clear", "Thunderstorms"]
    forecast_lines = []
    
    for day in range(1, min(days + 1, 8)):  # Limit to 7 days max
        condition = random.choice(conditions)
        high_temp = random.randint(60, 85)
        low_temp = random.randint(45, high_temp - 10)
        
        day_names = ["Tomorrow", "Day after tomorrow", "In 3 days", "In 4 days", 
                    "In 5 days", "In 6 days", "In 7 days"]
        day_name = day_names[day - 1] if day <= len(day_names) else f"In {day} days"
        
        forecast_lines.append(f"{day_name}: {condition}, High {high_temp}°F, Low {low_temp}°F")
        
        # Small async delay between forecast days
        await asyncio.sleep(0.01)
    
    forecast = "\n".join(forecast_lines)
    return f"{days}-day forecast for {location.title()}:\n{forecast}"

@mcp.tool()
async def get_weather_alerts(location: str) -> str:
    """Get weather alerts and warnings for the specified location asynchronously."""
    location_clean = location.lower().strip()
    
    print(f"[Async-MCP-Weather] Checking weather alerts for: {location}", flush=True)
    
    # Simulate async alert checking
    await asyncio.sleep(0.05)
    
    # Random alert generation for demo
    alert_chance = random.random()
    
    if alert_chance < 0.3:  # 30% chance of alert
        alerts = [
            "High wind warning in effect until 6 PM",
            "Flood watch issued for low-lying areas",
            "Heat advisory - temperatures may exceed 95°F",
            "Winter storm watch - snow accumulation possible",
            "Severe thunderstorm warning until 8 PM"
        ]
        alert = random.choice(alerts)
        return f"⚠️ Weather Alert for {location.title()}: {alert}"
    else:
        return f"✅ No active weather alerts for {location.title()}"

@mcp.tool()
async def get_air_quality(location: str) -> str:
    """Get air quality index information for the specified location asynchronously."""
    location_clean = location.lower().strip()
    
    print(f"[Async-MCP-Weather] Checking air quality for: {location}", flush=True)
    
    # Simulate async API call
    await asyncio.sleep(0.08)
    
    # Generate random AQI data
    aqi_value = random.randint(15, 200)
    
    if aqi_value <= 50:
        category = "Good"
        color = "🟢"
    elif aqi_value <= 100:
        category = "Moderate"
        color = "🟡"
    elif aqi_value <= 150:
        category = "Unhealthy for Sensitive Groups"
        color = "🟠"
    else:
        category = "Unhealthy"
        color = "🔴"
    
    return f"Air Quality in {location.title()}: {color} AQI {aqi_value} ({category})"

async def main():
    """Main async entry point for the weather MCP server."""
    port = int(os.getenv('MCP_SERVER_PORT', 8000))
    
    print(f"[Async-MCP-Weather] Starting Async Weather MCP Server on port {port}...", flush=True)
    print("[Async-MCP-Weather] Server will use streamable_http transport", flush=True)
    
    try:
        # Run with streamable_http transport - enables async HTTP communication
        await mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"[Async-MCP-Weather] Error starting server: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
        