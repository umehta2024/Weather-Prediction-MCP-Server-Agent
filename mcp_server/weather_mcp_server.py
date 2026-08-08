"""Weather Forecasting MCP server.

Exposes weather forecasting tools over MCP (Model Context Protocol) so a
Databricks Agent Bricks agent can call them like any other tool:
    - get_weather(latitude, longitude, forecast_days)
    - get_weather_by_city(city_name, forecast_days)
    - interpret_weather_code(code)

These tools are backed by Open-Meteo's free public API (see weather_broker.py),
so students can safely retrieve weather data without authentication or API keys.

Deploy this as its own Databricks App (same app.yaml + FastMCP entrypoint
pattern documented at
https://docs.databricks.com/aws/en/agents/mcp-tools/custom-mcp), separate
from the dashboard app, so an Agent Bricks agent (or any MCP client) can
register its URL as an external MCP server.

Run locally:
    python weather_mcp_server.py
"""

import os
import logging

from fastmcp import FastMCP
import weather_broker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-mcp-server")

mcp = FastMCP("weather-forecasting")


@mcp.tool
def get_weather(latitude: float, longitude: float, forecast_days: int = 7) -> dict:
    """
    Get weather forecast for a location using coordinates from Open-Meteo.
    
    Args:
        latitude: Latitude coordinate (e.g., 52.52 for Berlin)
        longitude: Longitude coordinate (e.g., 13.41 for Berlin)
        forecast_days: Number of days to forecast (1-16, default 7)
    
    Returns:
        Dict with current weather conditions and daily forecast
    """
    return weather_broker.get_weather_forecast(latitude, longitude, forecast_days)


@mcp.tool
def get_weather_by_city(city_name: str, forecast_days: int = 7) -> dict:
    """
    Get weather forecast by city name.
    
    Supported cities: Berlin, New York, London, Tokyo, Paris, Sydney, 
    Mumbai, Singapore, Dubai, Toronto
    
    Args:
        city_name: Name of city (e.g., "Berlin", "New York")
        forecast_days: Number of days to forecast (1-16, default 7)
    
    Returns:
        Dict with weather data
    """
    return weather_broker.get_weather_by_city(city_name, forecast_days)


@mcp.tool
def interpret_weather_code(code: int) -> dict:
    """
    Convert WMO weather code to human-readable description.
    
    Args:
        code: WMO weather code (0-99)
    
    Returns:
        Dict with code and description
    """
    description = weather_broker.interpret_weather_code(code)
    return {
        "code": code,
        "description": description,
    }


if __name__ == "__main__":
    # Databricks Apps route external HTTP traffic to this port via app.yaml;
    # streamable-http is the transport Databricks' MCP client/gateway expects
    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", 8000)))
    mcp.run(transport="http", host="0.0.0.0", port=port)