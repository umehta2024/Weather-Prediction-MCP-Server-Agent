"""Weather data broker using Open-Meteo public API.

No authentication required - Open-Meteo is a free public API.
Perfect for weather forecasting and climate analysis.
"""
import requests
from datetime import datetime
from typing import Optional

BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Common city coordinates for easy access
CITY_COORDS = {
    "berlin": (52.52, 13.41),
    "new york": (40.71, -74.01),
    "london": (51.51, -0.13),
    "tokyo": (35.68, 139.65),
    "paris": (48.85, 2.35),
    "sydney": (-33.87, 151.21),
    "mumbai": (19.08, 72.88),
    "singapore": (1.35, 103.82),
    "dubai": (25.20, 55.27),
    "toronto": (43.65, -79.38),
}


def get_weather_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7,
) -> dict:
    """
    Get weather forecast for a location from Open-Meteo.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        forecast_days: Number of days to forecast (1-16, default 7)
    
    Returns:
        Dict with current weather and daily forecast data
    """
    try:
        response = requests.get(
            BASE_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_hours,wind_speed_10m_max",
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,rain",
                "timezone": "auto",
                "forecast_days": min(forecast_days, 16),  # API max is 16 days
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "status": "success",
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "elevation": data.get("elevation"),
            "current": data.get("current", {}),
            "daily": data.get("daily", {}),
            "retrieved_at": datetime.utcnow().isoformat(),
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": f"Failed to fetch weather: {str(e)}",
        }


def get_weather_by_city(city_name: str, forecast_days: int = 7) -> dict:
    """
    Get weather forecast by city name.
    
    Args:
        city_name: Name of city (e.g., "Berlin", "New York")
        forecast_days: Number of days to forecast (1-16, default 7)
    
    Returns:
        Dict with weather data or error message
    """
    coords = CITY_COORDS.get(city_name.lower())
    if not coords:
        return {
            "status": "error",
            "message": f"City '{city_name}' not found. Available cities: {', '.join(CITY_COORDS.keys())}",
        }
    
    result = get_weather_forecast(coords[0], coords[1], forecast_days)
    if result.get("status") == "success":
        result["city"] = city_name
    return result


def interpret_weather_code(code: int) -> str:
    """
    Convert WMO weather code to human-readable description.
    
    Args:
        code: WMO weather code (0-99)
    
    Returns:
        Human-readable weather description
    """
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return weather_codes.get(code, f"Unknown weather code: {code}")