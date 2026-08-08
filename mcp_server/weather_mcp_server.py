"""Weather Forecasting MCP server.

Exposes weather forecasting tools over MCP (Model Context Protocol) so a
Databricks Agent Bricks agent can call them like any other tool:
    - get_current_weather(latitude, longitude, city_name) - Current weather conditions (by coordinates or city)
    - get_forecast(latitude, longitude, city_name, forecast_days) - Current + forecast (by coordinates or city)
    - optimal_running_weather(latitude, longitude, city_name) - Check if conditions are ideal for running
    - interpret_weather_code(code) - Convert WMO weather codes to descriptions
    - vector_search(query, limit, search_type) - Semantic search over weather data

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

# Lakebase Postgres configuration (optional, enables vector search + weather logging)
# To enable Lakebase features, set these environment variables in your app.yaml:
#
# environment:
#   LAKEBASE_HOST: ep-noisy-king-d8v8sm9z.database.us-east-2.cloud.databricks.com
#   LAKEBASE_DATABASE: databricks-postgres
#   LAKEBASE_USER: <your-databricks-email>
#   LAKEBASE_PASSWORD: <oauth-token-or-native-password>
#   WEATHER_EMBEDDINGS_TABLE: weather_embeddings
#   EMBEDDING_MODEL: all-MiniLM-L6-v2
#
# When configured, the server will:
#   1. Track locations in the locations table (auto-created, tracks request counts)
#   2. Store current weather in the current_weather table (auto-created, linked to locations)
#   3. Store forecasts in the weather_forecasts table (auto-created, daily forecast data)
#   4. Enable vector_search tool for semantic search over weather_embeddings table
#
# For Databricks Apps with Lakebase integration, credentials can be auto-injected.
# See: https://docs.databricks.com/aws/en/oltp/projects/authentication

LAKEBASE_HOST = os.getenv("LAKEBASE_HOST")
LAKEBASE_DATABASE = os.getenv("LAKEBASE_DATABASE")
LAKEBASE_USER = os.getenv("LAKEBASE_USER")
LAKEBASE_PASSWORD = os.getenv("LAKEBASE_PASSWORD")
WEATHER_EMBEDDINGS_TABLE = os.getenv("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Lazy-load dependencies for vector search
_embedding_model = None
_lakebase_connection = None

def get_embedding_model():
    """Lazy-load sentence transformers model for embeddings."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        except ImportError:
            raise ImportError(
                "sentence-transformers required for vector search. "
                "Install with: %pip install sentence-transformers"
            )
    return _embedding_model

def get_lakebase_connection():
    """Lazy-load Lakebase Postgres connection.
    
    For Databricks Apps with Lakebase integration, credentials are automatically
    injected via environment variables. For local development, you can either:
    1. Set LAKEBASE_PASSWORD to a static native Postgres password, or
    2. Use the Databricks SDK to generate OAuth tokens (see docs)
    
    Connection uses SSL/TLS (required by Lakebase).
    """
    global _lakebase_connection
    if _lakebase_connection is None:
        try:
            import psycopg2
            if not all([LAKEBASE_HOST, LAKEBASE_DATABASE, LAKEBASE_USER, LAKEBASE_PASSWORD]):
                raise ValueError(
                    "Lakebase credentials not configured. Set environment variables: \n"
                    "  LAKEBASE_HOST=ep-noisy-king-d8v8sm9z.database.us-east-2.cloud.databricks.com\n"
                    "  LAKEBASE_DATABASE=databricks-postgres\n"
                    "  LAKEBASE_USER=<your-databricks-email>\n"
                    "  LAKEBASE_PASSWORD=<oauth-token-or-native-password>\n"
                    "For Databricks Apps, these are auto-injected via app.yaml Lakebase config."
                )
            _lakebase_connection = psycopg2.connect(
                host=LAKEBASE_HOST,
                database=LAKEBASE_DATABASE,
                user=LAKEBASE_USER,
                password=LAKEBASE_PASSWORD,
                sslmode="require",  # Required by Lakebase
                connect_timeout=10
            )
            logger.info(f"Connected to Lakebase at {LAKEBASE_HOST}")
        except ImportError:
            raise ImportError(
                "psycopg2 required for Lakebase. "
                "Install with: pip install psycopg2-binary"
            )
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to Lakebase: {e}")
            raise
    return _lakebase_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-mcp-server")

mcp = FastMCP("weather-forecasting")


def ensure_locations_table():
    """Create the locations table if it doesn't exist."""
    try:
        conn = get_lakebase_connection()
        cursor = conn.cursor()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS locations (
            id SERIAL PRIMARY KEY,
            latitude FLOAT NOT NULL,
            longitude FLOAT NOT NULL,
            location_name VARCHAR(255),
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            request_count INTEGER DEFAULT 1,
            UNIQUE(latitude, longitude)
        );
        """
        cursor.execute(create_table_sql)
        
        # Create index on coordinates for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_locations_coords 
            ON locations(latitude, longitude);
        """)
        
        conn.commit()
        cursor.close()
        logger.debug("locations table verified")
        return True
    except Exception as e:
        logger.warning(f"Could not create locations table: {e}")
        return False


def ensure_current_weather_table():
    """Create the current_weather table if it doesn't exist."""
    try:
        conn = get_lakebase_connection()
        cursor = conn.cursor()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS current_weather (
            id SERIAL PRIMARY KEY,
            location_id INTEGER REFERENCES locations(id),
            latitude FLOAT NOT NULL,
            longitude FLOAT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            temperature FLOAT,
            weather_code INTEGER,
            windspeed FLOAT,
            winddirection FLOAT,
            humidity INTEGER,
            pressure FLOAT,
            cloudcover INTEGER,
            precipitation FLOAT,
            timezone VARCHAR(100),
            location_name VARCHAR(255),
            forecast_days INTEGER,
            raw_data JSONB
        );
        """
        cursor.execute(create_table_sql)
        
        # Create index on timestamp for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_current_weather_timestamp 
            ON current_weather(timestamp DESC);
        """)
        
        # Create index on location for spatial queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_current_weather_location 
            ON current_weather(latitude, longitude);
        """)
        
        # Create index on location_id for joins
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_current_weather_location_id 
            ON current_weather(location_id);
        """)
        
        conn.commit()
        cursor.close()
        logger.debug("current_weather table verified")
        return True
    except Exception as e:
        logger.warning(f"Could not create current_weather table: {e}")
        return False


def ensure_weather_forecasts_table():
    """Create the weather_forecasts table if it doesn't exist."""
    try:
        conn = get_lakebase_connection()
        cursor = conn.cursor()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS weather_forecasts (
            id SERIAL PRIMARY KEY,
            location_id INTEGER REFERENCES locations(id),
            forecast_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            temperature_min FLOAT,
            temperature_max FLOAT,
            precipitation FLOAT,
            precipitation_probability INTEGER,
            weather_code INTEGER,
            windspeed_max FLOAT,
            sunrise TIME,
            sunset TIME,
            raw_forecast_data JSONB
        );
        """
        cursor.execute(create_table_sql)
        
        # Create index on location_id and forecast_date for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_weather_forecasts_location_date 
            ON weather_forecasts(location_id, forecast_date);
        """)
        
        # Create index on created_at to track forecast accuracy over time
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_weather_forecasts_created_at 
            ON weather_forecasts(created_at DESC);
        """)
        
        conn.commit()
        cursor.close()
        logger.debug("weather_forecasts table verified")
        return True
    except Exception as e:
        logger.warning(f"Could not create weather_forecasts table: {e}")
        return False


def upsert_location(latitude: float, longitude: float, location_name: str = None) -> int:
    """Insert or update location, return location_id.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        location_name: Optional location name
    
    Returns:
        location_id of the inserted/updated location
    """
    conn = get_lakebase_connection()
    cursor = conn.cursor()
    
    # Try to insert, or update if exists (using ON CONFLICT)
    upsert_sql = """
    INSERT INTO locations (latitude, longitude, location_name, first_seen, last_seen, request_count)
    VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
    ON CONFLICT (latitude, longitude) 
    DO UPDATE SET 
        last_seen = CURRENT_TIMESTAMP,
        request_count = locations.request_count + 1,
        location_name = COALESCE(EXCLUDED.location_name, locations.location_name)
    RETURNING id;
    """
    
    cursor.execute(upsert_sql, (latitude, longitude, location_name))
    location_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    
    return location_id


def insert_forecast_data(location_id: int, weather_data: dict):
    """Insert daily forecast data into weather_forecasts table.
    
    Args:
        location_id: ID from locations table
        weather_data: Weather data dict from weather_broker
    """
    conn = get_lakebase_connection()
    cursor = conn.cursor()
    
    daily = weather_data.get("daily", {})
    dates = daily.get("time", [])
    
    if not dates:
        logger.debug("No forecast data to insert")
        return
    
    # Prepare batch insert
    insert_sql = """
    INSERT INTO weather_forecasts (
        location_id, forecast_date, temperature_min, temperature_max,
        precipitation, precipitation_probability, weather_code,
        windspeed_max, sunrise, sunset, raw_forecast_data
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """
    
    # Insert each day's forecast
    for i, date in enumerate(dates):
        forecast_day_data = {
            "date": date,
            "temperature_min": daily.get("temperature_2m_min", [])[i] if i < len(daily.get("temperature_2m_min", [])) else None,
            "temperature_max": daily.get("temperature_2m_max", [])[i] if i < len(daily.get("temperature_2m_max", [])) else None,
            "precipitation": daily.get("precipitation_sum", [])[i] if i < len(daily.get("precipitation_sum", [])) else None,
            "precipitation_probability": daily.get("precipitation_probability_max", [])[i] if i < len(daily.get("precipitation_probability_max", [])) else None,
            "weather_code": daily.get("weather_code", [])[i] if i < len(daily.get("weather_code", [])) else None,
            "windspeed_max": daily.get("windspeed_10m_max", [])[i] if i < len(daily.get("windspeed_10m_max", [])) else None,
            "sunrise": daily.get("sunrise", [])[i] if i < len(daily.get("sunrise", [])) else None,
            "sunset": daily.get("sunset", [])[i] if i < len(daily.get("sunset", [])) else None,
        }
        
        cursor.execute(insert_sql, (
            location_id,
            date,
            forecast_day_data["temperature_min"],
            forecast_day_data["temperature_max"],
            forecast_day_data["precipitation"],
            forecast_day_data["precipitation_probability"],
            forecast_day_data["weather_code"],
            forecast_day_data["windspeed_max"],
            forecast_day_data["sunrise"],
            forecast_day_data["sunset"],
            forecast_day_data  # Store daily data as JSONB
        ))
    
    conn.commit()
    cursor.close()
    logger.info(f"Inserted {len(dates)} forecast days for location_id {location_id}")


def insert_weather_data(latitude: float, longitude: float, weather_data: dict, location_name: str = None):
    """Insert weather data into locations, current_weather, and weather_forecasts tables.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        weather_data: Weather data dict from weather_broker
        location_name: Optional location name (city, coordinates, etc.)
    """
    try:
        # Ensure all tables exist
        if not ensure_locations_table():
            return
        if not ensure_current_weather_table():
            return
        if not ensure_weather_forecasts_table():
            return
        
        # Step 1: Insert/update location, get location_id
        location_id = upsert_location(latitude, longitude, location_name)
        logger.debug(f"Location ID: {location_id}")
        
        # Step 2: Insert current weather
        conn = get_lakebase_connection()
        cursor = conn.cursor()
        
        current = weather_data.get("current", {})
        
        insert_current_sql = """
        INSERT INTO current_weather (
            location_id, latitude, longitude, temperature, weather_code, windspeed, 
            winddirection, humidity, pressure, cloudcover, precipitation,
            timezone, location_name, forecast_days, raw_data
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        cursor.execute(insert_current_sql, (
            location_id,
            latitude,
            longitude,
            current.get("temperature"),
            current.get("weather_code"),
            current.get("windspeed"),
            current.get("winddirection"),
            current.get("humidity"),
            current.get("pressure"),
            current.get("cloudcover"),
            current.get("precipitation"),
            weather_data.get("timezone"),
            location_name or f"{latitude},{longitude}",
            len(weather_data.get("daily", {}).get("time", [])),
            weather_data  # Store complete response as JSONB
        ))
        
        conn.commit()
        cursor.close()
        logger.info(f"Inserted current weather for {location_name or f'{latitude},{longitude}'}")
        
        # Step 3: Insert forecast data
        insert_forecast_data(location_id, weather_data)
        
    except ValueError as e:
        # Lakebase not configured - skip silently
        logger.debug(f"Lakebase not configured, skipping insert: {e}")
    except Exception as e:
        logger.warning(f"Failed to insert weather data: {e}")
        # Don't raise - we don't want to fail the weather request if insert fails


@mcp.tool
def get_current_weather(
    latitude: float = None,
    longitude: float = None,
    city_name: str = None
) -> dict:
    """
    Get current weather conditions for a location from Open-Meteo.
    
    Returns only current weather (temperature, conditions, wind, etc.) without forecast data.
    Stores current conditions in Lakebase Postgres current_weather table if configured.
    
    Provide either coordinates (latitude + longitude) OR city_name, not both.
    
    Supported cities: Berlin, New York, London, Tokyo, Paris, Sydney, 
    Mumbai, Singapore, Dubai, Toronto
    
    Args:
        latitude: Latitude coordinate (e.g., 52.52 for Berlin). Required if city_name not provided.
        longitude: Longitude coordinate (e.g., 13.41 for Berlin). Required if city_name not provided.
        city_name: Name of city (e.g., "Berlin", "New York"). Required if coordinates not provided.
    
    Returns:
        Dict with current weather conditions only
    
    Examples:
        get_current_weather(latitude=52.52, longitude=13.41)
        get_current_weather(city_name="Berlin")
    """
    # Validate input: either coordinates or city, not both
    has_coords = latitude is not None and longitude is not None
    has_city = city_name is not None
    
    if not has_coords and not has_city:
        return {"error": "Must provide either (latitude, longitude) or city_name", "success": False}
    
    if has_coords and has_city:
        return {"error": "Provide either coordinates OR city_name, not both", "success": False}
    
    # Fetch weather data
    if has_city:
        weather_data = weather_broker.get_weather_by_city(city_name, forecast_days=1)
        latitude = weather_data.get("latitude")
        longitude = weather_data.get("longitude")
        location_label = city_name
    else:
        weather_data = weather_broker.get_weather_forecast(latitude, longitude, forecast_days=1)
        location_label = f"{latitude},{longitude}"
    
    # Insert into Lakebase if configured (only locations and current_weather, skip forecasts)
    if latitude and longitude:
        try:
            if all([LAKEBASE_HOST, LAKEBASE_DATABASE, LAKEBASE_USER, LAKEBASE_PASSWORD]):
                if ensure_locations_table() and ensure_current_weather_table():
                    location_id = upsert_location(latitude, longitude, city_name)
                    
                    conn = get_lakebase_connection()
                    cursor = conn.cursor()
                    current = weather_data.get("current", {})
                    
                    insert_current_sql = """
                    INSERT INTO current_weather (
                        location_id, latitude, longitude, temperature, weather_code, windspeed, 
                        winddirection, humidity, pressure, cloudcover, precipitation,
                        timezone, location_name, forecast_days, raw_data
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    
                    cursor.execute(insert_current_sql, (
                        location_id, latitude, longitude,
                        current.get("temperature"), current.get("weather_code"),
                        current.get("windspeed"), current.get("winddirection"),
                        current.get("humidity"), current.get("pressure"),
                        current.get("cloudcover"), current.get("precipitation"),
                        weather_data.get("timezone"), location_label,
                        0, current  # Store only current data
                    ))
                    
                    conn.commit()
                    cursor.close()
                    logger.info(f"Inserted current weather for {location_label}")
        except Exception as e:
            logger.debug(f"Skipping weather data insert: {e}")
    
    # Return only current weather portion
    result = {
        "latitude": weather_data.get("latitude"),
        "longitude": weather_data.get("longitude"),
        "timezone": weather_data.get("timezone"),
        "current": weather_data.get("current", {})
    }
    if city_name:
        result["location"] = city_name
    return result


@mcp.tool
def get_forecast(
    latitude: float = None,
    longitude: float = None,
    city_name: str = None,
    forecast_days: int = 7
) -> dict:
    """
    Get weather forecast for a location from Open-Meteo.
    
    Returns current conditions plus daily forecast for the specified number of days.
    Stores data in Lakebase Postgres (locations, current_weather, weather_forecasts) if configured.
    
    Provide either coordinates (latitude + longitude) OR city_name, not both.
    
    Supported cities: Berlin, New York, London, Tokyo, Paris, Sydney, 
    Mumbai, Singapore, Dubai, Toronto
    
    Args:
        latitude: Latitude coordinate (e.g., 52.52 for Berlin). Required if city_name not provided.
        longitude: Longitude coordinate (e.g., 13.41 for Berlin). Required if city_name not provided.
        city_name: Name of city (e.g., "Berlin", "New York"). Required if coordinates not provided.
        forecast_days: Number of days to forecast (1-16, default 7)
    
    Returns:
        Dict with current weather conditions and daily forecast
    
    Examples:
        get_forecast(latitude=52.52, longitude=13.41, forecast_days=7)
        get_forecast(city_name="Berlin", forecast_days=7)
    """
    # Validate input: either coordinates or city, not both
    has_coords = latitude is not None and longitude is not None
    has_city = city_name is not None
    
    if not has_coords and not has_city:
        return {"error": "Must provide either (latitude, longitude) or city_name", "success": False}
    
    if has_coords and has_city:
        return {"error": "Provide either coordinates OR city_name, not both", "success": False}
    
    # Fetch weather data
    if has_city:
        weather_data = weather_broker.get_weather_by_city(city_name, forecast_days)
        latitude = weather_data.get("latitude")
        longitude = weather_data.get("longitude")
    else:
        weather_data = weather_broker.get_weather_forecast(latitude, longitude, forecast_days)
    
    # Insert into all Lakebase tables if configured (non-blocking)
    if latitude and longitude:
        try:
            insert_weather_data(latitude, longitude, weather_data, location_name=city_name)
        except Exception as e:
            logger.debug(f"Skipping weather data insert: {e}")
    
    return weather_data


@mcp.tool
def optimal_running_weather(
    latitude: float = None,
    longitude: float = None,
    city_name: str = None
) -> dict:
    """
    Check if current weather conditions are optimal for running.
    
    Evaluates temperature and humidity against ideal running conditions:
    - Temperature: 40°F to 60°F (4°C to 15°C)
    - Humidity: 30% to 50%
    
    Provide either coordinates (latitude + longitude) OR city_name, not both.
    
    Supported cities: Berlin, New York, London, Tokyo, Paris, Sydney, 
    Mumbai, Singapore, Dubai, Toronto
    
    Args:
        latitude: Latitude coordinate (e.g., 52.52 for Berlin). Required if city_name not provided.
        longitude: Longitude coordinate (e.g., 13.41 for Berlin). Required if city_name not provided.
        city_name: Name of city (e.g., "Berlin", "New York"). Required if coordinates not provided.
    
    Returns:
        Dict with recommendation, current conditions, and assessment details
    
    Examples:
        optimal_running_weather(latitude=52.52, longitude=13.41)
        optimal_running_weather(city_name="Berlin")
    """
    # Validate input: either coordinates or city, not both
    has_coords = latitude is not None and longitude is not None
    has_city = city_name is not None
    
    if not has_coords and not has_city:
        return {"error": "Must provide either (latitude, longitude) or city_name", "success": False}
    
    if has_coords and has_city:
        return {"error": "Provide either coordinates OR city_name, not both", "success": False}
    
    # Fetch current weather data
    if has_city:
        weather_data = weather_broker.get_weather_by_city(city_name, forecast_days=1)
        location_label = city_name
    else:
        weather_data = weather_broker.get_weather_forecast(latitude, longitude, forecast_days=1)
        location_label = f"{latitude},{longitude}"
    
    current = weather_data.get("current", {})
    
    # Extract current conditions
    temp_celsius = current.get("temperature")
    humidity = current.get("humidity")
    
    # Check if we have the required data
    if temp_celsius is None or humidity is None:
        return {
            "error": "Unable to retrieve temperature or humidity data",
            "success": False,
            "location": location_label
        }
    
    # Define ideal ranges
    TEMP_MIN_C = 4.0
    TEMP_MAX_C = 15.0
    TEMP_MIN_F = 40.0
    TEMP_MAX_F = 60.0
    HUMIDITY_MIN = 30
    HUMIDITY_MAX = 50
    
    # Convert to Fahrenheit for display
    temp_fahrenheit = (temp_celsius * 9/5) + 32
    
    # Check conditions
    temp_optimal = TEMP_MIN_C <= temp_celsius <= TEMP_MAX_C
    humidity_optimal = HUMIDITY_MIN <= humidity <= HUMIDITY_MAX
    
    # Build assessment
    conditions_met = []
    conditions_failed = []
    
    if temp_optimal:
        conditions_met.append(f"Temperature is ideal: {temp_celsius:.1f}°C ({temp_fahrenheit:.1f}°F)")
    else:
        if temp_celsius < TEMP_MIN_C:
            conditions_failed.append(f"Temperature too cold: {temp_celsius:.1f}°C ({temp_fahrenheit:.1f}°F) - ideal is {TEMP_MIN_C}°C to {TEMP_MAX_C}°C ({TEMP_MIN_F}°F to {TEMP_MAX_F}°F)")
        else:
            conditions_failed.append(f"Temperature too warm: {temp_celsius:.1f}°C ({temp_fahrenheit:.1f}°F) - ideal is {TEMP_MIN_C}°C to {TEMP_MAX_C}°C ({TEMP_MIN_F}°F to {TEMP_MAX_F}°F)")
    
    if humidity_optimal:
        conditions_met.append(f"Humidity is ideal: {humidity}%")
    else:
        if humidity < HUMIDITY_MIN:
            conditions_failed.append(f"Humidity too low: {humidity}% - ideal is {HUMIDITY_MIN}% to {HUMIDITY_MAX}%")
        else:
            conditions_failed.append(f"Humidity too high: {humidity}% - ideal is {HUMIDITY_MIN}% to {HUMIDITY_MAX}%")
    
    # Overall recommendation
    is_optimal = temp_optimal and humidity_optimal
    
    if is_optimal:
        recommendation = "Perfect conditions for running!"
        status = "optimal"
    elif temp_optimal or humidity_optimal:
        recommendation = "Conditions are acceptable but not ideal. Consider running if you're comfortable with these conditions."
        status = "acceptable"
    else:
        recommendation = "Conditions are not ideal for running. Consider waiting for better weather."
        status = "not_recommended"
    
    return {
        "success": True,
        "location": location_label,
        "status": status,
        "recommendation": recommendation,
        "is_optimal": is_optimal,
        "current_conditions": {
            "temperature_celsius": temp_celsius,
            "temperature_fahrenheit": round(temp_fahrenheit, 1),
            "humidity_percent": humidity,
            "weather_code": current.get("weather_code"),
            "weather_description": weather_broker.interpret_weather_code(current.get("weather_code")) if current.get("weather_code") is not None else None
        },
        "ideal_ranges": {
            "temperature_celsius": f"{TEMP_MIN_C}°C to {TEMP_MAX_C}°C",
            "temperature_fahrenheit": f"{TEMP_MIN_F}°F to {TEMP_MAX_F}°F",
            "humidity_percent": f"{HUMIDITY_MIN}% to {HUMIDITY_MAX}%"
        },
        "assessment": {
            "conditions_met": conditions_met,
            "conditions_failed": conditions_failed
        }
    }


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


@mcp.tool
def vector_search(
    query: str,
    limit: int = 10,
    search_type: str = "weather_events"
) -> dict:
    """
    Semantic search over historical weather data using vector embeddings.
    
    Search through historical weather events, climate patterns, and weather
    anomalies using natural language queries. Requires a Lakebase Postgres
    database with pgvector extension and pre-computed weather embeddings.
    
    Example queries:
        - "severe thunderstorms in midwest"
        - "heat waves during summer months"
        - "hurricane patterns in Atlantic"
        - "winter storm impacts on infrastructure"
    
    Args:
        query: Natural language search query describing weather patterns/events
        limit: Maximum number of results to return (1-50, default 10)
        search_type: Type of weather data to search:
            - "weather_events": Significant weather events and patterns
            - "climate_data": Historical climate observations
            - "forecasts": Past forecast accuracy and patterns
    
    Returns:
        Dict with query, results, similarity scores, and metadata
    
    Database Schema Requirements:
        The Lakebase database should have a table with:
        - id: Unique identifier
        - event_type: Type of weather event (storm, heat_wave, etc.)
        - location: Geographic location
        - date: Event date
        - description: Text description of the event
        - embedding: pgvector embedding (384 dimensions for all-MiniLM-L6-v2)
        - severity: Optional severity rating
        - metadata: Optional JSON with additional details
    """
    if not query or not query.strip():
        return {"error": "Query text is required", "success": False}
    
    if limit < 1 or limit > 50:
        return {"error": "Limit must be between 1 and 50", "success": False}
    
    valid_search_types = ["weather_events", "climate_data", "forecasts"]
    if search_type not in valid_search_types:
        return {
            "error": f"Invalid search_type. Must be one of: {valid_search_types}",
            "success": False
        }
    
    try:
        # Compute embedding for the query
        model = get_embedding_model()
        query_embedding = model.encode(query)
        embedding_list = query_embedding.tolist()
        
        # Get database connection
        conn = get_lakebase_connection()
        cursor = conn.cursor()
        
        # Build query based on search type
        table_name = WEATHER_EMBEDDINGS_TABLE
        
        # Vector similarity search using pgvector's <=> operator (cosine distance)
        query_sql = f"""
            SELECT 
                id,
                event_type,
                location,
                date,
                description,
                severity,
                metadata,
                1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            WHERE search_type = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        
        # Execute query
        cursor.execute(
            query_sql,
            (str(embedding_list), search_type, str(embedding_list), limit)
        )
        
        # Fetch results
        columns = [
            "id", "event_type", "location", "date", 
            "description", "severity", "metadata", "similarity"
        ]
        results = []
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            # Convert date to string if needed
            if result["date"]:
                result["date"] = str(result["date"])
            results.append(result)
        
        cursor.close()
        
        return {
            "success": True,
            "query": query,
            "search_type": search_type,
            "results": results,
            "count": len(results),
            "model": EMBEDDING_MODEL,
            "note": "Results ranked by semantic similarity to query"
        }
        
    except ImportError as e:
        return {
            "error": str(e),
            "success": False,
            "note": "Vector search requires additional dependencies"
        }
    except ValueError as e:
        return {
            "error": str(e),
            "success": False,
            "note": "Check Lakebase configuration environment variables"
        }
    except Exception as e:
        logger.exception("Vector search failed")
        return {
            "error": f"Search failed: {str(e)}",
            "success": False
        }


if __name__ == "__main__":
    # Databricks Apps route external HTTP traffic to this port via app.yaml;
    # streamable-http is the transport Databricks' MCP client/gateway expects
    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", 8000)))
    mcp.run(transport="http", host="0.0.0.0", port=port)