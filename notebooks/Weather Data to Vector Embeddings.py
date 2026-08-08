# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # Weather Data -> Delta Tables (with optional Postgres sync)
# MAGIC
# MAGIC This notebook ingests weather data from the Open-Meteo API and generates
# MAGIC vector embeddings for semantic search.
# MAGIC
# MAGIC ## Architecture:
# MAGIC 1. **Fetch** weather from Open-Meteo API for tracked locations
# MAGIC 2. **Write to Delta Tables** in Unity Catalog:
# MAGIC    - `main.default.weather_locations` - tracked cities
# MAGIC    - `main.default.weather_current` - current observations
# MAGIC    - `main.default.weather_forecasts` - 7-day forecasts
# MAGIC    - `main.default.weather_embeddings` - vector embeddings (ARRAY<DOUBLE>)
# MAGIC 3. **Optional**: Sync Delta tables to Lakebase Postgres using Foreign Catalog
# MAGIC
# MAGIC Delta is the single source of truth - fast, queryable, and can be synced to
# MAGIC Postgres for external tools (like your MCP server).

# COMMAND ----------

# DBTITLE 1,Install all required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers requests pandas psycopg2-binary

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let you override the source/destination table names and the
# MAGIC embedding model without editing the notebook - useful when running this
# MAGIC as a scheduled Databricks Job.

# COMMAND ----------

# DBTITLE 1,Configuration - Delta Tables
dbutils.widgets.text("catalog", "main", "Unity Catalog catalog name")
dbutils.widgets.text("schema", "default", "Unity Catalog schema name")
dbutils.widgets.text("locations_table", "weather_locations", "Locations Delta table name")
dbutils.widgets.text("current_weather_table", "weather_current", "Current weather Delta table name")
dbutils.widgets.text("forecasts_table", "weather_forecasts", "Forecasts Delta table name")
dbutils.widgets.text("embeddings_table", "weather_embeddings", "Embeddings Delta table name")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("openmeteo_api_base_url", "https://api.open-meteo.com/v1", "Open-Meteo API base URL")
dbutils.widgets.text("forecast_days", "7", "Number of forecast days to fetch")
dbutils.widgets.text("max_requests_per_minute", "60", "Open-Meteo API rate limit (generous free tier)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
LOCATIONS_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('locations_table')}"
CURRENT_WEATHER_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('current_weather_table')}"
FORECASTS_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('forecasts_table')}"
EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('embeddings_table')}"
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
OPENMETEO_API_BASE_URL = dbutils.widgets.get("openmeteo_api_base_url")
FORECAST_DAYS = int(dbutils.widgets.get("forecast_days"))
MAX_REQUESTS_PER_MINUTE = int(dbutils.widgets.get("max_requests_per_minute"))

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly. Rather than hardcoding
# one dimension, switch on the model name so swapping EMBEDDING_MODEL_NAME via
# the widget above automatically resizes the destination table's vector column.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024
    case "text-embedding-3-small":
        EMBEDDING_DIM = 1536
    case "text-embedding-3-large":
        EMBEDDING_DIM = 3072
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")
print(f"\nDelta tables:")
print(f"  Locations: {LOCATIONS_TABLE}")
print(f"  Current weather: {CURRENT_WEATHER_TABLE}")
print(f"  Forecasts: {FORECASTS_TABLE}")
print(f"  Embeddings: {EMBEDDINGS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Delta Tables
# MAGIC
# MAGIC Define schemas and create Delta tables if they don't already exist.
# MAGIC Using Unity Catalog managed tables for automatic lifecycle management.

# COMMAND ----------

# DBTITLE 1,Parse Lakebase Connection Info
# Create catalog and schema if they don't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# Create locations table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {LOCATIONS_TABLE} (
        location_id STRING NOT NULL,
        city_name STRING NOT NULL,
        country STRING,
        latitude DOUBLE NOT NULL,
        longitude DOUBLE NOT NULL,
        timezone STRING,
        created_at TIMESTAMP
    )
    USING DELTA
    COMMENT 'Tracked weather locations'
""")

# Create current weather table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CURRENT_WEATHER_TABLE} (
        observation_id STRING NOT NULL,
        location_id STRING NOT NULL,
        observed_at TIMESTAMP NOT NULL,
        temperature DOUBLE,
        humidity INT,
        wind_speed DOUBLE,
        precipitation DOUBLE,
        rain DOUBLE,
        weather_code INT,
        created_at TIMESTAMP
    )
    USING DELTA
    COMMENT 'Current weather observations'
""")

# Create weather forecasts table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FORECASTS_TABLE} (
        forecast_id STRING NOT NULL,
        location_id STRING NOT NULL,
        forecast_date DATE NOT NULL,
        temperature_max DOUBLE,
        temperature_min DOUBLE,
        precipitation_sum DOUBLE,
        precipitation_hours INT,
        wind_speed_max DOUBLE,
        weather_code INT,
        created_at TIMESTAMP
    )
    USING DELTA
    COMMENT '7-day weather forecasts'
""")

# Create embeddings table with ARRAY<DOUBLE> for vectors
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} (
        id STRING NOT NULL,
        event_type STRING,
        location STRING,
        date TIMESTAMP,
        description STRING,
        embedding ARRAY<DOUBLE>,
        severity STRING,
        metadata STRING,
        search_type STRING,
        created_at TIMESTAMP
    )
    USING DELTA
    COMMENT 'Weather embeddings for semantic search'
""")

print("✅ Delta tables created/verified:")
print(f"  {LOCATIONS_TABLE}")
print(f"  {CURRENT_WEATHER_TABLE}")
print(f"  {FORECASTS_TABLE}")
print(f"  {EMBEDDINGS_TABLE}")

# COMMAND ----------

# DBTITLE 1,Test JDBC and check weather tables
# Verify Delta tables and show row counts
from pyspark.sql.utils import AnalysisException

print("✅ Delta Tables Status:\n")

tables = [
    ("Locations", LOCATIONS_TABLE),
    ("Current Weather", CURRENT_WEATHER_TABLE),
    ("Forecasts", FORECASTS_TABLE),
    ("Embeddings", EMBEDDINGS_TABLE)
]

for name, table_name in tables:
    try:
        count = spark.table(table_name).count()
        print(f"  {name:20} ({table_name}): {count:,} rows")
    except AnalysisException:
        print(f"  {name:20} ({table_name}): ❌ Table not found")

print("\nℹ️  Ready to ingest weather data from Open-Meteo API")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seed Initial Locations
# MAGIC
# MAGIC Before ingesting weather data, we need to populate the locations table with
# MAGIC cities to track. This cell creates a starter set of 10 global cities.

# COMMAND ----------

# DBTITLE 1,Seed Locations
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import uuid

# Check if locations table is empty
if spark.table(LOCATIONS_TABLE).count() == 0:
    print("Seeding initial locations...\n")
    
    # Define starter locations (10 global cities)
    locations_data = [
        {"location_id": str(uuid.uuid4()), "city_name": "New York", "country": "USA", "latitude": 40.7128, "longitude": -74.0060, "timezone": "America/New_York"},
        {"location_id": str(uuid.uuid4()), "city_name": "London", "country": "UK", "latitude": 51.5074, "longitude": -0.1278, "timezone": "Europe/London"},
        {"location_id": str(uuid.uuid4()), "city_name": "Tokyo", "country": "Japan", "latitude": 35.6762, "longitude": 139.6503, "timezone": "Asia/Tokyo"},
        {"location_id": str(uuid.uuid4()), "city_name": "Sydney", "country": "Australia", "latitude": -33.8688, "longitude": 151.2093, "timezone": "Australia/Sydney"},
        {"location_id": str(uuid.uuid4()), "city_name": "Mumbai", "country": "India", "latitude": 19.0760, "longitude": 72.8777, "timezone": "Asia/Kolkata"},
        {"location_id": str(uuid.uuid4()), "city_name": "Dubai", "country": "UAE", "latitude": 25.2048, "longitude": 55.2708, "timezone": "Asia/Dubai"},
        {"location_id": str(uuid.uuid4()), "city_name": "São Paulo", "country": "Brazil", "latitude": -23.5505, "longitude": -46.6333, "timezone": "America/Sao_Paulo"},
        {"location_id": str(uuid.uuid4()), "city_name": "Toronto", "country": "Canada", "latitude": 43.6532, "longitude": -79.3832, "timezone": "America/Toronto"},
        {"location_id": str(uuid.uuid4()), "city_name": "Singapore", "country": "Singapore", "latitude": 1.3521, "longitude": 103.8198, "timezone": "Asia/Singapore"},
        {"location_id": str(uuid.uuid4()), "city_name": "Berlin", "country": "Germany", "latitude": 52.5200, "longitude": 13.4050, "timezone": "Europe/Berlin"},
    ]
    
    # Create DataFrame and write to Delta
    locations_df = spark.createDataFrame(locations_data)
    locations_df.write.format("delta").mode("append").saveAsTable(LOCATIONS_TABLE)
    
    print(f"✅ Seeded {len(locations_data)} locations")
    display(spark.table(LOCATIONS_TABLE).select("city_name", "country", "latitude", "longitude"))
else:
    count = spark.table(LOCATIONS_TABLE).count()
    print(f"ℹ️  Locations table already has {count} entries")
    display(spark.table(LOCATIONS_TABLE).select("city_name", "country", "latitude", "longitude"))

# COMMAND ----------

# DBTITLE 1,Add US Cities to Locations
# MAGIC %sql
# MAGIC INSERT INTO main.default.weather_locations 
# MAGIC (location_id, city_name, country, latitude, longitude, timezone, created_at)
# MAGIC SELECT 
# MAGIC   uuid() as location_id,
# MAGIC   city_name,
# MAGIC   country,
# MAGIC   latitude,
# MAGIC   longitude,
# MAGIC   timezone,
# MAGIC   current_timestamp() as created_at
# MAGIC FROM VALUES
# MAGIC   -- West Coast
# MAGIC   ('Los Angeles', 'USA', 34.0522, -118.2437, 'America/Los_Angeles'),
# MAGIC   ('San Francisco', 'USA', 37.7749, -122.4194, 'America/Los_Angeles'),
# MAGIC   ('Seattle', 'USA', 47.6062, -122.3321, 'America/Los_Angeles'),
# MAGIC   ('Portland', 'USA', 45.5152, -122.6784, 'America/Los_Angeles'),
# MAGIC   ('San Diego', 'USA', 32.7157, -117.1611, 'America/Los_Angeles'),
# MAGIC   
# MAGIC   -- Southwest
# MAGIC   ('Phoenix', 'USA', 33.4484, -112.0740, 'America/Phoenix'),
# MAGIC   ('Las Vegas', 'USA', 36.1699, -115.1398, 'America/Los_Angeles'),
# MAGIC   ('Denver', 'USA', 39.7392, -104.9903, 'America/Denver'),
# MAGIC   ('Albuquerque', 'USA', 35.0844, -106.6504, 'America/Denver'),
# MAGIC   
# MAGIC   -- Texas
# MAGIC   ('Houston', 'USA', 29.7604, -95.3698, 'America/Chicago'),
# MAGIC   ('Dallas', 'USA', 32.7767, -96.7970, 'America/Chicago'),
# MAGIC   ('Austin', 'USA', 30.2672, -97.7431, 'America/Chicago'),
# MAGIC   ('San Antonio', 'USA', 29.4241, -98.4936, 'America/Chicago'),
# MAGIC   
# MAGIC   -- Midwest
# MAGIC   ('Chicago', 'USA', 41.8781, -87.6298, 'America/Chicago'),
# MAGIC   ('Minneapolis', 'USA', 44.9778, -93.2650, 'America/Chicago'),
# MAGIC   ('Detroit', 'USA', 42.3314, -83.0458, 'America/Detroit'),
# MAGIC   ('St. Louis', 'USA', 38.6270, -90.1994, 'America/Chicago'),
# MAGIC   
# MAGIC   -- East Coast
# MAGIC   ('Boston', 'USA', 42.3601, -71.0589, 'America/New_York'),
# MAGIC   ('Philadelphia', 'USA', 39.9526, -75.1652, 'America/New_York'),
# MAGIC   ('Washington DC', 'USA', 38.9072, -77.0369, 'America/New_York'),
# MAGIC   ('Atlanta', 'USA', 33.7490, -84.3880, 'America/New_York'),
# MAGIC   ('Miami', 'USA', 25.7617, -80.1918, 'America/New_York'),
# MAGIC   ('Charlotte', 'USA', 35.2271, -80.8431, 'America/New_York'),
# MAGIC   
# MAGIC   -- Other regions
# MAGIC   ('New Orleans', 'USA', 29.9511, -90.0715, 'America/Chicago'),
# MAGIC   ('Nashville', 'USA', 36.1627, -86.7816, 'America/Chicago'),
# MAGIC   ('Salt Lake City', 'USA', 40.7608, -111.8910, 'America/Denver')
# MAGIC AS t(city_name, country, latitude, longitude, timezone)

# COMMAND ----------

# DBTITLE 1,Verify US Cities
# MAGIC %sql
# MAGIC SELECT city_name, country, latitude, longitude, timezone 
# MAGIC FROM main.default.weather_locations 
# MAGIC WHERE country = 'USA'
# MAGIC ORDER BY city_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch weather data from Open-Meteo for tracked locations
# MAGIC
# MAGIC This ETL queries the `locations` table in Lakebase to find geographic
# MAGIC locations being tracked, then fetches both current weather conditions and
# MAGIC forecasts from the Open-Meteo API for each location.
# MAGIC
# MAGIC The Open-Meteo API is free and generous (no API key required), but requests
# MAGIC are still rate-limited via `MAX_REQUESTS_PER_MINUTE` (default 60/min) to be
# MAGIC respectful of the free service.

# COMMAND ----------

# DBTITLE 1,Fetch weather data from Open-Meteo
import time
from datetime import datetime, timezone
import requests
import uuid

def get_tracked_locations() -> list[dict]:
    """Get all tracked locations from the Delta table."""
    locations_df = spark.table(LOCATIONS_TABLE).toPandas()
    return locations_df.to_dict('records')

def fetch_weather_for_location(session: requests.Session, lat: float, lon: float) -> dict:
    """Fetch weather data from Open-Meteo API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,rain,weather_code",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_hours,wind_speed_10m_max",
        "forecast_days": FORECAST_DAYS,
        "timezone": "UTC"
    }
    resp = session.get(f"{OPENMETEO_API_BASE_URL}/forecast", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

print(f"Fetching weather data for tracked locations...\n")

locations = get_tracked_locations()
print(f"Found {len(locations)} tracked locations")

if not locations:
    print("No locations found. Add locations to the locations table first.")
    all_current_weather_rows = []
    all_forecast_rows = []
else:
    seconds_between_requests = 60.0 / MAX_REQUESTS_PER_MINUTE
    weather_session = requests.Session()
    all_current_weather_rows = []
    all_forecast_rows = []
    
    for i, location in enumerate(locations):
        if i > 0:
            time.sleep(seconds_between_requests)
        
        try:
            weather_data = fetch_weather_for_location(weather_session, location["latitude"], location["longitude"])
            
            # Process current weather
            current = weather_data.get("current", {})
            if current:
                current_row = {
                    "observation_id": str(uuid.uuid4()),
                    "location_id": location["location_id"],
                    "observed_at": datetime.now(timezone.utc),
                    "temperature": current.get("temperature_2m"),
                    "humidity": current.get("relative_humidity_2m"),
                    "wind_speed": current.get("wind_speed_10m"),
                    "precipitation": current.get("precipitation"),
                    "rain": current.get("rain"),
                    "weather_code": current.get("weather_code")
                }
                all_current_weather_rows.append(current_row)
            
            # Process daily forecasts
            daily = weather_data.get("daily", {})
            if daily and "time" in daily:
                for j, forecast_date in enumerate(daily["time"]):
                    forecast_row = {
                        "forecast_id": str(uuid.uuid4()),
                        "location_id": location["location_id"],
                        "forecast_date": forecast_date,
                        "temperature_max": daily.get("temperature_2m_max", [])[j] if j < len(daily.get("temperature_2m_max", [])) else None,
                        "temperature_min": daily.get("temperature_2m_min", [])[j] if j < len(daily.get("temperature_2m_min", [])) else None,
                        "precipitation_sum": daily.get("precipitation_sum", [])[j] if j < len(daily.get("precipitation_sum", [])) else None,
                        "precipitation_hours": daily.get("precipitation_hours", [])[j] if j < len(daily.get("precipitation_hours", [])) else None,
                        "wind_speed_max": daily.get("wind_speed_10m_max", [])[j] if j < len(daily.get("wind_speed_10m_max", [])) else None,
                        "weather_code": daily.get("weather_code", [])[j] if j < len(daily.get("weather_code", [])) else None,
                    }
                    all_forecast_rows.append(forecast_row)
            
            print(f"  ✅ {location['city_name']}: fetched current + {len(daily.get('time', []))} forecast days")
        except Exception as exc:
            print(f"  ❌ {location['city_name']}: failed to fetch weather ({exc})")
            continue
    
    print(f"\n✅ Collected {len(all_current_weather_rows)} current weather records")
    print(f"✅ Collected {len(all_forecast_rows)} forecast records")
    print(f"\nℹ️  Run the next cell to write them to Delta tables.")

# COMMAND ----------

# DBTITLE 1,Insert collected news articles using psycopg2
if not all_current_weather_rows and not all_forecast_rows:
    print("⚠️  No weather data to insert.")
else:
    # Write current weather to Delta
    if all_current_weather_rows:
        current_df = spark.createDataFrame(all_current_weather_rows)
        current_df.write.format("delta").mode("append").saveAsTable(CURRENT_WEATHER_TABLE)
        print(f"✅ Wrote {len(all_current_weather_rows)} current weather records to Delta")
    
    # Write forecasts to Delta
    if all_forecast_rows:
        forecast_df = spark.createDataFrame(all_forecast_rows)
        forecast_df.write.format("delta").mode("append").saveAsTable(FORECASTS_TABLE)
        print(f"✅ Wrote {len(all_forecast_rows)} forecast records to Delta")
    
    print(f"\n✅ Data successfully written to Delta tables!")
    print(f"\nℹ️  Next: Run the cells below to compute and store embeddings.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load weather data for embedding
# MAGIC
# MAGIC Reads weather data from `current_weather` and `weather_forecasts` tables
# MAGIC (just synced from Open-Meteo above) into a pandas DataFrame for embedding
# MAGIC computation.

# COMMAND ----------

# DBTITLE 1,Load weather data (JDBC alternative)
# Load weather data from Delta tables using Spark
from pyspark.sql import functions as F

# Read from Delta tables
current_weather_spark = spark.table(CURRENT_WEATHER_TABLE)
locations_spark = spark.table(LOCATIONS_TABLE)

# Join and prepare data using Spark SQL
weather_with_location = current_weather_spark.alias("cw") \
    .join(locations_spark.alias("l"), F.col("cw.location_id") == F.col("l.location_id")) \
    .select(
        F.col("cw.observation_id").alias("id"),
        F.col("l.city_name"),
        F.col("l.latitude"),
        F.col("l.longitude"),
        F.col("cw.observed_at").alias("timestamp"),
        F.col("cw.temperature"),
        F.col("cw.humidity"),
        F.col("cw.wind_speed"),
        F.col("cw.precipitation"),
        F.col("cw.weather_code"),
        # Create a natural language description for embedding
        F.concat_ws(
            ", ",
            F.concat(F.lit("Temperature "), F.col("cw.temperature"), F.lit("°C")),
            F.concat(F.lit("humidity "), F.col("cw.humidity"), F.lit("%")),
            F.concat(F.lit("wind "), F.col("cw.wind_speed"), F.lit(" km/h"))
        ).alias("embedding_text")
    ) \
    .filter(F.col("temperature").isNotNull()) \
    .orderBy(F.col("timestamp").desc())

# Check record count first
record_count = weather_with_location.count()
print(f"Found {record_count} weather records in database")

if record_count > 0:
    # Convert to pandas for embedding computation
    weather_df = weather_with_location.toPandas()
    print(f"Loaded {len(weather_df)} weather records using JDBC")
    display(weather_df.head(5))
else:
    print("\n⚠️  No weather data found. Run cells 11-12 to fetch and insert weather data first.")
    weather_df = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings
# MAGIC
# MAGIC Loads the sentence-transformers model once and applies it in batches
# MAGIC to the weather data descriptions.

# COMMAND ----------

# DBTITLE 1,Compute embeddings (distributed pandas UDF)
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Set up HuggingFace cache
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

# Compute embeddings in batches for memory efficiency
print("Computing embeddings...")
batch_size = 32
all_embeddings = []

for i in range(0, len(weather_df), batch_size):
    batch = weather_df.iloc[i:i+batch_size]
    vectors = model.encode(batch["embedding_text"].tolist(), show_progress_bar=False)
    all_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"  Processed {min(i + batch_size, len(weather_df))}/{len(weather_df)} weather records")

# Create embeddings DataFrame
embeddings_df = pd.DataFrame({
    "id": weather_df["id"],
    "city_name": weather_df["city_name"],
    "latitude": weather_df["latitude"],
    "longitude": weather_df["longitude"],
    "timestamp": weather_df["timestamp"].astype(str),
    "embedding_text": weather_df["embedding_text"],
    "embedding": all_embeddings,
})

print(f"Computed {len(embeddings_df)} embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# DBTITLE 1,Insert embeddings using psycopg2
from datetime import datetime
import uuid
import json
import pandas as pd

if len(embeddings_df) > 0:
    print(f"Writing {len(embeddings_df)} embeddings to Delta...")
    
    # Prepare embeddings data for Delta
    embeddings_records = []
    for _, row in embeddings_df.iterrows():
        # Build metadata dict with safe access to columns
        metadata = {
            "observation_id": str(row.get('id', '')),
            "latitude": float(row.get('latitude', 0.0)),
            "longitude": float(row.get('longitude', 0.0)),
            "model": EMBEDDING_MODEL_NAME
        }
        
        # Add optional weather fields if they exist
        if 'temperature' in row and row['temperature'] is not None:
            metadata['temperature'] = float(row['temperature'])
        if 'humidity' in row and row['humidity'] is not None:
            metadata['humidity'] = int(row['humidity'])
        if 'wind_speed' in row and row['wind_speed'] is not None:
            metadata['wind_speed'] = float(row['wind_speed'])
        
        # Convert timestamp to Python datetime
        timestamp_val = row.get('timestamp')
        if timestamp_val is not None:
            if isinstance(timestamp_val, pd.Timestamp):
                timestamp_val = timestamp_val.to_pydatetime()
            elif not isinstance(timestamp_val, datetime):
                # Try to parse as datetime if it's a string
                timestamp_val = pd.to_datetime(timestamp_val).to_pydatetime()
        
        record = {
            "id": str(uuid.uuid4()),
            "event_type": "weather_observation",
            "location": row.get('city_name', 'Unknown'),
            "date": timestamp_val,
            "description": row.get('embedding_text', ''),
            "embedding": row['embedding'].tolist() if hasattr(row['embedding'], 'tolist') else list(row['embedding']),
            "severity": None,
            "search_type": "current_weather",
            "metadata": json.dumps(metadata)
        }
        embeddings_records.append(record)
    
    # Create Spark DataFrame with explicit schema and write to Delta
    from pyspark.sql.types import StructType, StructField, StringType, TimestampType, ArrayType, DoubleType
    
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("event_type", StringType(), True),
        StructField("location", StringType(), True),
        StructField("date", TimestampType(), True),
        StructField("description", StringType(), True),
        StructField("embedding", ArrayType(DoubleType()), True),
        StructField("severity", StringType(), True),
        StructField("search_type", StringType(), True),
        StructField("metadata", StringType(), True)
    ])
    
    embeddings_spark_df = spark.createDataFrame(embeddings_records, schema=schema)
    embeddings_spark_df.write.format("delta").mode("append").saveAsTable(EMBEDDINGS_TABLE)
    
    print(f"✅ Successfully wrote {len(embeddings_df)} embeddings to {EMBEDDINGS_TABLE}")
    print(f"\n🔍 Embeddings are ready for semantic search!")
    print(f"   - Query using Spark SQL with array operations")
    print(f"   - Or sync to Postgres and use pgvector")
else:
    print("⚠️  No embeddings to write.")

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ **Weather data pipeline complete!**
# MAGIC
# MAGIC This notebook:
# MAGIC 1. ✅ Creates Delta tables in Unity Catalog (`main.default.*`)
# MAGIC 2. ✅ Seeds initial location data (10 global cities)
# MAGIC 3. ✅ Fetches weather from Open-Meteo API
# MAGIC 4. ✅ Writes current weather & forecasts to Delta tables
# MAGIC 5. ✅ Computes sentence embeddings using transformers
# MAGIC 6. ✅ Stores embeddings in Delta as `ARRAY<DOUBLE>`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Querying in Databricks
# MAGIC
# MAGIC ```sql
# MAGIC -- Query weather data
# MAGIC SELECT * FROM main.default.weather_current
# MAGIC ORDER BY observed_at DESC
# MAGIC LIMIT 10;
# MAGIC
# MAGIC -- View embeddings
# MAGIC SELECT id, location, description, 
# MAGIC        array_size(embedding) as vector_dim
# MAGIC FROM main.default.weather_embeddings;
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Syncing to Postgres (Optional)
# MAGIC
# MAGIC To sync these Delta tables to your Lakebase Postgres database:
# MAGIC
# MAGIC ### Option 1: Manual Sync (One-time)
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC # Your Lakebase connection details here
# MAGIC postgres_host = "your-endpoint.cloud.databricks.com"
# MAGIC postgres_db = "databricks_postgres"
# MAGIC
# MAGIC # Sync each table
# MAGIC for table in ['weather_locations', 'weather_current', 'weather_forecasts', 'weather_embeddings']:
# MAGIC     spark.table(f"main.default.{table}") \
# MAGIC         .write \
# MAGIC         .format("jdbc") \
# MAGIC         .option("url", f"jdbc:postgresql://{postgres_host}:5432/{postgres_db}") \
# MAGIC         .option("dbtable", table) \
# MAGIC         .option("user", "your_user") \
# MAGIC         .option("password", "your_password") \
# MAGIC         .mode("overwrite") \
# MAGIC         .save()
# MAGIC ```
# MAGIC
# MAGIC ### Option 2: Scheduled Sync Job
# MAGIC - Create a new notebook that reads from Delta and writes to Postgres
# MAGIC - Schedule it to run after this notebook completes
# MAGIC - Use Databricks Workflows to orchestrate both notebooks
# MAGIC
# MAGIC ### Option 3: Foreign Catalog (Recommended)
# MAGIC - Set up Lakebase as a Foreign Catalog in Unity Catalog
# MAGIC - Query both Delta and Postgres tables seamlessly
# MAGIC - Let Databricks handle the sync automatically
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Scheduling This Notebook
# MAGIC
# MAGIC Run this notebook on a schedule to keep weather data fresh:
# MAGIC
# MAGIC ```yaml
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     weather_ingestion:
# MAGIC       name: Weather Data Ingestion
# MAGIC       tasks:
# MAGIC         - task_key: ingest_weather
# MAGIC           notebook_task:
# MAGIC             notebook_path: ./Weather Data to Vector Embeddings
# MAGIC           new_cluster:
# MAGIC             spark_version: 14.3.x-scala2.12
# MAGIC             node_type_id: i3.xlarge
# MAGIC             num_workers: 2
# MAGIC       schedule:
# MAGIC         quartz_cron_expression: "0 0 */6 * * ?"  # Every 6 hours
# MAGIC         timezone_id: UTC
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Sync to Postgres
# MAGIC %md
# MAGIC ## Sync Delta Tables to Lakebase Postgres
# MAGIC
# MAGIC This section syncs the Delta tables to your Lakebase Postgres database,
# MAGIC converting the embeddings `ARRAY<DOUBLE>` to pgvector format.

# COMMAND ----------

# DBTITLE 1,Sync Delta to Postgres (Part 1)
import base64
from urllib.parse import urlparse
from databricks.sdk import WorkspaceClient
import psycopg2
from psycopg2.extras import execute_values

# Get Lakebase connection details from secret
w = WorkspaceClient()
secret = w.secrets.get_secret(scope="database", key="lakebase-url")
lakebase_url = base64.b64decode(secret.value).decode("utf-8")
parsed = urlparse(lakebase_url)

db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

jdbc_url = f"jdbc:postgresql://{db_host}:{db_port}/{db_name}?sslmode=require"
jdbc_props = {
    "user": db_user,
    "password": db_password,
    "driver": "org.postgresql.Driver"
}

print(f"🔄 Syncing Delta tables to Lakebase Postgres...")
print(f"   Host: {db_host}")
print(f"   Database: {db_name}\n")

# Sync simple tables via JDBC (locations, current_weather, forecasts)
simple_tables = [
    ('weather_locations', 'locations'),
    ('weather_current', 'current_weather'),
    ('weather_forecasts', 'weather_forecasts')
]

for delta_table, postgres_table in simple_tables:
    print(f"Syncing {delta_table} → {postgres_table}...")
    df = spark.table(f"{CATALOG}.{SCHEMA}.{delta_table}")
    
    # Write to Postgres via JDBC
    df.write \
        .jdbc(url=jdbc_url, table=postgres_table, mode="overwrite", properties=jdbc_props)
    
    row_count = df.count()
    print(f"  ✅ Synced {row_count} rows to {postgres_table}")

print("\n✅ Simple tables synced!")
print("\n🔄 Syncing embeddings table (requires pgvector conversion)...")

# COMMAND ----------

# DBTITLE 1,Sync Embeddings to Postgres (Part 2)
# Sync embeddings table with pgvector conversion
# JDBC doesn't support pgvector type directly, so we use psycopg2

print("Reading embeddings from Delta...")
embeddings_delta = spark.table(f"{CATALOG}.{SCHEMA}.weather_embeddings").toPandas()

if len(embeddings_delta) == 0:
    print("⚠️  No embeddings found in Delta table.")
else:
    print(f"Found {len(embeddings_delta)} embeddings to sync")
    
    # Connect to Postgres
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    
    try:
        cursor = conn.cursor()
        
        # Check if table has existing data
        cursor.execute("SELECT COUNT(*) FROM weather_embeddings")
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            print(f"  Found {existing_count} existing embeddings in Postgres")
            print("  Clearing them before sync...")
            cursor.execute("DELETE FROM weather_embeddings WHERE 1=1")
            print("  Cleared existing embeddings")
        
        # Prepare data for batch insert
        insert_data = []
        for _, row in embeddings_delta.iterrows():
            # Convert ARRAY<DOUBLE> to pgvector format: [x,y,z]
            embedding_list = row['embedding']
            vector_str = '[' + ','.join(str(float(x)) for x in embedding_list) + ']'
            
            # Build metadata JSON
            import json
            metadata_dict = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
            metadata_json = json.dumps(metadata_dict)
            
            insert_data.append((
                row['event_type'],
                row['location'],
                row['date'],
                row['description'],
                vector_str,  # Will be cast to vector in SQL
                row['severity'],
                row['search_type'],
                metadata_json
            ))
        
        # Batch insert with pgvector casting
        insert_sql = """
            INSERT INTO weather_embeddings (
                event_type, location, date, description, embedding,
                severity, search_type, metadata
            ) VALUES %s
        """
        
        # Template with ::vector cast for the embedding column
        template = "(%s, %s, %s, %s, %s::vector, %s, %s, %s::jsonb)"
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)
        
        conn.commit()
        print(f"  ✅ Synced {len(embeddings_delta)} embeddings to Postgres")
        print(f"\n🎉 All tables synced to Lakebase Postgres!")
        print(f"\n🔍 Ready for pgvector semantic search:")
        print(f"   SELECT * FROM weather_embeddings")
        print(f"   ORDER BY embedding <-> '[...]'::vector")
        print(f"   LIMIT 10;")
        
    finally:
        cursor.close()
        conn.close()