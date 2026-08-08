# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # Delta → Postgres Sync
# MAGIC
# MAGIC This notebook syncs weather data from Unity Catalog Delta tables to Lakebase Postgres.
# MAGIC
# MAGIC **Sync Strategy:**
# MAGIC - **Locations**: Full refresh (TRUNCATE + INSERT) - static reference data
# MAGIC - **Current Weather**: Keep last 7 days only - rolling window
# MAGIC - **Forecasts**: Keep last 30 days only - rolling window  
# MAGIC - **Embeddings**: Keep last 30 days only - rolling window
# MAGIC
# MAGIC **Schedule**: Run this notebook hourly to keep Postgres in sync with Delta

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# psycopg2 is already available in Serverless environment
%pip install -q 'databricks-sdk>=0.118.0'

# COMMAND ----------

# DBTITLE 1,Configuration
# Delta table configuration
CATALOG = "main"
SCHEMA = "default"
LOCATIONS_TABLE = f"{CATALOG}.{SCHEMA}.weather_locations"
CURRENT_WEATHER_TABLE = f"{CATALOG}.{SCHEMA}.weather_current"
FORECASTS_TABLE = f"{CATALOG}.{SCHEMA}.weather_forecasts"
EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.weather_embeddings"

# Lakebase Postgres configuration
LAKEBASE_PROJECT = "dataexpert-student"
LAKEBASE_BRANCH = "production"
LAKEBASE_DATABASE = "databricks-postgres"

# Retention windows
CURRENT_WEATHER_RETENTION_DAYS = 7
FORECAST_RETENTION_DAYS = 30
EMBEDDING_RETENTION_DAYS = 30

print(f"Delta Tables:")
print(f"  Locations: {LOCATIONS_TABLE}")
print(f"  Current: {CURRENT_WEATHER_TABLE}")
print(f"  Forecasts: {FORECASTS_TABLE}")
print(f"  Embeddings: {EMBEDDINGS_TABLE}")
print(f"\nLakebase Target: {LAKEBASE_PROJECT}/{LAKEBASE_BRANCH}/{LAKEBASE_DATABASE}")

# COMMAND ----------

# DBTITLE 1,Get Postgres Connection Info
import base64
from urllib.parse import urlparse
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Parse the lakebase-url secret (same pattern as the ticker news notebook)
def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")

lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract connection details directly from the secret URL
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"Connection details:")
print(f"  Host: {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")
print(f"  Using credentials from lakebase-url secret")

# Store connection properties for JDBC
jdbc_url = f"jdbc:postgresql://{db_host}:{db_port}/{db_name}?sslmode=require"
jdbc_properties = {
    "user": db_user,
    "password": db_password,
    "driver": "org.postgresql.Driver"
}

print(f"\n✅ JDBC connection configured")

# COMMAND ----------

# DBTITLE 1,Sync Locations & Build Mapping
import psycopg2
from datetime import datetime

print("Syncing locations and building ID mapping...")

# Read from Delta and deduplicate by coordinates (keep most recent)
from pyspark.sql.window import Window
from pyspark.sql import functions as F

window_spec = Window.partitionBy("latitude", "longitude").orderBy(F.desc("created_at"))
locations_df = spark.table(LOCATIONS_TABLE) \
    .withColumn("row_num", F.row_number().over(window_spec)) \
    .filter(F.col("row_num") == 1) \
    .drop("row_num") \
    .toPandas()

print(f"   Deduplicated to {len(locations_df)} unique locations")

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
    
    # Truncate all tables (CASCADE deletes dependent rows)
    cursor.execute("TRUNCATE TABLE public.locations CASCADE")
    
    # Build mapping: Delta UUID -> Postgres integer ID
    # Insert locations and capture new IDs
    delta_uuid_to_postgres_id = {}
    delta_uuid_to_city = {}
    
    for _, row in locations_df.iterrows():
        delta_uuid = row['location_id']
        city = row['city_name']
        
        # Insert and get new Postgres ID
        cursor.execute("""
            INSERT INTO public.locations (city_name, country, latitude, longitude, timezone, elevation, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING location_id
        """, (city, row['country'], row['latitude'], row['longitude'], 
              row.get('timezone'), None, row.get('created_at')))
        
        postgres_id = cursor.fetchone()[0]
        delta_uuid_to_postgres_id[delta_uuid] = postgres_id
        delta_uuid_to_city[delta_uuid] = city
    
    conn.commit()
    print(f"✅ Synced {len(locations_df)} locations")
    print(f"   Built mapping for {len(delta_uuid_to_postgres_id)} location IDs")
    
finally:
    cursor.close()
    conn.close()

# Store mapping globally for other cells
spark.conf.set("location_id_mapping", str(delta_uuid_to_postgres_id))

# COMMAND ----------

# DBTITLE 1,Sync Current Weather (Rolling Window)
import psycopg2
from datetime import timedelta
from pyspark.sql import functions as F
import ast

print(f"Syncing current_weather (last {CURRENT_WEATHER_RETENTION_DAYS} days)...")

# Get location ID mapping from previous cell
location_id_mapping = ast.literal_eval(spark.conf.get("location_id_mapping"))

# Calculate cutoff date
cutoff_date = datetime.now() - timedelta(days=CURRENT_WEATHER_RETENTION_DAYS)

# Read recent data from Delta
current_df = spark.table(CURRENT_WEATHER_TABLE) \
    .filter(F.col("observed_at") >= F.lit(cutoff_date)) \
    .toPandas()

if len(current_df) == 0:
    print("   No current weather data in retention window")
else:
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
        
        # Note: observation_id auto-generates, no need to truncate since we're only inserting
        # But we'll truncate anyway for consistency with the full refresh approach
        cursor.execute("TRUNCATE TABLE public.current_weather")
        
        # Insert current weather records
        skipped = 0
        for _, row in current_df.iterrows():
            delta_location_uuid = row['location_id']
            
            # Map Delta UUID to Postgres integer ID
            if delta_location_uuid not in location_id_mapping:
                skipped += 1
                continue
            
            postgres_location_id = location_id_mapping[delta_location_uuid]
            
            # observation_id auto-generates
            cursor.execute("""
                INSERT INTO public.current_weather (location_id, temperature, humidity, wind_speed, precipitation, rain, weather_code, observed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (postgres_location_id, row['temperature'], row['humidity'], row['wind_speed'],
                  row['precipitation'], row['rain'], row['weather_code'], row['observed_at']))
        
        conn.commit()
        print(f"✅ Synced {len(current_df) - skipped} current weather records (skipped {skipped})")
        
    finally:
        cursor.close()
        conn.close()

# COMMAND ----------

# DBTITLE 1,Sync Forecasts (Rolling Window)
import psycopg2
import ast

print(f"Syncing weather_forecasts (last {FORECAST_RETENTION_DAYS} days)...")

# Get location ID mapping
location_id_mapping = ast.literal_eval(spark.conf.get("location_id_mapping"))

# Calculate cutoff date
cutoff_date = datetime.now() - timedelta(days=FORECAST_RETENTION_DAYS)

# Read recent data from Delta
forecasts_df = spark.table(FORECASTS_TABLE) \
    .filter(F.col("forecast_date") >= F.lit(cutoff_date.date())) \
    .toPandas()

if len(forecasts_df) == 0:
    print("   No forecast data in retention window")
else:
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
        
        # Truncate and reload (rolling window)
        cursor.execute("TRUNCATE TABLE public.weather_forecasts")
        
        # Insert forecast records (forecast_id auto-generates)
        skipped = 0
        for _, row in forecasts_df.iterrows():
            delta_location_uuid = row['location_id']
            
            # Map Delta UUID to Postgres integer ID
            if delta_location_uuid not in location_id_mapping:
                skipped += 1
                continue
            
            postgres_location_id = location_id_mapping[delta_location_uuid]
            
            cursor.execute("""
                INSERT INTO public.weather_forecasts (location_id, forecast_date, temperature_max, temperature_min, 
                                                       precipitation_sum, precipitation_hours, wind_speed_max, weather_code, retrieved_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (postgres_location_id, row['forecast_date'], row['temperature_max'], row['temperature_min'],
                  row['precipitation_sum'], row['precipitation_hours'], row['wind_speed_max'], row['weather_code']))
        
        conn.commit()
        print(f"✅ Synced {len(forecasts_df) - skipped} forecast records (skipped {skipped})")
        
    finally:
        cursor.close()
        conn.close()

# COMMAND ----------

# DBTITLE 1,Sync Embeddings (Rolling Window)
import psycopg2
import json

print(f"Syncing weather_embeddings (last {EMBEDDING_RETENTION_DAYS} days)...")

# Calculate cutoff date
cutoff_date = datetime.now() - timedelta(days=EMBEDDING_RETENTION_DAYS)

# Read recent data from Delta
embeddings_df = spark.table(EMBEDDINGS_TABLE) \
    .filter(F.col("date") >= F.lit(cutoff_date)) \
    .toPandas()

if len(embeddings_df) == 0:
    print("   No embedding data in retention window")
else:
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
        
        # Truncate and reload (rolling window)
        cursor.execute("TRUNCATE TABLE public.weather_embeddings")
        
        # Insert embedding records
        for _, row in embeddings_df.iterrows():
            # Convert embedding array to PostgreSQL array format (for vector type)
            embedding_array = list(row['embedding'])
            
            # Parse metadata if it's a JSON string, otherwise use as-is
            metadata_value = row.get('metadata')
            if metadata_value and isinstance(metadata_value, str):
                try:
                    metadata_json = json.loads(metadata_value)
                except:
                    metadata_json = {}
            else:
                metadata_json = metadata_value if metadata_value else {}
            
            cursor.execute("""
                INSERT INTO public.weather_embeddings (id, event_type, location, date, description, embedding, severity, metadata, search_type, created_at)
                VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (row['id'], row['event_type'], row['location'], row['date'], row['description'],
                  embedding_array, row.get('severity'), json.dumps(metadata_json), row.get('search_type'), row.get('created_at')))
        
        conn.commit()
        print(f"✅ Synced {len(embeddings_df)} embedding records")
        
    finally:
        cursor.close()
        conn.close()

# COMMAND ----------

# DBTITLE 1,Summary
import psycopg2

# Get final row counts from Postgres using psycopg2
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
    
    cursor.execute("SELECT COUNT(*) FROM public.locations")
    locations_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM public.current_weather")
    current_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM public.weather_forecasts")
    forecasts_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM public.weather_embeddings")
    embeddings_count = cursor.fetchone()[0]
    
    print("\n" + "="*50)
    print("✅ Sync Complete!")
    print("="*50)
    print(f"\nPostgres Table Counts:")
    print(f"  locations:          {locations_count:,} rows")
    print(f"  current_weather:    {current_count:,} rows")
    print(f"  weather_forecasts:  {forecasts_count:,} rows") 
    print(f"  weather_embeddings: {embeddings_count:,} rows")
    print(f"\nℹ️  Your MCP server can now query these tables from Lakebase Postgres")
    
finally:
    cursor.close()
    conn.close()

# COMMAND ----------

