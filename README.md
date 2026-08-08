# Weather Prediction MCP Server Agent

A production-ready weather forecasting system built on Databricks, featuring an MCP (Model Context Protocol) server that exposes weather data and analysis tools to AI agents. The system integrates real-time weather data from Open-Meteo's free API with Databricks Lakebase Postgres for storage, Delta Lake for analytics, and vector search for semantic weather queries.

## Features

- **MCP Server**: FastMCP-based server exposing 6 weather tools via Model Context Protocol
- **Real-time Weather Data**: Integration with Open-Meteo API (no API keys required)
- **Lakebase Postgres Storage**: Auto-managed relational storage for locations, current weather, and forecasts
- **Delta Lake Integration**: Scheduled sync jobs for analytics and historical tracking
- **Vector Search**: Semantic search over weather data using sentence-transformers embeddings
- **Multiple Query Methods**: By coordinates or city name for 10 major global cities
- **Health & Activity Insights**: Specialized tools for running conditions and humidity monitoring

## Architecture

```
Agent Bricks Agent  <--(MCP)-->  weather_mcp_server.py  <--(REST)-->  Open-Meteo API
        |                              |                                      |
        |                              v                                      |
        |                    Lakebase Postgres                               |
        |                   (locations, current_weather,                     |
        |                    weather_forecasts,                              |
        |                    weather_embeddings)                             |
        |                              |                                      |
        v                              v                                      v
   Unity Catalog         <-- Delta-to-Postgres Sync -->           Delta Lake Tables
   Functions/Schemas              (Databricks Job)          (weather_locations, weather_current,
                                                             weather_forecasts, weather_embeddings)
```

### Components

1. **MCP Server** (`mcp_server/weather_mcp_server.py`)
   - FastMCP server exposing 6 weather tools
   - Lazy-loads Lakebase connection and embedding model
   - Auto-creates tables on first use
   - Handles coordinate and city-based queries

2. **Weather Broker** (`mcp_server/weather_broker.py`)
   - Adapter for Open-Meteo API
   - City-to-coordinates mapping for 10 major cities
   - WMO weather code interpretation
   - Zero authentication required

3. **Lakebase Postgres**
   - Operational database for real-time weather tracking
   - pgvector extension for semantic search
   - Auto-managed schema with foreign key relationships

4. **Delta Lake**
   - Analytics-ready historical weather data
   - Partitioned by date for efficient queries
   - Source for ML/AI workloads

5. **Sync Jobs**
   - Delta-to-Postgres full refresh sync
   - Handles schema mapping and deduplication
   - CASCADE truncation for referential integrity

## Project Structure

```
Weather-Prediction-MCP-Server-Agent/
├── mcp_server/
│   ├── weather_mcp_server.py      # FastMCP server with 6 weather tools
│   ├── weather_broker.py           # Open-Meteo API adapter
│   ├── app.yaml                    # Databricks App configuration
│   └── requirements.txt            # Python dependencies
├── notebooks/
│   ├── Delta to Postgres Sync.ipynb  # ETL job for Delta → Postgres
│   └── Weather Data to Vector Embeddings.ipynb  # Embedding generation
└── README.md                       # This file
```

## Available MCP Tools

### 1. `get_current_weather`
Retrieve current weather conditions for a location.
- **Inputs**: `latitude`, `longitude` OR `city_name`
- **Returns**: Current temperature, humidity, wind, pressure, weather code
- **Example**: `get_current_weather(city_name="Berlin")`

### 2. `get_forecast`
Get multi-day weather forecast with current conditions.
- **Inputs**: `latitude`, `longitude` OR `city_name`, `forecast_days` (1-16)
- **Returns**: Current + daily forecasts (temp min/max, precipitation, sunrise/sunset)
- **Example**: `get_forecast(city_name="New York", forecast_days=7)`

### 3. `optimal_running_weather`
Evaluate if conditions are ideal for outdoor running.
- **Ideal Range**: 40-60°F (4-15°C) temperature, 30-50% humidity
- **Returns**: Recommendation, current conditions, assessment
- **Example**: `optimal_running_weather(latitude=52.52, longitude=13.41)`

### 4. `check_humidifier_needed`
Check if indoor humidifier should be activated based on outdoor humidity.
- **Threshold**: Recommends action when humidity > 60%
- **Returns**: Recommendation, severity level, health impact notes
- **Example**: `check_humidifier_needed(city_name="Mumbai")`

### 5. `interpret_weather_code`
Convert WMO weather codes to human-readable descriptions.
- **Inputs**: `code` (0-99)
- **Returns**: Weather description (e.g., "Clear sky", "Thunderstorm")
- **Example**: `interpret_weather_code(61)` → "Rain: Slight"

### 6. `vector_search`
Semantic search over historical weather embeddings.
- **Inputs**: `query`, `limit`, `search_type`
- **Returns**: Similar weather events with similarity scores
- **Example**: `vector_search("severe thunderstorms in midwest", limit=10)`
- **Requires**: pgvector extension + pre-computed embeddings

### Supported Cities
Berlin, New York, London, Tokyo, Paris, Sydney, Mumbai, Singapore, Dubai, Toronto

## Setup

### Prerequisites
- Databricks workspace (AWS, Azure, or GCP)
- Lakebase Postgres instance (optional but recommended)
- Unity Catalog enabled (for Delta Lake storage)

### 1. Create Lakebase Postgres Instance

**Via Databricks SDK:**
```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Create project
project = w.postgres.create_project(
    name="weather-db",
    description="Weather forecasting database"
)

# Create branch
branch = w.postgres.create_branch(
    project_id=project.id,
    name="main"
)

# Create compute
compute = w.postgres.create_compute(
    project_id=project.id,
    branch_id=branch.id,
    compute_type="STARTER"  # or STANDARD, PROFESSIONAL
)

# Get connection details
endpoint = w.postgres.get_endpoint(project.id, branch.id)
print(f"Host: {endpoint.host}")
print(f"Database: {endpoint.database}")
```

**Note the connection details:**
- Host: `ep-xxxxxxxx.database.region.cloud.databricks.com`
- Database: `databricks-postgres`
- User: Your Databricks email
- Password: OAuth token or native Postgres password

### 2. Configure Environment Variables

For Databricks App deployment, set in `mcp_server/app.yaml`:

```yaml
environment:
  LAKEBASE_HOST: ep-noisy-king-d8v8sm9z.database.us-east-2.cloud.databricks.com
  LAKEBASE_DATABASE: databricks-postgres
  LAKEBASE_USER: your-email@databricks.com
  LAKEBASE_PASSWORD: ${secrets.database.lakebase-password}
  WEATHER_EMBEDDINGS_TABLE: weather_embeddings
  EMBEDDING_MODEL: all-MiniLM-L6-v2
```

For local development:
```bash
export LAKEBASE_HOST="ep-xxxxx.database.us-east-2.cloud.databricks.com"
export LAKEBASE_DATABASE="databricks-postgres"
export LAKEBASE_USER="your-email@databricks.com"
export LAKEBASE_PASSWORD="your-oauth-token-or-password"
```

### 3. Install Dependencies

```bash
cd mcp_server
pip install -r requirements.txt
```

**Core dependencies:**
- `fastmcp` - MCP server framework
- `psycopg2-binary` - Postgres driver (for Lakebase)
- `sentence-transformers` - Embedding model (for vector search)
- `requests` - HTTP client (for Open-Meteo API)

### 4. Run Locally

```bash
python weather_mcp_server.py
```

Server starts on `http://0.0.0.0:8000` with streamable HTTP transport.

**Test the tools:**
```bash
# Using curl
curl -X POST http://localhost:8000/tools/get_current_weather \
  -H "Content-Type: application/json" \
  -d '{"city_name": "Berlin"}'
```

### 5. Deploy to Databricks Apps

**Create Databricks App:**
1. Navigate to **Compute** > **Apps** > **Create App**
2. Choose **Custom** app type
3. Select the `mcp_server/` directory as source
4. Configure secrets (Lakebase credentials)
5. Deploy

**App Configuration** (`app.yaml`):
```yaml
command: ["python", "weather_mcp_server.py"]
env:
  - name: DATABRICKS_APP_PORT
    value: "8000"
  - name: LAKEBASE_HOST
    value: "${secrets.database.lakebase-host}"
  - name: LAKEBASE_DATABASE
    value: "${secrets.database.lakebase-database}"
  - name: LAKEBASE_USER
    value: "${secrets.database.lakebase-user}"
  - name: LAKEBASE_PASSWORD
    value: "${secrets.database.lakebase-password}"
```

### 6. Register MCP Server with Agent Bricks

1. Copy the deployed app URL
2. Navigate to **AI Gateway** > **MCPs** > **Register External MCP**
3. Paste the app URL as endpoint
4. Name it (e.g., `weather-forecasting`)
5. Databricks introspects and lists all 6 tools

### 7. Create Agent Bricks Agent

1. Go to **Agents** > **Create Agent**
2. Add the `weather-forecasting` MCP as a tool source
3. Optionally add Unity Catalog functions for weather analytics
4. Deploy and test with queries like:
   - "What's the weather in Berlin?"
   - "Should I go running in New York today?"
   - "Get a 7-day forecast for Tokyo"

## Database Schema

### Lakebase Postgres Tables

All tables are auto-created on first use by the MCP server.

#### `locations`
Tracks unique weather query locations.
```sql
CREATE TABLE locations (
    id SERIAL PRIMARY KEY,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    location_name VARCHAR(255),
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    request_count INTEGER DEFAULT 1,
    UNIQUE(latitude, longitude)
);
```

#### `current_weather`
Stores real-time weather observations.
```sql
CREATE TABLE current_weather (
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
```

#### `weather_forecasts`
Daily forecast data with foreign key to locations.
```sql
CREATE TABLE weather_forecasts (
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
```

#### `weather_embeddings`
Vector embeddings for semantic search (requires pgvector extension).
```sql
CREATE TABLE weather_embeddings (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100),
    location VARCHAR(255),
    date DATE,
    description TEXT,
    severity VARCHAR(50),
    metadata JSONB,
    embedding VECTOR(384),  -- all-MiniLM-L6-v2 dimensions
    search_type VARCHAR(50)
);

CREATE INDEX idx_embedding_vector ON weather_embeddings 
USING ivfflat (embedding vector_cosine_ops);
```

### Delta Lake Tables

Mirror schema in Unity Catalog for analytics:
- `weather_locations` - Partitioned by date
- `weather_current` - Partitioned by timestamp
- `weather_forecasts` - Partitioned by forecast_date
- `weather_embeddings` - For ML/AI workloads

## Delta-to-Postgres Sync

Scheduled job syncs Delta tables to Lakebase for operational queries.

**Key Features:**
- Full refresh (TRUNCATE + INSERT)
- CASCADE truncation for foreign key constraints
- UUID-to-integer ID mapping for locations
- Deduplication on (latitude, longitude)
- JSON parsing for embeddings

**Schema Mapping:**
```python
# Delta UUID location_id → Postgres integer location_id
mapping = {
    delta_uuid: postgres_integer_id
}

# Deduplication logic
latest_per_location = df.groupBy("latitude", "longitude") \
    .agg(max("timestamp").alias("max_timestamp"))
```

**Job Configuration:**
```python
# Truncate with CASCADE for referential integrity
cursor.execute("TRUNCATE locations, current_weather, weather_forecasts CASCADE")

# Insert locations, capture generated IDs
for row in deduplicated_locations:
    cursor.execute(
        "INSERT INTO locations (...) VALUES (...) RETURNING id",
        values
    )
    postgres_id = cursor.fetchone()[0]
    mapping[row.location_id] = postgres_id
```

## Development

### Testing Tools Locally

```python
import weather_broker

# Test direct API calls
weather = weather_broker.get_weather_by_city("Berlin", forecast_days=7)
print(weather)

# Test weather code interpretation
code = weather_broker.interpret_weather_code(61)
print(code)  # "Rain: Slight"
```

### MCP Inspector

Use [MCP Inspector](https://github.com/modelcontextprotocol/inspector) for debugging:

```bash
npx @modelcontextprotocol/inspector python weather_mcp_server.py
```

### Vector Search Setup

1. Enable pgvector extension in Lakebase:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2. Generate embeddings (see `Weather Data to Vector Embeddings.ipynb`)

3. Insert sample data:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
description = "Severe thunderstorm with hail"
embedding = model.encode(description).tolist()

cursor.execute(
    "INSERT INTO weather_embeddings (..., embedding) VALUES (..., %s)",
    (embedding,)
)
```

### Debugging

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Common Issues:**

1. **Lakebase connection fails**
   - Verify credentials in environment variables
   - Check SSL/TLS configuration (`sslmode=require`)
   - Ensure compute is running

2. **Vector search returns empty**
   - Verify pgvector extension is installed
   - Check embedding dimensions match (384 for all-MiniLM-L6-v2)
   - Confirm search_type filter matches data

3. **Sync job fails**
   - Check UUID-to-integer mapping logic
   - Verify CASCADE truncation order
   - Review deduplication query

## API Reference

### Open-Meteo API

**Base URL:** `https://api.open-meteo.com/v1/forecast`

**Current Weather:**
```
GET /forecast?latitude=52.52&longitude=13.41&current=temperature_2m,humidity,windspeed
```

**Forecast:**
```
GET /forecast?latitude=52.52&longitude=13.41&daily=temperature_2m_max,temperature_2m_min&forecast_days=7
```

**No authentication required** - Free public API with rate limits.

## Performance

- **MCP Tool Response**: < 500ms average (network dependent)
- **Lakebase Write**: < 100ms for upsert operations
- **Vector Search**: < 200ms for 10k embeddings with IVFFlat index
- **Delta Sync**: ~5-10 minutes for 1M rows (full refresh)

## Security

- All Lakebase connections use TLS/SSL encryption
- Credentials stored in Databricks Secrets
- No API keys required for Open-Meteo
- MCP server authentication via Databricks App integration

## Troubleshooting

**Server won't start:**
```bash
# Check port availability
lsof -i :8000

# Verify dependencies
pip list | grep fastmcp
```

**Tools return errors:**
- Check Open-Meteo API status
- Verify city name spelling (case-insensitive)
- Ensure coordinates are valid (-90 to 90 latitude, -180 to 180 longitude)

**Agent can't connect:**
- Verify app URL is accessible
- Check Databricks App logs
- Confirm MCP registration in AI Gateway

## License

MIT License - See LICENSE file for details

## Credits

- Weather data: [Open-Meteo](https://open-meteo.com/)
- MCP framework: [FastMCP](https://gofastmcp.com/)
- Embeddings: [Sentence Transformers](https://www.sbert.net/)
- Database: [Databricks Lakebase](https://docs.databricks.com/lakebase/)
