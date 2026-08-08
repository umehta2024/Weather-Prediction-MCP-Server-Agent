-- Weather Forecasting Database Schema
-- Three-table normalized design for storing locations and weather data
-- Database: Lakebase Postgres
-- Created: 2025

-- ============================================================================
-- Table 1: locations
-- Master list of tracked weather locations with coordinates and metadata
-- ============================================================================

CREATE TABLE IF NOT EXISTS locations (
    location_id SERIAL PRIMARY KEY,
    city_name VARCHAR(100),
    country VARCHAR(100),
    latitude DECIMAL(10, 7) NOT NULL,
    longitude DECIMAL(10, 7) NOT NULL,
    timezone VARCHAR(50),
    elevation INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (latitude, longitude)
);

COMMENT ON TABLE locations IS 'Master registry of weather monitoring locations';
COMMENT ON COLUMN locations.location_id IS 'Primary key for location';
COMMENT ON COLUMN locations.latitude IS 'Latitude coordinate (decimal degrees)';
COMMENT ON COLUMN locations.longitude IS 'Longitude coordinate (decimal degrees)';
COMMENT ON COLUMN locations.timezone IS 'IANA timezone identifier (e.g. America/New_York)';
COMMENT ON COLUMN locations.elevation IS 'Elevation in meters above sea level';

-- ============================================================================
-- Table 2: weather_forecasts
-- Daily weather forecast data (typically 7-16 day forecasts from Open-Meteo)
-- ============================================================================

CREATE TABLE IF NOT EXISTS weather_forecasts (
    forecast_id BIGSERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    forecast_date DATE NOT NULL,
    temperature_max DECIMAL(5, 2),
    temperature_min DECIMAL(5, 2),
    precipitation_sum DECIMAL(6, 2),
    precipitation_hours INT,
    wind_speed_max DECIMAL(5, 2),
    weather_code INT,
    retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (location_id) REFERENCES locations(location_id) ON DELETE CASCADE,
    UNIQUE (location_id, forecast_date, retrieved_at)
);

COMMENT ON TABLE weather_forecasts IS 'Daily weather forecasts retrieved from Open-Meteo API';
COMMENT ON COLUMN weather_forecasts.forecast_date IS 'Date this forecast is for (local date at location)';
COMMENT ON COLUMN weather_forecasts.temperature_max IS 'Maximum temperature in Celsius';
COMMENT ON COLUMN weather_forecasts.temperature_min IS 'Minimum temperature in Celsius';
COMMENT ON COLUMN weather_forecasts.precipitation_sum IS 'Total precipitation in millimeters';
COMMENT ON COLUMN weather_forecasts.precipitation_hours IS 'Number of hours with precipitation';
COMMENT ON COLUMN weather_forecasts.wind_speed_max IS 'Maximum wind speed in km/h';
COMMENT ON COLUMN weather_forecasts.weather_code IS 'WMO weather code (0-99)';
COMMENT ON COLUMN weather_forecasts.retrieved_at IS 'Timestamp when this forecast was fetched';

-- Index for efficient queries by location and date
CREATE INDEX IF NOT EXISTS idx_location_forecast_date 
ON weather_forecasts(location_id, forecast_date);

-- ============================================================================
-- Table 3: current_weather
-- Current weather observations (real-time or near-real-time data)
-- ============================================================================

CREATE TABLE IF NOT EXISTS current_weather (
    observation_id BIGSERIAL PRIMARY KEY,
    location_id INT NOT NULL,
    temperature DECIMAL(5, 2),
    humidity INT,
    wind_speed DECIMAL(5, 2),
    precipitation DECIMAL(6, 2),
    rain DECIMAL(6, 2),
    weather_code INT,
    observed_at TIMESTAMP NOT NULL,
    FOREIGN KEY (location_id) REFERENCES locations(location_id) ON DELETE CASCADE
);

COMMENT ON TABLE current_weather IS 'Current weather observations from Open-Meteo API';
COMMENT ON COLUMN current_weather.temperature IS 'Current temperature in Celsius';
COMMENT ON COLUMN current_weather.humidity IS 'Relative humidity percentage (0-100)';
COMMENT ON COLUMN current_weather.wind_speed IS 'Current wind speed in km/h';
COMMENT ON COLUMN current_weather.precipitation IS 'Current precipitation in millimeters';
COMMENT ON COLUMN current_weather.rain IS 'Current rain amount in millimeters';
COMMENT ON COLUMN current_weather.weather_code IS 'WMO weather code (0-99)';
COMMENT ON COLUMN current_weather.observed_at IS 'Timestamp of this observation';

-- Index for efficient time-series queries (most recent first)
CREATE INDEX IF NOT EXISTS idx_location_observed_at 
ON current_weather(location_id, observed_at DESC);

-- ============================================================================
-- Sample Data: Seed common cities
-- ============================================================================

INSERT INTO locations (city_name, country, latitude, longitude, timezone, elevation)
VALUES 
    ('Berlin', 'Germany', 52.52, 13.41, 'Europe/Berlin', 34),
    ('New York', 'United States', 40.71, -74.01, 'America/New_York', 10),
    ('London', 'United Kingdom', 51.51, -0.13, 'Europe/London', 11),
    ('Tokyo', 'Japan', 35.68, 139.65, 'Asia/Tokyo', 40),
    ('Paris', 'France', 48.85, 2.35, 'Europe/Paris', 35),
    ('Sydney', 'Australia', -33.87, 151.21, 'Australia/Sydney', 58),
    ('Mumbai', 'India', 19.08, 72.88, 'Asia/Kolkata', 14),
    ('Singapore', 'Singapore', 1.35, 103.82, 'Asia/Singapore', 15),
    ('Dubai', 'United Arab Emirates', 25.20, 55.27, 'Asia/Dubai', 5),
    ('Toronto', 'Canada', 43.65, -79.38, 'America/Toronto', 76)
ON CONFLICT (latitude, longitude) DO NOTHING;

-- ============================================================================
-- Example Queries
-- ============================================================================

-- Get 7-day forecast for a city
-- SELECT l.city_name, wf.forecast_date, wf.temperature_max, wf.temperature_min
-- FROM locations l
-- JOIN weather_forecasts wf ON l.location_id = wf.location_id
-- WHERE l.city_name = 'Berlin'
--   AND wf.forecast_date >= CURRENT_DATE
-- ORDER BY wf.forecast_date;

-- Get current weather for all locations
-- SELECT l.city_name, cw.temperature, cw.humidity, cw.wind_speed
-- FROM locations l
-- JOIN LATERAL (
--     SELECT * FROM current_weather
--     WHERE location_id = l.location_id
--     ORDER BY observed_at DESC
--     LIMIT 1
-- ) cw ON true;

-- Compare forecasts across cities
-- SELECT l.city_name, AVG(wf.temperature_max) as avg_temp
-- FROM locations l
-- JOIN weather_forecasts wf ON l.location_id = wf.location_id
-- WHERE wf.forecast_date = CURRENT_DATE
-- GROUP BY l.city_name
-- ORDER BY avg_temp DESC;