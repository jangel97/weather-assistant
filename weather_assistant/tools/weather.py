"""Weather tools using the Open-Meteo API (free, no API key needed)."""

import httpx

from framework.tools.registry import register_examples, register_tool

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


async def _geocode(city: str) -> dict | None:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(_GEOCODE_URL, params={"name": city, "count": 1})
        resp.raise_for_status()
        data = resp.json()
    results = data.get("results")
    if not results:
        return None
    r = results[0]
    return {
        "name": r["name"],
        "country": r.get("country", ""),
        "latitude": r["latitude"],
        "longitude": r["longitude"],
    }


async def _fetch_current(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                   "weather_code,wind_speed_10m,wind_direction_10m",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(_WEATHER_URL, params=params)
        resp.raise_for_status()
    return resp.json()


async def _fetch_forecast(lat: float, lon: float, days: int) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,"
                 "precipitation_sum,wind_speed_10m_max",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "forecast_days": days,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(_WEATHER_URL, params=params)
        resp.raise_for_status()
    return resp.json()


@register_tool
async def get_current_weather(city: str) -> dict:
    """Get current weather conditions for a city."""
    geo = await _geocode(city)
    if not geo:
        return {"error": f"City '{city}' not found"}

    data = await _fetch_current(geo["latitude"], geo["longitude"])
    current = data.get("current", {})

    return {
        "city": geo["name"],
        "country": geo["country"],
        "temperature_c": current.get("temperature_2m"),
        "feels_like_c": current.get("apparent_temperature"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "wind_direction_deg": current.get("wind_direction_10m"),
        "conditions": _WMO_CODES.get(current.get("weather_code", -1), "Unknown"),
    }


@register_tool
async def get_forecast(city: str, days: int = 3) -> dict:
    """Get daily weather forecast for a city (1-7 days)."""
    days = max(1, min(days, 7))
    geo = await _geocode(city)
    if not geo:
        return {"error": f"City '{city}' not found"}

    data = await _fetch_forecast(geo["latitude"], geo["longitude"], days)
    daily = data.get("daily", {})

    forecast = []
    dates = daily.get("time", [])
    for i, date in enumerate(dates):
        forecast.append({
            "date": date,
            "max_temp_c": daily["temperature_2m_max"][i],
            "min_temp_c": daily["temperature_2m_min"][i],
            "conditions": _WMO_CODES.get(daily["weather_code"][i], "Unknown"),
            "precipitation_mm": daily["precipitation_sum"][i],
            "max_wind_kmh": daily["wind_speed_10m_max"][i],
        })

    return {
        "city": geo["name"],
        "country": geo["country"],
        "forecast": forecast,
    }


@register_tool
async def compare_weather(cities: str) -> dict:
    """Compare current weather across multiple cities (comma-separated)."""
    city_list = [c.strip() for c in cities.split(",") if c.strip()]
    if len(city_list) < 2:
        return {"error": "Provide at least 2 cities separated by commas"}

    results = []
    for city_name in city_list:
        result = await get_current_weather(city_name)
        results.append(result)

    return {"comparison": results}


register_examples("get_current_weather", [
    {"question": "What's the weather in Tokyo?", "arguments": {"city": "Tokyo"}},
    {"question": "Temperature in New York", "arguments": {"city": "New York"}},
    {"question": "Is it raining in London?", "arguments": {"city": "London"}},
])

register_examples("get_forecast", [
    {"question": "5-day forecast for Paris", "arguments": {"city": "Paris", "days": 5}},
    {"question": "Will it rain in Berlin this week?", "arguments": {"city": "Berlin", "days": 7}},
    {"question": "Tomorrow's weather in Madrid", "arguments": {"city": "Madrid", "days": 1}},
])

register_examples("compare_weather", [
    {
        "question": "Compare weather in Tokyo and London",
        "arguments": {"cities": "Tokyo, London"},
    },
    {
        "question": "Which is warmer, Madrid or Paris?",
        "arguments": {"cities": "Madrid, Paris"},
    },
])
