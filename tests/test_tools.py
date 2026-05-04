"""Basic tests for weather tool registration and functionality."""

from unittest.mock import AsyncMock, patch

import pytest

# Import tools to trigger registration
from weather_assistant.tools import weather  # noqa: F401

from framework.tools.registry import (
    _TOOL_REGISTRY,
    get_all_tool_examples,
    get_tool_catalog,
)


class TestToolRegistry:
    def test_all_tools_registered(self):
        expected = {"get_current_weather", "get_forecast", "compare_weather"}
        assert set(_TOOL_REGISTRY.keys()) == expected

    def test_tool_catalog_generated(self):
        catalog = get_tool_catalog()
        assert "get_current_weather" in catalog
        assert "get_forecast" in catalog
        assert "compare_weather" in catalog

    def test_all_tools_have_examples(self):
        all_examples = get_all_tool_examples()
        for tool_name in _TOOL_REGISTRY:
            assert tool_name in all_examples, f"No examples for {tool_name}"
            assert len(all_examples[tool_name]) > 0


class _MockResponse:
    """Sync-style mock matching httpx.Response (json() is not a coroutine)."""

    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class TestWeatherTools:
    @pytest.mark.asyncio
    async def test_get_current_weather_city_not_found(self):
        mock_get = AsyncMock(return_value=_MockResponse({}))
        with patch("httpx.AsyncClient.get", mock_get):
            result = await weather.get_current_weather("Nonexistentcity12345")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_get_forecast_clamps_days(self):
        geo_data = {
            "results": [{"name": "Test", "country": "TC", "latitude": 0, "longitude": 0}]
        }
        weather_data = {
            "daily": {
                "time": ["2026-01-01"],
                "temperature_2m_max": [20],
                "temperature_2m_min": [10],
                "weather_code": [0],
                "precipitation_sum": [0],
                "wind_speed_10m_max": [5],
            }
        }
        mock_get = AsyncMock(side_effect=[
            _MockResponse(geo_data),
            _MockResponse(weather_data),
        ])
        with patch("httpx.AsyncClient.get", mock_get):
            result = await weather.get_forecast("Test", days=99)
            assert "forecast" in result
