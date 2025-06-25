"""
Test configuration and fixtures for Auro-PAI Platform Backend
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_chat_request():
    """Sample chat request for testing."""
    return {
        "message": "Hello, how can you help me?",
        "include_rag": True,
        "include_tools": False,
        "temperature": 0.7
    }


@pytest.fixture
def sample_search_request():
    """Sample search request for testing."""
    return {
        "query": "Python FastAPI best practices",
        "max_results": 5
    }


@pytest.fixture
def sample_index_request():
    """Sample index request for testing."""
    return {
        "directory_path": "/tmp/test_code"
    }
