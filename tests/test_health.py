"""
Test health endpoints for Auro-PAI Platform Backend
"""

import pytest
from fastapi.testclient import TestClient


def test_simple_health_check(client: TestClient):
    """Test simple health check endpoint."""
    response = client.get("/api/v1/health/simple")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert data["service"] == "auro-pai-backend"


def test_liveness_check(client: TestClient):
    """Test liveness probe endpoint."""
    response = client.get("/api/v1/health/liveness")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "alive"
    assert "timestamp" in data
    assert "uptime_seconds" in data


def test_comprehensive_health_check(client: TestClient):
    """Test comprehensive health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "uptime_seconds" in data
    assert "services" in data
    
    # Check that all expected services are present
    expected_services = ["llm", "rag", "tools", "context"]
    for service in expected_services:
        assert service in data["services"]


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["name"] == "Auro-PAI Platform Backend"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"


def test_status_endpoint(client: TestClient):
    """Test status endpoint."""
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    
    data = response.json()
    assert "platform" in data
    assert "services" in data
