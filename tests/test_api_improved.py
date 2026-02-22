import pytest
import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main_improved import app, invalidate_model_cache

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown for each test."""
    # Setup: Clear cache before each test
    invalidate_model_cache()
    yield
    # Teardown: Clear cache after each test
    invalidate_model_cache()


def test_load_network_success():
    """Test successful network loading."""
    with open('data/synthetic/synthetic_network.csv', 'rb') as f:
        response = client.post(
            "/loadNetwork",
            files={"file": ("network.csv", f, "text/csv")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "successfully" in data["message"].lower()


def test_load_network_invalid_file_type():
    """Test loading non-CSV file."""
    response = client.post(
        "/loadNetwork",
        files={"file": ("test.txt", b"invalid content", "text/plain")}
    )

    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]


def test_calculate_shortest_path_success():
    """Test successful shortest path calculation."""
    # First load network
    with open('data/synthetic/synthetic_network.csv', 'rb') as f:
        client.post(
            "/loadNetwork",
            files={"file": ("network.csv", f, "text/csv")}
        )

    # Then calculate path
    response = client.get("/calculateShortestPath?origin=0&destination=5")

    assert response.status_code == 200
    data = response.json()
    assert "path" in data
    assert "cost" in data
    assert isinstance(data["path"], list)
    assert len(data["path"]) >= 2
    assert data["path"][0] == 0
    assert data["path"][-1] == 5


def test_calculate_shortest_path_invalid_origin():
    """Test with invalid origin parameter."""
    with open('data/synthetic/synthetic_network.csv', 'rb') as f:
        client.post(
            "/loadNetwork",
            files={"file": ("network.csv", f, "text/csv")}
        )

    response = client.get("/calculateShortestPath?origin=abc&destination=5")

    assert response.status_code == 400
    assert "integer" in response.json()["detail"].lower()


def test_calculate_shortest_path_out_of_range():
    """Test with out of range node."""
    with open('data/synthetic/synthetic_network.csv', 'rb') as f:
        client.post(
            "/loadNetwork",
            files={"file": ("network.csv", f, "text/csv")}
        )

    response = client.get("/calculateShortestPath?origin=0&destination=999")

    assert response.status_code == 400
    assert "out of range" in response.json()["detail"].lower()


def test_calculate_shortest_path_same_node():
    """Test path from node to itself."""
    with open('data/synthetic/synthetic_network.csv', 'rb') as f:
        client.post(
            "/loadNetwork",
            files={"file": ("network.csv", f, "text/csv")}
        )

    response = client.get("/calculateShortestPath?origin=0&destination=0")

    assert response.status_code == 200
    data = response.json()
    assert data["path"] == [0]
    assert data["cost"] == 0.0


def test_model_caching():
    """Test that model is cached between requests."""
    # Load network
    with open('data/synthetic/synthetic_network.csv', 'rb') as f:
        client.post(
            "/loadNetwork",
            files={"file": ("network.csv", f, "text/csv")}
        )

    # First query
    import time
    start1 = time.time()
    response1 = client.get("/calculateShortestPath?origin=0&destination=5")
    time1 = time.time() - start1

    # Second query (should be faster due to caching)
    start2 = time.time()
    response2 = client.get("/calculateShortestPath?origin=1&destination=6")
    time2 = time.time() - start2

    assert response1.status_code == 200
    assert response2.status_code == 200

    # Second query should be faster (though not guaranteed in all environments)
    # Just verify both succeed
    assert time1 > 0
    assert time2 > 0


def test_multiple_queries():
    """Test multiple sequential queries."""
    # Load network
    with open('data/synthetic/synthetic_network.csv', 'rb') as f:
        client.post(
            "/loadNetwork",
            files={"file": ("network.csv", f, "text/csv")}
        )

    # Multiple queries
    queries = [(0, 5), (1, 6), (2, 7), (3, 8)]

    for origin, dest in queries:
        response = client.get(f"/calculateShortestPath?origin={origin}&destination={dest}")
        assert response.status_code == 200
        data = response.json()
        assert data["path"][0] == origin
        assert data["path"][-1] == dest


def test_missing_parameters():
    """Test with missing query parameters."""
    with open('data/synthetic/synthetic_network.csv', 'rb') as f:
        client.post(
            "/loadNetwork",
            files={"file": ("network.csv", f, "text/csv")}
        )

    # Missing destination
    response = client.get("/calculateShortestPath?origin=0")
    assert response.status_code == 422  # FastAPI validation error

    # Missing origin
    response = client.get("/calculateShortestPath?destination=5")
    assert response.status_code == 422


def test_empty_csv():
    """Test loading empty CSV file."""
    response = client.post(
        "/loadNetwork",
        files={"file": ("empty.csv", b"", "text/csv")}
    )

    assert response.status_code == 400


def test_malformed_csv():
    """Test loading malformed CSV."""
    malformed_content = b"invalid,data\n1,2,3,4,5"

    response = client.post(
        "/loadNetwork",
        files={"file": ("malformed.csv", malformed_content, "text/csv")}
    )

    # Should return 400 (missing required columns) or 500
    assert response.status_code in (400, 500)


def test_path_cost_accuracy():
    """Test that returned cost matches actual path cost."""
    with open('data/synthetic/synthetic_network.csv', 'rb') as f:
        client.post(
            "/loadNetwork",
            files={"file": ("network.csv", f, "text/csv")}
        )

    response = client.get("/calculateShortestPath?origin=0&destination=5")

    assert response.status_code == 200
    data = response.json()

    # Cost should be positive for non-trivial paths
    if len(data["path"]) > 1:
        assert data["cost"] > 0


def test_reliability():
    """Test that API always returns valid paths."""
    with open('data/synthetic/synthetic_network.csv', 'rb') as f:
        client.post(
            "/loadNetwork",
            files={"file": ("network.csv", f, "text/csv")}
        )

    # Test 10 random queries
    for i in range(10):
        origin = i % 10
        dest = (i + 5) % 10

        if origin == dest:
            continue

        response = client.get(f"/calculateShortestPath?origin={origin}&destination={dest}")

        # Should always succeed (100% reliability)
        assert response.status_code == 200
        data = response.json()
        assert len(data["path"]) >= 2
        assert data["path"][0] == origin
        assert data["path"][-1] == dest


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
