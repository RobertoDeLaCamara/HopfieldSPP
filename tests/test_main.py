# File: tests/test_main.py

import pytest
from fastapi.testclient import TestClient
from src.main import app
import os

client = TestClient(app)

@pytest.fixture
def sample_csv(tmpdir):
    """Creates a temporary CSV file for testing."""
    csv_content = "origin,destination,weight\nA,B,1\nB,C,2\nC,D,1\nA,D,4"
    csv_file = tmpdir.join("network.csv")
    csv_file.write(csv_content)
    return csv_file

def test_load_network(sample_csv):
    """Test the /loadNetwork endpoint."""
    with open(sample_csv, "rb") as file:
        response = client.post("/loadNetwork", files={"file": file})
    assert response.status_code == 200
    assert response.json() == {"message": "Network loaded successfully", "status": "success"}

def test_load_network_from_file():
    """Test the /loadNetwork endpoint with a file."""
    file_path = os.path.join(os.path.dirname(__file__), "../data/synthetic/synthetic_network.csv")
    with open(file_path, "rb") as file:
        response = client.post("/loadNetwork", files={"file": file})
    assert response.status_code == 200
    assert response.json() == {"message": "Network loaded successfully", "status": "success"}
    

def test_calculate_shortest_path(sample_csv):
    """Test the /calculateShortestPath endpoint."""
    # Load the network first
    with open(sample_csv, "rb") as file:
        client.post("/loadNetwork", files={"file": file})

    # Test shortest path calculation
    response = client.get("/calculateShortestPath?origin=A&destination=D")
    assert response.status_code == 200
    assert response.json() == {"path": ["A", "D"], "distance": 4}

def test_calculate_shortest_path_synthetic_network():
    """Test the /calculateShortestPath endpoint with a synthetic network."""
    file_path = os.path.join(os.path.dirname(__file__), "../data/synthetic/synthetic_network.csv")
    with open(file_path, "rb") as file:
        client.post("/loadNetwork", files={"file": file})

    response = client.get("/calculateShortestPath?origin=1&destination=3")
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"path": ["1", "48", "15", "29", "3"], "distance": 169} 

def test_shortest_path_no_network():
    """Test shortest path when network is not loaded."""
    response = client.get("/calculateShortestPath?origin=A&destination=D")
    assert response.status_code == 400
    assert response.json()["detail"] == "Network has not been loaded yet."

def test_shortest_path_no_path(sample_csv):
    """Test shortest path with unreachable nodes."""
    # Load the network
    with open(sample_csv, "rb") as file:
        client.post("/learnNetwork", files={"file": file})

    # Test for unreachable nodes
    response = client.get("/calculateShortestPath?origin=A&destination=Z")
    assert response.status_code == 404
    assert response.json()["detail"] == "No path found between the specified nodes."

