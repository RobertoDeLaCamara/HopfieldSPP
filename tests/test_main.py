# File: tests/test_main.py

import pytest
from fastapi.testclient import TestClient
from src.main import app
import os
import shutil

client = TestClient(app)



@pytest.fixture(scope="function")
def setup_synthetic_data():
    synthetic_dir = os.path.join(os.path.dirname(__file__), '../data/synthetic/tests')
    os.makedirs(synthetic_dir, exist_ok=True)
    yield
    shutil.rmtree(synthetic_dir)

def test_load_network_from_file(setup_synthetic_data):
    """Test the /loadNetwork endpoint with a file."""
    file_path = os.path.join(os.path.dirname(__file__), "../data/synthetic/tests/synthetic_test_network.csv")
    
    # Create synthetic test network CSV file
    with open(file_path, "w") as file:
        file.write("origin,destination,weight\n")
        file.write("1,48,50\n")
        file.write("48,15,30\n")
        file.write("15,29,40\n")
        file.write("29,3,49\n")
            
    with open(file_path, "rb") as file:
        response = client.post("/loadNetwork", files={"file": file})
        
    assert response.status_code == 200
    assert response.json() == {"message": "Network loaded successfully", "status": "success"}
    

def test_calculate_shortest_path_synthetic_network(setup_synthetic_data):
    """Test the /calculateShortestPath endpoint with a synthetic network."""
    file_path = os.path.join(os.path.dirname(__file__), "../data/synthetic/synthetic_test_network.csv")
    with open(file_path, "rb") as file:
        client.post("/loadNetwork", files={"file": file})

    response = client.get("/calculateShortestPath?origin=1&destination=3")
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"path": ["1", "48", "15", "29", "3"], "distance": 169} 





