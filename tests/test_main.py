# File: tests/test_main.py

import pytest
from fastapi.testclient import TestClient
from src.main import app
import os
import shutil
import pandas as pd

client = TestClient(app)

@pytest.fixture(scope="function")
def setup_synthetic_data():
    synthetic_dir = os.path.join(os.path.dirname(__file__), '../data/synthetic/tests')
    os.makedirs(synthetic_dir, exist_ok=True)
    yield
    shutil.rmtree(synthetic_dir)


def test_load_network_from_file(setup_synthetic_data):
    """Test the /loadNetwork endpoint with a file."""
    
    network_data = {'origin': ['1', '48', '15', '29'], 'destination': ['48', '15', '29', '3'], 'weight': [50, 30, 40,49]}
    network_df = pd.DataFrame(network_data)
    adjacency_matrix_file = '../data/synthetic/tests/test_network.csv'
    network_df.to_csv(adjacency_matrix_file, index=False)
    
    with open(adjacency_matrix_file, "rb") as file:
        response = client.post("/loadNetwork", files={"file": file})
        
    assert response.status_code == 200
    assert response.json() == {"message": "Network loaded successfully", "status": "success"}

def test_load_network_invalid_file_type(setup_synthetic_data):
    """Test the /loadNetwork endpoint with an invalid file type."""
    file_path = os.path.join(os.path.dirname(__file__), "../data/synthetic/tests/synthetic_test_network.txt")
    
    # Create synthetic test network text file
    with open(file_path, "w") as file:
        file.write("This is not a CSV file.")
            
    with open(file_path, "rb") as file:
        response = client.post("/loadNetwork", files={"file": file})
        
    assert response.status_code == 400
    assert response.json() == {"detail": "Only CSV files are supported."}

def test_load_network_missing_columns(setup_synthetic_data):
    """Test the /loadNetwork endpoint with a CSV file missing required columns."""
    file_path = os.path.join(os.path.dirname(__file__), "../data/synthetic/tests/synthetic_test_network.csv")
    
    # Create synthetic test network CSV file with missing columns
    with open(file_path, "w") as file:
        file.write("origin,destination\n")
        file.write("1,48\n")
        file.write("48,15\n")
        file.write("15,29\n")
        file.write("29,3\n")
            
    with open(file_path, "rb") as file:
        response = client.post("/loadNetwork", files={"file": file})
        
    assert response.status_code == 400
    assert response.json() == {"detail": "CSV file must contain 'origin', 'destination', and 'weight' columns."}

def test_calculate_shortest_path_synthetic_network(setup_synthetic_data):
    """Test the /calculateShortestPath endpoint with a synthetic network."""
    file_path = os.path.join(os.path.dirname(__file__), "../data/synthetic/tests/synthetic_test_network.csv")
    
    # Create synthetic test network CSV file
    with open(file_path, "w") as file:
        file.write("origin,destination,weight\n")
        file.write("1,48,50\n")
        file.write("48,15,30\n")
        file.write("15,29,40\n")
        file.write("29,3,49\n")
            
    with open(file_path, "rb") as file:
        client.post("/loadNetwork", files={"file": file})

    response = client.get("/calculateShortestPath?origin=1&destination=3")
    assert response.status_code == 200
    assert response.json() == {"path": ["1", "48", "15", "29", "3"], "distance": 169}


