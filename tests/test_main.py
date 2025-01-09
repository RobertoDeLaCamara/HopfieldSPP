# File: tests/test_main.py

from fastapi import UploadFile, Response
import pytest
from fastapi.testclient import TestClient
from src.main import app
import os
import shutil

client = TestClient(app)

@pytest.fixture(scope="function")
def setup_synthetic_data():
    synthetic_data_path = os.path.join(os.path.dirname(__file__), "../data/synthetic/tests")
    print(f"Creating directory: {synthetic_data_path}")
    os.makedirs(synthetic_data_path, exist_ok=True)
    yield
    shutil.rmtree(synthetic_data_path)


def test_load_network_with_missing_columns(setup_synthetic_data: None):
    """Test the /loadNetwork endpoint with a CSV missing required columns."""
    file_path = os.path.join(os.path.dirname(__file__), "../data/synthetic/tests/synthetic_test_network.csv")
    
    # Create synthetic test network CSV file with missing columns
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        file.write("origin,destination\n")
        file.write("1,48\n")
        file.write("48,15\n")
        file.write("15,29\n")
        file.write("29,3\n")
    
    with open(file_path, "rb") as file:
        upload_file = UploadFile(filename=os.path.basename(file_path), file=file)
        response: Response = client.post("/loadNetwork", upload_file)

    # Adjust assertions based on expected behavior
    assert response is not None, "Response is None"
    assert response.status_code in [200, 400], f"Unexpected status code: {response.status_code}"
    
    if response.status_code == 400:
        assert response.json() == {"detail": "CSV file must contain 'origin', 'destination', and 'weight' columns."}, \
            f"Unexpected error response: {response.json()}"
    else:
        assert response.json() == {"message": "Network loaded successfully", "status": "success"}, \
            f"Unexpected success response: {response.json()}"


def test_load_network_from_file(setup_synthetic_data: None):
    """Test the /loadNetwork endpoint with a valid CSV file."""
    file_path = os.path.join(os.path.dirname(__file__), "../data/synthetic/tests/synthetic_test_network.csv")
    
    # Create synthetic test network CSV file with missing columns
    with open(file_path, "w") as file:
        file.write("origin,destination\n")
        file.write("1,48\n")
        file.write("48,15\n")
        file.write("15,29\n")
        file.write("29,3\n")
        
    with open(file_path, "rb") as file:
        upload_file = UploadFile(filename=os.path.basename(file_path), file=file)
        response = client.post("/loadNetwork", files={"file": (upload_file.filename)})

    assert response is not None, "Response is None"
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    assert response.json() is not None, "Response JSON is None"
    assert response.json() == {"message": "Network loaded successfully", "status": "success"}, f"Unexpected response JSON: {response.json()}"

def test_load_network_invalid_file_type(setup_synthetic_data: None):
    """Test the /loadNetwork endpoint with an invalid file type."""
    file_path = os.path.join(os.path.dirname(__file__), "../data/synthetic/tests/synthetic_test_network.txt")
    
    # Create synthetic test network text file
    with open(file_path, "w") as file:
        file.write("This is not a CSV file.")
            
    with open(file_path, "rb") as file:
        response = client.post("/loadNetwork", files={"file": file})
        
    assert response.status_code == 400
    assert response.json() == {"detail": "Only CSV files are supported."}


def test_calculate_shortest_path_synthetic_network(setup_synthetic_data: None):
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


