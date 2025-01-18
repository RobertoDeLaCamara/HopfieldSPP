# File: main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import pandas as pd
import os
from src.train_model import HopfieldLayer, HopfieldModel
from src.train_model import train_offline_model
from tensorflow.keras.saving import custom_object_scope
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import logging
import os
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

def calculate_real_path_cost(real_cost_matrix, path):
    """
    Calculates the cost of the shortest path given the real cost matrix and a path.

    Args:
        real_cost_matrix (list of list of float): A 2D list representing the cost matrix of the network
            without normalization.
        path (list of int): A list of node indices representing the shortest path.

    Returns:
        float: The cost of the shortest path.

    """
    cost = 0
    for i in range(len(path) - 1):
        cost += real_cost_matrix[path[i], path[i + 1]]
    return cost


def get_shortest_path(origin, destination):
    """
    Calculates the shortest path between two nodes in the network using a pre-trained Hopfield model.
    
    Args: 
        origin (int): The starting node.
        destination (int): The destination node.
    Returns:
        dict: A dictionary containing the predicted path and the cost of the shortest path.

    Raises:
        ValueError: If cost_matrix, origin, or destination is None.
        RuntimeError: If the model prediction returns an empty path or if any error occurs during the calculation.
    """
    # Load the pre-trained model from the right path depending on the environment (test or production)
    model_path = '../models/'
    if 'PYTEST_CURRENT_TEST' in os.environ:
        model_path = '../data/synthetic/tests/'
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at path: {model_path}")
        raise HTTPException(status_code=404, detail=f"Model not found at path: {model_path}")

    # Configure logging
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    
    
    try:
        # Load the pre-trained model from the right path depending on the environment (test or production)
        with custom_object_scope({'HopfieldModel': HopfieldModel, 'HopfieldLayer': HopfieldLayer}):
            loaded_model = load_model(model_path + 'trained_model.keras', custom_objects={'HopfieldModel': HopfieldModel, 'HopfieldLayer': HopfieldLayer})
        logger.info("Model loaded")
        
        # Load cost matrix
        with open(model_path + 'cost_matrix.pkl', 'rb') as f:
            cost_matrix = pickle.load(f)
        print(cost_matrix)
        logger.info("Cost matrix loaded")
        
        loaded_model.compile(optimizer=Adam(learning_rate=0.01))
        logger.info("Model compiled")
        origin = int(origin)
        destination = int(destination)
        logger.info(f"Origin: {origin}, Destination: {destination}")
        
        
        # Make prediction
        path = loaded_model.predict(origin, destination)
        if not path:
            raise RuntimeError("Model prediction returned an empty path.")
        logger.info(f"Predicted Path: {path}")
        
        # Calculate the cost of the shortest path with real costs
        path_cost = calculate_real_path_cost(cost_matrix, path)
        logger.info(f"Cost of the Shortest Path (Real Costs): {path_cost}")
        
        result = {
            "path": [int(node) for node in path],
            "cost": float(path_cost)
        }
        return result
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while calculating the shortest path: {str(e)}")
        

@app.post("/loadNetwork")
async def load_network(file: UploadFile = File(...)):
    '''
    Loads a network from a CSV file.
    
    Args:
        file (UploadFile): An uploaded file object expected to be a CSV file.
    Raises:
        HTTPException: If the file is not a CSV or if the CSV does not contain the required columns.
        HTTPException: If there is an error processing the file.
    Returns:
        dict: A dictionary containing a success message and status.
    '''
    
    logger.info("Loading network")
    logger.info(f"File name: {file.filename}, File content type: {file.content_type}")

    try:
        content = file.file.read()
        logger.debug(f"File content:\n{content.decode('utf-8')}")
    except Exception as e:
        logger.error(f"Error reading file content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading file content: {str(e)}")
    finally:
        file.file.seek(0)

    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        df = pd.read_csv(
            file.file, usecols=["origin", "destination", "weight"],
            dtype={"origin": str, "destination": str, "weight": float}
        )
        if df.empty:
            raise HTTPException(status_code=400, detail="The CSV file is empty or invalid.")

        logger.info("Training model")
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(content)
        logger.info(f"File saved to {temp_file_path}")

        train_offline_model(temp_file_path)
        os.remove(temp_file_path)
        return {"message": "Network loaded successfully", "status": "success"}
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The CSV file is empty or invalid.")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing the CSV file: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing the file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")

@app.get("/calculateShortestPath")
async def calculate_shortest_path(
    origin: str = Query(..., description="The starting node"),
    destination: str = Query(..., description="The destination node")
):
    """
    Calculates the shortest path between two nodes in the graph.
    """
    logger.info(f"Calculating shortest path from {origin} to {destination}")

    try:
        result = {
            "path": [],
            "cost": 0.0
        }
        origin = int(origin)
        destination = int(destination)
        result = get_shortest_path(origin, destination)  
        logger.info(f"Shortest path calculation result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating the shortest path: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating the shortest path: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=63234)
