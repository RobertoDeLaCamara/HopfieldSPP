# File: main_improved.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import pandas as pd
import os
from src.train_model_improved import ImprovedHopfieldLayer, ImprovedHopfieldModel, train_improved_model
from tensorflow.keras.saving import custom_object_scope
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import logging
import pickle
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global cache for model (FIXED: avoid reloading on every request)
_model_cache = {
    "model": None,
    "cost_matrix": None,
    "timestamp": None
}


def get_cached_model():
    """Load model once and cache in memory."""
    model_path = os.path.join(os.getcwd(), 'models/')

    if 'PYTEST_CURRENT_TEST' in os.environ:
        model_path = 'data/synthetic/tests/'

    if not os.path.exists(model_path):
        logger.error(f"Model not found at path: {model_path}")
        raise HTTPException(status_code=404, detail=f"Model not found at path: {model_path}")

    # Check if model is already cached
    if _model_cache["model"] is not None:
        logger.info("Using cached model")
        return _model_cache["model"], _model_cache["cost_matrix"]

    try:
        # Load model
        with custom_object_scope({
            'ImprovedHopfieldModel': ImprovedHopfieldModel,
            'ImprovedHopfieldLayer': ImprovedHopfieldLayer
        }):
            loaded_model = load_model(
                model_path + 'trained_model_improved.keras',
                custom_objects={
                    'ImprovedHopfieldModel': ImprovedHopfieldModel,
                    'ImprovedHopfieldLayer': ImprovedHopfieldLayer
                }
            )
        logger.info("Model loaded")

        # Load cost matrix
        with open(model_path + 'cost_matrix_improved.pkl', 'rb') as f:
            cost_matrix = pickle.load(f)
        logger.info("Cost matrix loaded")

        loaded_model.compile(optimizer=Adam(learning_rate=0.02))
        loaded_model.set_cost_matrix(cost_matrix)

        # Cache the model
        _model_cache["model"] = loaded_model
        _model_cache["cost_matrix"] = cost_matrix
        _model_cache["timestamp"] = time.time()

        return loaded_model, cost_matrix

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


def invalidate_model_cache():
    """Clear model cache when new model is trained."""
    global _model_cache
    _model_cache = {"model": None, "cost_matrix": None, "timestamp": None}
    logger.info("Model cache invalidated")


def get_shortest_path(origin, destination):
    """
    Calculates the shortest path using improved Hopfield model with fallback.
    """
    try:
        model, cost_matrix = get_cached_model()

        origin = int(origin)
        destination = int(destination)
        logger.info(f"Origin: {origin}, Destination: {destination}")

        # Validate node indices
        n = len(cost_matrix)
        if origin < 0 or origin >= n:
            raise ValueError(f"Origin node {origin} is out of range [0, {n-1}]")
        if destination < 0 or destination >= n:
            raise ValueError(f"Destination node {destination} is out of range [0, {n-1}]")

        # Make prediction with improved algorithm
        path = model.predict_path(origin, destination, num_restarts=3, validate=True)

        if not path:
            raise RuntimeError("Model prediction returned an empty path.")

        logger.info(f"Predicted Path: {path}")

        # Calculate path cost
        path_cost = model._calculate_path_cost(path)
        logger.info(f"Cost of the Shortest Path: {path_cost}")

        result = {
            "path": [int(node) for node in path],
            "cost": float(path_cost)
        }
        return result

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error calculating shortest path: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/loadNetwork")
async def load_network(file: UploadFile = File(...)):
    '''
    Loads a network from a CSV file and trains improved model.
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

        logger.info("Training improved model")
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(content)
        logger.info(f"File saved to {temp_file_path}")

        train_improved_model(temp_file_path)
        os.remove(temp_file_path)

        # Invalidate cache so new model is loaded
        invalidate_model_cache()

        return {"message": "Network loaded successfully", "status": "success"}

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The CSV file is empty or invalid.")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing the CSV file: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing the file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")


@app.get("/calculateShortestPath")
async def calculate_shortest_path(
    origin: str = Query(..., description="The starting node"),
    destination: str = Query(..., description="The destination node")
):
    """
    Calculates the shortest path between two nodes using improved algorithm.
    """
    logger.info(f"Calculating shortest path from {origin} to {destination}")

    try:
        origin = int(origin)
        destination = int(destination)
    except ValueError:
        raise HTTPException(status_code=400, detail="Origin and destination must be valid integers")

    try:
        result = get_shortest_path(origin, destination)
        logger.info(f"Shortest path calculation result: {result}")
        return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Path not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error calculating the shortest path: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating the shortest path: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=63235)
