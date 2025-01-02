# File: main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from typing import List
import pandas as pd
import networkx as nx
import os

from src.train_model import train_offline_model


app = FastAPI()

# In-memory graph storage
graph = None

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
    global graph
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file.file, usecols=["origin", "destination", "weight"], dtype={"origin": str, "destination": str, "weight": float})

        # Create a graph using NetworkX
        graph = nx.from_pandas_edgelist(df, source="origin", target="destination", edge_attr="weight")
        
        #Create and train the neural model
        #train_offline_model()

        return {"message": "Network loaded successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")


@app.get("/calculateShortestPath")
async def calculate_shortest_path(
    origin: str = Query(..., description="The starting node"),
    destination: str = Query(..., description="The destination node")
):
    """
    Calculates the shortest path between two nodes in the graph.
    """
    global graph
    if graph is None:
        raise HTTPException(status_code=400, detail="Network has not been loaded yet.")

    try:
        # Calculate the shortest path using A* search algorithm
        path = nx.astar_path(graph, source=origin, target=destination, weight="weight", heuristic=lambda u, v: 1)
        distance = nx.astar_path_length(graph, source=origin, target=destination, weight="weight", heuristic=lambda u, v: 1)
        return {"path": path, "distance": distance}
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="No path found between the specified nodes.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating the shortest path: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
