# File: main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from typing import List
import pandas as pd
import os
from src.train_model import train_offline_model
from src.calculate_shortest_path import calculate_shortest_path 

app = FastAPI()


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
    
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file.file, usecols=["origin", "destination", "weight"], dtype={"origin": str, "destination": str, "weight": float})
           
        #Create and train the neural model
        adjacency_matrix_file = file.file
        train_offline_model(adjacency_matrix_file, test=False)
       
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
    
    try:
        # return calculate_shortest_path(graph, origin, destination
        return calculate_shortest_path(None, origin, destination)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating the shortest path: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
