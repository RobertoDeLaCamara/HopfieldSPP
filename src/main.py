# File: main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from typing import List
import pandas as pd
import networkx as nx
import os

app = FastAPI()

# In-memory graph storage
graph = None

@app.post("/loadNetwork")
async def load_network(file: UploadFile = File(...)):
    """
    Loads a network from a CSV file.
    """
    global graph
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file.file)

        # Validate the DataFrame has the required columns
        if not {"source", "target", "weight"}.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'source', 'target', and 'weight' columns."
            )

        # Create a graph using NetworkX
        graph = nx.Graph()
        for _, row in df.iterrows():
            graph.add_edge(row["source"], row["target"], weight=row["weight"])

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
        # Calculate the shortest path using Dijkstra's algorithm
        path = nx.shortest_path(graph, source=origin, target=destination, weight="weight")
        distance = nx.shortest_path_length(graph, source=origin, target=destination, weight="weight")
        return {"path": path, "distance": distance}
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="No path found between the specified nodes.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating the shortest path: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
