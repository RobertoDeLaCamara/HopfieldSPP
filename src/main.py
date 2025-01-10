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
    
    print("Loading network")
    
    # Debug: Check the file content type and name
    print(f"File name: {file.filename}")
    print(f"File content type: {file.content_type}")

    # Debug: Read the content of the file
    try:
        content = file.file.read()  # Read the file content
        print(f"File content:\n{content.decode('utf-8')}")  # Decode for readability if it's text
    except Exception as e:
        print(f"Error reading file content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading file content: {str(e)}")
    finally:
        # Reset the file pointer to the beginning for further use
        file.file.seek(0)

    # Continue with your normal processing logic
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        df = pd.read_csv(
            file.file, usecols=["origin", "destination", "weight"],
            dtype={"origin": str, "destination": str, "weight": float}
        )
        if df.empty:
            raise HTTPException(status_code=400, detail="The CSV file is empty or invalid.")
        
        print("Training model")
        # train_offline_model(file.file)
        # Save the uploaded file to a temporary location
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(content)
        print(f"File saved to {temp_file_path}")
        # Train the model using the path to the temporary file
        train_offline_model(temp_file_path)
        # Optionally, remove the temporary file after processing
        os.remove(temp_file_path)
        return {"message": "Network loaded successfully", "status": "success"}
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The CSV file is empty or invalid.")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing the CSV file: {str(e)}")
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
