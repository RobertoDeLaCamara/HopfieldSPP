# HopfieldSPP
Neural model for solving the Shortest Path Problem based on a Hopfield Network based on minimizing the energy function of the network.

# HopfieldSPP Network API Documentation

## Overview

The HopfieldSPP Network API allows users to load the adjacency file of a network and calculate the shortest path between 2 nodes of the network. The API uses the Hopfield Neural Model to solve the Shortest Path Problem.

HopfieldSPP finds the shortest path between two nodes in the network by minimizing the following energy function:


# Energy Function

Energy function to solve the optimization problem of finding the **shortest path** between two nodes in a graph. This function combines multiple terms that impose constraints and objectives on the solution. Here is a general energy function for finding shortest paths without particularizing to a specific (origin, destination) pair:

$$
F = \frac{\mu_1}{2} \sum_{i=1}^n \sum_{j=1}^n C_{ij} x_{ij} + 
    \frac{\mu_2}{2} \sum_{i=1}^n \left( \sum_{j=1}^n x_{ij} - 1 \right)^2 + 
    \frac{\mu_2}{2} \sum_{j=1}^n \left( \sum_{i=1}^n x_{ij} - 1 \right)^2 + 
    \frac{\mu_3}{2} \sum_{i=1}^n \sum_{j=1}^n x_{ij}(1 - x_{ij}) 
$$

## Components of the Energy Function

### 1. **Path Cost**
$$
\frac{\mu_1}{2} \sum_{i=1}^n \sum_{j=1}^n C_{ij} x_{ij}
$$
- **Description**: Minimizes the total cost of the path.
- **Variables**:
  - $C_{ij}$: Cost (or distance) between nodes $i$ and $j$.
  - $x_{ij}$: Binary variable indicating if the path between $i$ and $j$ is part of the solution $(x_{ij} = 1)$ or not $(x_{ij} = 0)$.
- **Purpose**: Encourages the selection of paths with lower cost.

---

### 2. **Row Constraints**
$$
\frac{\mu_2}{2} \sum_{i=1}^n \left( \sum_{j=1}^n x_{ij} - 1 \right)^2
$$
- **Description**: This term ensures that each node has exactly **one outgoing edge**.
- **Variables**:
  - $\sum_{j=1}^n x_{ij}$: Represents the number of outgoing edges from node $i$.
- **Purpose**: Penalizes solutions in which a node has more than one outgoing edge or none.

---

### 3. **Column Constraints**
$$
\frac{\mu_2}{2} \sum_{j=1}^n \left( \sum_{i=1}^n x_{ij} - 1 \right)^2
$$
- **Description**: This term ensures that each node has exactly **one incoming edge**.
- **Variables**:
  - $\sum_{i=1}^n x_{ij}$: Represents the number of incoming edges to node $j$.
- **Purpose**: Penalizes solutions in which a node has more than one incoming edge or none. 

---

### 4. **Binariness Constraint**
$$
\frac{\mu_3}{2} \sum_{i=1}^n \sum_{j=1}^n x_{ij}(1 - x_{ij})
$$
- **Description**: This term forces the variables $(x_{ij})$ to be binary (0 or 1).
- **Variables**:
  - $x_{ij}(1 - x_{ij})$: This product is zero if $(x_{ij})$ is 0 or 1, but is positive if $(x_{ij})$ takes intermediate values.
- **Purpose**: Penalizes solutions in which $x_{ij}$ takes values other than 0 or 1.

---

## Parameters
- $\mu_1, \mu_2, \mu_3$: Weights that balance the importance of each term in the energy function. 
  - $\mu_1$: Prioritizes the minimization of the total cost of the path. 
  - $\mu_2$: Emphasizes path validity. 
  - $\mu_3$: Controls the binariness of the variables. 

---
## Source and destination node restrictions

### Term 5: Source Node Constraint
$$
\left( \sum_{j=1}^n x_{s,j} - 1 \right)^2
$$
- **Variables**:
  - $x_{s,j}$: Binary decision variable for the edge from source node $s$ to node $j$.
- **Purpose**: Ensures that the source node $s$ has exactly one outgoing edge.

### Term 6: Destination Node Constraint
$$
\left( \sum_{i=1}^n x_{i,d} - 1 \right)^2
$$
- **Variables**:
  - $x_{i,d}$: Binary decision variable for the edge from node $i$ to the destination node $d$.
- **Purpose**: Ensures that the destination node $d$ has exactly one incoming edge.

## Summary
The Energy Function combines **strong** (like path validity) and **weak** (like cost minimization and binariness) constraints to: 
1. **Find a valid path**.
2. **Minimize the total cost of the path**.
---

## Endpoints

### 1. **Load Network from File**

#### **POST** `/learnNetwork`

This endpoint loads a network from a CSV file and builds its internal representation.

**Request Body**

- Content-Type: `multipart/form-data`
- Schema:
  - `file` (required): CSV file containing the network data.
    - Type: `string`
    - Format: `binary`
    - Description: The file should contain the network's nodes and edges.

**Responses**

- **200 OK**
  - Description: Network successfully loaded.
  - Content-Type: `application/json`
  - Schema:
    - `message` (`string`): Confirmation message.
    - `status` (`string`): Either `success` or `error`.

- **400 Bad Request**
  - Description: The provided request is invalid.

- **500 Internal Server Error**
  - Description: An error occurred while processing the request.

---

### 2. **Calculate Shortest Path**

#### **GET** `/calculateShortestPath`

This endpoint calculates the shortest path between two nodes in the loaded network.

**Query Parameters**

- `origin` (required): Origin node.
  - Type: `string`
  - Description: The starting point of the path.
- `destination` (required): Destination node.
  - Type: `string`
  - Description: The endpoint of the path.

**Responses**

- **200 OK**
  - Description: Shortest path successfully calculated.
  - Content-Type: `application/json`
  - Schema:
    - `path` (`array of strings`): List of nodes in the shortest path.
    - `distance` (`number`): Total distance of the path.

- **400 Invalid Parameters**
  - Description: Invalid or missing query parameters.

- **404 Path Not Found**
  - Description: No path exists between the specified nodes.

- **500 Internal Server Error**
  - Description: An error occurred while processing the request.

---

## Example Usage

### Load Network

**Request:**
```bash
curl -X POST https://api.hopfieldspp.com/v1/learnNetwork \
  -F "file=@network.csv"

---
### Calculate Shortest Path

**Request:**  
```bash
curl -X GET "https://api.hopfieldspp.com/v1/calculateShortestPath?origin=A&destination=B"

---