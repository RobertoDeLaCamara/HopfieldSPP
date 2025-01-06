# HopfieldSPP
Neural model for solving the Shortest Path Problem based on a Hopfield Network based on minimizing the energy function of the network.

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

**Term 5: Source Node Constraint**

$
\left( \sum_{j=1}^n x_{s,j} - 1 \right)^2
$

- $x_{s,j}$: Binary decision variable for the edge from source node $s$ to node $j$.
- Ensures that the source node $s$ has exactly one outgoing edge.

**Term 6: Destination Node Constraint**

$
\left( \sum_{i=1}^n x_{i,d} - 1 \right)^2
$

- $ x_{i,d} $: Binary decision variable for the edge from node $ i $ to the destination node $ d $.
- Ensures that the destination node $ d $ has exactly one incoming edge.

## Summary
The Energy Function combines **strong** (like path validity) and **weak** (like cost minimization and binariness) constraints to: 
1. **Find a valid path**.
2. **Minimize the total cost of the path**.
