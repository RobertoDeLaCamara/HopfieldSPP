def calculate_shortest_path(origin, destination):
    """
    Calculates the shortest path between two nodes in the network.

    Args:
        origin: The starting node.
        destination: The destination node.

    Returns:
        dict: A dictionary with two keys: "path" and "distance". The "path" key
            contains a list of nodes representing the shortest path from the origin
            to the destination. The "distance" key contains the total cost of the
            shortest path.
    """
    