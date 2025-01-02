# src/visualize_graph.py

import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(df):
    """
    Visualize a network represented in a DataFrame as a graph.
    
    The graph is drawn using an automatic layout to improve visualization.
    Nodes are drawn as blue rectangles, edges are drawn as black lines, and
    edge weights are drawn as red labels.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['source', 'target', 'weight'].
    """
    # Check that df is not None and does not have empty rows
    if df is None or df.empty:
        raise ValueError("The DataFrame cannot be None or empty")
    
    # Check that the DataFrame has the necessary columns
    for col in ['source', 'target', 'weight']:
        if col not in df.columns:
            raise ValueError(f"The DataFrame must have the column '{col}'")
    
    # Check that there are no rows with null values
    if df.isnull().any().any():
        raise ValueError("The DataFrame cannot have null values")
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges and their weights
    G.add_weighted_edges_from(df.values.tolist())
    
    # Get node positions using a layout
    # The 'spring' layout is a good option for networks with a moderate number of nodes
    pos = nx.spring_layout(G)
    
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')
    
    # Extract weights to label the edges
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')
    
    # Show the graph
    plt.title("Network")
    plt.show()
