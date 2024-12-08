# src/visualize_graph.py

import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(df):
    """
    Visualiza una red representada en un DataFrame como un grafo.
    
    El grafo se dibuja utilizando un layout automático para mejorar la visualización.
    Los nodos se dibujan como rectángulos azules, los arcos se dibujan como líneas negras y los
    pesos de los arcos se dibujan como etiquetas rojas.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas ['origen', 'destino', 'costo'].
    """
    # Verificar que df no sea None ni tenga filas vacías
    if df is None or df.empty:
        raise ValueError("El DataFrame no puede ser None ni estar vacío")
    
    # Verificar que el DataFrame tenga las columnas necesarias
    for col in ['origen', 'destino', 'costo']:
        if col not in df.columns:
            raise ValueError(f"El DataFrame debe tener la columna '{col}'")
    
    # Verificar que no haya filas con valores nulos
    if df.isnull().any().any():
        raise ValueError("El DataFrame no puede tener valores nulos")
    
    # Crear un grafo dirigido
    G = nx.DiGraph()
    
    # Añadir arcos y sus costos
    G.add_weighted_edges_from(df.values.tolist())
    
    # Obtener las posiciones de los nodos utilizando un layout
    # El layout 'spring' es una buena opción para redes con un número moderado de nodos
    pos = nx.spring_layout(G)
    
    # Dibujar nodos y arcos
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')
    
    # Extraer pesos para etiquetar los arcos
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')
    
    # Mostrar el grafo
    plt.title("Red")
    plt.show()
