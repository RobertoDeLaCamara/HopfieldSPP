import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Carga datos desde un archivo CSV de manera eficiente."""
    # leer los datos con un buffer para evitar sobrecarga
    # de memoria al leer grandes datasets
    buffer_size = 100 * (2**20)  # 100MB
    return pd.read_csv(
        file_path,
        chunksize=buffer_size,
        low_memory=False
    )
