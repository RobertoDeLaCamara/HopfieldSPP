import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Realiza el preprocesamiento básico de los datos."""
    # Manejo de valores nulos de manera más eficiente
    return data.fillna(value=0, inplace=False)
