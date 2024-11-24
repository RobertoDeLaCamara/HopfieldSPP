import os
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.data.split_data import split_data
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.models.save_load_model import save_model, load_model
from src.utils.visualizations import plot_metrics

def main():
    """
    Flujo completo del pipeline de ML: carga, preprocesamiento, división de datos,
    entrenamiento, evaluación y visualización.
    """
    # Definir las rutas
    raw_data_path = "data/raw/dataset.csv"
    model_save_path = "models/model.pt"

    # Paso 1: Carga de datos
    print("Cargando datos...")
    data = load_data(raw_data_path, chunksize=100 * (2**20))

    # Paso 2: Preprocesamiento
    print("Preprocesando datos...")
    data_processed = preprocess_data(data)

    # Paso 3: División de datos
    print("Dividiendo los datos...")
    train_data, test_data = split_data(data_processed, test_size=0.2, random_state=42)

    # Paso 4: Entrenamiento del modelo
    print("Entrenando el modelo...")
    model, train_metrics = train_model(train_data)

    # Guardar el modelo entrenado
    print(f"Guardando el modelo en {model_save_path}...")
    save_model(model, model_save_path)

    # Paso 5: Evaluación del modelo
    print("Evaluando el modelo...")
    test_metrics = evaluate_model(model, test_data)

    # Visualización de métricas
    print("Generando visualizaciones de métricas...")
    plot_metrics(train_metrics, test_metrics)

    print("Pipeline completado exitosamente.")

if __name__ == "__main__":
    main()
