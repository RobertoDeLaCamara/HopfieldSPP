import pandas as pd
import numpy as np
import os
import pytest
import tensorflow as tf
from src.train_model import calculate_cost_matrix, train_offline_model, HopfieldLayer, HopfieldModel

@pytest.fixture(scope="function")
def setup_synthetic_data():
    os.makedirs('../data/synthetic', exist_ok=True)
    yield
    import shutil
    shutil.rmtree('../data/synthetic')

# Tests for calculate_cost_matrix
def test_valid_csv_file(setup_synthetic_data):
    data = {'origin': ['A', 'A', 'B'], 'destination': ['B', 'C', 'C'], 'weight': [1, 2, 3]}
    df = pd.DataFrame(data)
    df.to_csv('../data/synthetic/test_network.csv', index=False)

    cost_matrix = calculate_cost_matrix('../data/synthetic/test_network.csv')
    expected_cost_matrix = np.array([[0, 1, 2], [np.inf, 0, 3], [np.inf, np.inf, 0]])

    np.testing.assert_array_equal(cost_matrix, expected_cost_matrix)

def test_empty_csv_file(setup_synthetic_data):
    open('../data/synthetic/test_network.csv', 'w').close()

    with pytest.raises(SystemExit):
        calculate_cost_matrix('../data/synthetic/test_network.csv')

def test_invalid_csv_file(setup_synthetic_data):
    data = {'origin': ['A', 'A', 'B'], 'destination': ['B', 'C', 'C'], 'weight': [1, 'a', 3]}
    df = pd.DataFrame(data)
    df.to_csv('../data/synthetic/test_network.csv', index=False)

    with pytest.raises(SystemExit):
        calculate_cost_matrix('../data/synthetic/test_network.csv')

# Tests for train_offline_model
def test_train_offline_model_valid_adjacency_matrix(setup_synthetic_data):
    """Test train_offline_model with a valid adjacency matrix CSV file."""
    network_data = {'origin': ['A', 'A', 'B'], 'destination': ['B', 'C', 'C'], 'weight': [1, 2, 3]}
    network_df = pd.DataFrame(network_data)
    adjacency_matrix_file = '../data/synthetic/test_network.csv'
    network_df.to_csv(adjacency_matrix_file, index=False)
    
    try:
        train_offline_model(adjacency_matrix_file,test=True)
    except FileNotFoundError:
        pytest.fail("File not found. Ensure the file path is correct.")
    except pd.errors.EmptyDataError:
        pytest.fail("CSV file is empty or invalid.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")


# Tests for HopfieldLayer
def test_hopfield_layer_init():
    n = 10
    distance_matrix = np.random.rand(n, n)
    layer = HopfieldLayer(n, distance_matrix)

    assert layer.n == n
    assert layer.distance_matrix.shape == (n, n)

def test_hopfield_layer_energy():
    n = 10
    distance_matrix = np.random.rand(n, n)
    layer = HopfieldLayer(n, distance_matrix)
    energy = layer.energy()

    assert isinstance(energy, tf.Tensor)

# Tests for HopfieldModel
def test_hopfield_model_init():
    n = 10
    distance_matrix = np.random.rand(n, n)
    model = HopfieldModel(n, distance_matrix)

    assert model.hopfield_layer.n == n
    assert model.hopfield_layer.distance_matrix.shape == (n, n)

def test_hopfield_model_train_step():
    """Test that the custom train_step function of the HopfieldModel works correctly."""
    num_nodes = 10
    distance_matrix = tf.random.uniform((num_nodes, num_nodes))
    model = HopfieldModel(num_nodes, distance_matrix)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    dummy_target = tf.zeros((1, num_nodes, num_nodes), dtype=tf.float32)
    history = model.fit(dummy_target, epochs=10)

    assert isinstance(history.history["loss"], list)

def test_hopfield_model_predict():
    n = 10
    distance_matrix = np.random.rand(n, n)
    model = HopfieldModel(n, distance_matrix)
    source, destination = 0, 1
    path = model.predict(source, destination)

    assert isinstance(path, list)
