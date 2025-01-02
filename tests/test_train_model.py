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
    df.to_csv('../data/synthetic/network.csv', index=False)

    cost_matrix = calculate_cost_matrix('../data/synthetic/network.csv')
    expected_cost_matrix = np.array([[0, 1, 2], [np.inf, 0, 3], [np.inf, np.inf, 0]])

    np.testing.assert_array_equal(cost_matrix, expected_cost_matrix)

def test_empty_csv_file(setup_synthetic_data):
    open('../data/synthetic/network.csv', 'w').close()

    with pytest.raises(SystemExit):
        calculate_cost_matrix('../data/synthetic/network.csv')

def test_invalid_csv_file(setup_synthetic_data):
    data = {'origin': ['A', 'A', 'B'], 'destination': ['B', 'C', 'C'], 'weight': [1, 'a', 3]}
    df = pd.DataFrame(data)
    df.to_csv('../data/synthetic/network.csv', index=False)

    with pytest.raises(SystemExit):
        calculate_cost_matrix('../data/synthetic/network.csv')

# Tests for train_offline_model
def test_valid_adjacency_matrix(setup_synthetic_data):
    data = {'origin': ['A', 'A', 'B'], 'destination': ['B', 'C', 'C'], 'weight': [1, 2, 3]}
    df = pd.DataFrame(data)
    df.to_csv('../data/synthetic/network.csv', index=False)
    train_offline_model('../data/synthetic/network.csv')
    assert True  # Replace with an actual assertion

def test_invalid_adjacency_matrix():
    adjacency_matrix_file = "path/to/invalid/adjacency/matrix.csv"
    with pytest.raises(FileNotFoundError):
        train_offline_model(adjacency_matrix_file)

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

def test_hopfield_layer_fine_tune():
    n = 10
    distance_matrix = np.random.rand(n, n)
    layer = HopfieldLayer(n, distance_matrix)
    source, destination = 0, 1
    layer.fine_tune_with_constraints(source, destination)

    assert isinstance(layer.x, tf.Tensor)

# Tests for HopfieldModel
def test_hopfield_model_init():
    n = 10
    distance_matrix = np.random.rand(n, n)
    model = HopfieldModel(n, distance_matrix)

    assert model.hopfield_layer.n == n
    assert model.hopfield_layer.distance_matrix.shape == (n, n)

def test_hopfield_model_train():
    n = 10
    distance_matrix = np.random.rand(n, n)
    model = HopfieldModel(n, distance_matrix)
    dummy_target = tf.zeros((1, n, n), dtype=tf.float32)
    model.fit(dummy_target, epochs=10)

    assert isinstance(model.loss, tf.Tensor)

def test_hopfield_model_predict():
    n = 10
    distance_matrix = np.random.rand(n, n)
    model = HopfieldModel(n, distance_matrix)
    source, destination = 0, 1
    path = model.predict(source, destination)

    assert isinstance(path, list)
