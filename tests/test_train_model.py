import pandas as pd
import numpy as np
import unittest
import os
import tensorflow as tf
from unittest.mock import patch, mock_open
from src.train_model import calculate_cost_matrix
from src.train_model import train_offline_model, HopfieldLayer, HopfieldModel

class TestCalculateCostMatrix(unittest.TestCase):
    def setUp(self):
        # Ensure the directory exists
        os.makedirs('../data/synthetic', exist_ok=True)

    def tearDown(self):
        # Clean up the directory after tests
        import shutil
        shutil.rmtree('../data/synthetic')
        
    def test_valid_csv_file(self):
        # Create a test CSV file
        data = {'origin': ['A', 'A', 'B'], 'destination': ['B', 'C', 'C'], 'weight': [1, 2, 3]}
        df = pd.DataFrame(data)
        df.to_csv('../data/synthetic/network.csv', index=False)

        # Call the function
        cost_matrix = calculate_cost_matrix('../data/synthetic/network.csv')

        # Check the result
        expected_cost_matrix = np.array([[0, 1, 2], [np.inf, 0, 3], [np.inf, np.inf, 0]])
        np.testing.assert_array_equal(cost_matrix, expected_cost_matrix)

    def test_empty_csv_file(self):
        # Create an empty test CSV file
        open('../data/synthetic/network.csv', 'w').close()

        # Call the function
        with self.assertRaises(SystemExit):
            calculate_cost_matrix('../data/synthetic/network.csv')

    def test_invalid_csv_file(self):
        # Create a test CSV file with invalid data
        data = {'origin': ['A', 'A', 'B'], 'destination': ['B', 'C', 'C'], 'weight': [1, 'a', 3]}
        df = pd.DataFrame(data)
        df.to_csv('../data/synthetic/network.csv', index=False)

        # Call the function
        with self.assertRaises(SystemExit):
            calculate_cost_matrix('../data/synthetic/network.csv')


class TestTrainOfflineModel(unittest.TestCase):
    def test_valid_adjacency_matrix(self):
         # Create a test CSV file
        data = {'origin': ['A', 'A', 'B'], 'destination': ['B', 'C', 'C'], 'weight': [1, 2, 3]}
        df = pd.DataFrame(data)
        df.to_csv('../data/synthetic/network.csv', index=False)
        train_offline_model('../data/synthetic/network.csv')
        self.assertTrue(True)  # Replace with actual assertion

    def test_invalid_adjacency_matrix(self):
        adjacency_matrix_file = "path/to/invalid/adjacency/matrix.csv"
        with self.assertRaises(FileNotFoundError):
            train_offline_model(adjacency_matrix_file)

class TestHopfieldLayer(unittest.TestCase):
    def test_init(self):
        n = 10
        distance_matrix = np.random.rand(n, n)
        layer = HopfieldLayer(n, distance_matrix)
        self.assertEqual(layer.n, n)
        self.assertEqual(layer.distance_matrix.shape, (n, n))

    def test_energy(self):
        n = 10
        distance_matrix = np.random.rand(n, n)
        layer = HopfieldLayer(n, distance_matrix)
        energy = layer.energy()
        self.assertIsInstance(energy, tf.Tensor)

    def test_fine_tune(self):
        n = 10
        distance_matrix = np.random.rand(n, n)
        layer = HopfieldLayer(n, distance_matrix)
        source = 0
        destination = 1
        layer.fine_tune_with_constraints(source, destination)
        self.assertIsInstance(layer.x, tf.Tensor)

class TestHopfieldModel(unittest.TestCase):
    def test_init(self):
        n = 10
        distance_matrix = np.random.rand(n, n)
        model = HopfieldModel(n, distance_matrix)
        self.assertEqual(model.hopfield_layer.n, n)
        self.assertEqual(model.hopfield_layer.distance_matrix.shape, (n, n))

    def test_train(self):
        n = 10
        distance_matrix = np.random.rand(n, n)
        model = HopfieldModel(n, distance_matrix)
        dummy_target = tf.zeros((1, n, n), dtype=tf.float32)
        model.fit(dummy_target, epochs=10)
        self.assertIsInstance(model.loss, tf.Tensor)

    def test_predict(self):
        n = 10
        distance_matrix = np.random.rand(n, n)
        model = HopfieldModel(n, distance_matrix)
        source = 0
        destination = 1
        path = model.predict(source, destination)
        self.assertIsInstance(path, list)


if __name__ == '__main__':
    unittest.main()
