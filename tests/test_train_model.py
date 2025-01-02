import pandas as pd
import numpy as np
import unittest
import os
from unittest.mock import patch, mock_open
from src.train_model import calculate_cost_matrix

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

if __name__ == '__main__':
    unittest.main()
