# File: test_main.py

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app

class LearnNetworkTestCase(unittest.TestCase):
    def setUp(self):
            self.client = TestClient(app)

    @patch('src.main.pandas.read_csv')
    def test_learn_network_valid_file(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            'source': ['A', 'B', 'C'],
            'target': ['B', 'C', 'A'],
            'weight': [1, 2, 3]
        })

        response = self.client.post('/learnNetwork', files={'file': ('test.csv', b'content', 'text/csv')})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'message': 'Network loaded successfully', 'status': 'success'})

    @patch('src.main.pandas.read_csv')
    def test_learn_network_invalid_file(self, mock_read_csv):
        mock_read_csv.side_effect = Exception('Error reading file')

        response = self.client.post('/learnNetwork', files={'file': ('test.csv', b'content', 'text/csv')})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {'detail': 'Error processing the file: Error reading file'})

    @patch('src.main.pandas.read_csv')
    def test_learn_network_missing_columns(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            'source': ['A', 'B', 'C'],
            'target': ['B', 'C', 'A']
        })

        response = self.client.post('/learnNetwork', files={'file': ('test.csv', b'content', 'text/csv')})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {'detail': 'CSV must contain \'source\', \'target\', and \'weight\' columns.'})


class TestCalculateShortestPath(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.graph = None

    def test_network_not_loaded(self):
        response = self.client.get("/calculateShortestPath?origin=A&destination=B")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Network has not been loaded yet."})

    def test_nodes_not_connected(self):
        self.graph = nx.Graph()
        self.graph.add_node("A")
        self.graph.add_node("B")
        with patch("main.graph", self.graph):
            response = self.client.get("/calculateShortestPath?origin=A&destination=B")
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json(), {"detail": "No path found between the specified nodes."})

    def test_nodes_connected(self):
        self.graph = nx.Graph()
        self.graph.add_edge("A", "B", weight=1)
        with patch("main.graph", self.graph):
            response = self.client.get("/calculateShortestPath?origin=A&destination=B")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"path": ["A", "B"], "distance": 1})

    def test_invalid_origin_node(self):
        self.graph = nx.Graph()
        self.graph.add_node("A")
        with patch("main.graph", self.graph):
            response = self.client.get("/calculateShortestPath?origin=C&destination=A")
            self.assertEqual(response.status_code, 500)
            self.assertIn("Error calculating the shortest path", response.json()["detail"])

    def test_invalid_destination_node(self):
        self.graph = nx.Graph()
        self.graph.add_node("A")
        with patch("main.graph", self.graph):
            response = self.client.get("/calculateShortestPath?origin=A&destination=C")
            self.assertEqual(response.status_code, 500)
            self.assertIn("Error calculating the shortest path", response.json()["detail"])

    def test_internal_server_error(self):
        self.graph = nx.Graph()
        with patch("main.graph", self.graph):
            with patch("networkx.shortest_path", side_effect=Exception("Mock error")):
                response = self.client.get("/calculateShortestPath?origin=A&destination=B")
                self.assertEqual(response.status_code, 500)
                self.assertIn("Error calculating the shortest path", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()


