import pandas as pd
import numpy as np
import tensorflow as tf
from math import sqrt
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_cost_matrix(adjacency_matrix):   
    """
    Calculates the cost matrix for an adjacency matrix of a network given in a CSV file.

    The CSV file should have columns 'origin', 'destination', and 'weight'. The function
    will exit with an error message if the file is not found, is empty, or is
    invalid in any way.

    Args:
        adjacency_matrix (str): The path to the CSV file containing the adjacency matrix.
    Returns:
        cost_matrix: A 2D numpy array representing the cost matrix.

    """
    try:
        df = pd.read_csv(adjacency_matrix, usecols=['origin', 'destination', 'weight'])
    except FileNotFoundError:
        logger.error("File not found. Please check the file path.")
        exit()
    except pd.errors.EmptyDataError:
        logger.error("The file is empty or invalid.")
        exit()

    nodos = pd.unique(df[['origin', 'destination']].values.ravel())
    node_to_index = pd.Series(range(len(nodos)), index=nodos)
    n = len(nodos)

    cost_matrix = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(cost_matrix, 0)

    for _, row in df.iterrows():
        try:
            origen = row['origin']
            destino = row['destination']
            costo = float(row['weight'])
            cost_matrix[node_to_index[origen], node_to_index[destino]] = costo
        except KeyError:
            logger.error("Missing columns 'origen', 'destino', or 'costo'.")
            exit()
        except ValueError:
            logger.error(f"Invalid cost value on row {_}.")
            exit()
    return cost_matrix

@register_keras_serializable()
class HopfieldLayer(Layer):
    def __init__(self, n, distance_matrix, **kwargs):
        super(HopfieldLayer, self).__init__(**kwargs)
        self.n = n
        self.distance_matrix = tf.constant(distance_matrix, dtype=tf.float32)
        self.x = self.add_weight(
            name="x", 
            shape=(n, n), 
            initializer="random_uniform", 
            trainable=True
        )
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)
        self.valid_arcs = tf.constant((self.distance_matrix.numpy() != 0).astype(np.float32), dtype=tf.float32)
                                         
    def energy(self):
        mu1 = 1.0
        mu2 = 10.0
        mu3 = 10.0
        path_cost = tf.reduce_sum(self.distance_matrix * self.x)
        row_constraint = tf.reduce_sum(tf.square(tf.reduce_sum(self.x, axis=1) - 1))
        col_constraint = tf.reduce_sum(tf.square(tf.reduce_sum(self.x, axis=0) - 1))
        binary_constraint = tf.reduce_sum(tf.square(self.x * (1 - self.x)))
        invalid_arcs_penalty = tf.reduce_sum(tf.square(self.x * (1 - self.valid_arcs)))
        return (mu1/2)*path_cost + (mu2/2)*row_constraint + (mu2/2)*col_constraint + (mu3/2)*binary_constraint + 1000*invalid_arcs_penalty	
            
    def fine_tune_with_constraints(self, source, destination, iterations=500):
        logger.info("Initial Energy: %s", self.energy().numpy())
        for i in range(iterations):
            with tf.GradientTape() as tape:
                source_out = tf.reduce_sum(self.x[source, :]) - 1
                term5 = (10.0 / 2) * tf.square(source_out)
                dest_in = tf.reduce_sum(self.x[:, destination]) - 1
                term6 = (10.0 / 2) * tf.square(dest_in)
                energy = self.energy() + term5 + term6

            gradients = tape.gradient(energy, [self.x])
            self.optimizer.apply_gradients(zip(gradients, [self.x]))
            self.x.assign(tf.clip_by_value(self.x, 0.0, 1.0))
            if i % 100 == 0:
                logger.info("Fine-Tuning Iteration %d, Energy: %s", i, energy.numpy())
        logger.info("Final Energy: %s", self.energy().numpy())
        return self.x

    def call(self, inputs, training=False):
        return self.x

    def compile(self, optimizer):
        super(HopfieldLayer, self).compile()

    def get_config(self):
        config = super(HopfieldLayer, self).get_config()
        config.update({
            'n': self.n,
            'distance_matrix': self.distance_matrix.numpy().tolist(),
            'x': self.x.numpy().tolist(),
            'valid_arcs': self.valid_arcs.numpy().tolist() 
        })
        return config

    @classmethod
    def from_config(cls, config):
        x = config.pop('x')
        distance_matrix = config.pop('distance_matrix')
        instance = cls(distance_matrix=distance_matrix, **config)
        instance.x.assign(tf.constant(x, dtype=tf.float32))
        valid_arcs = config.pop('valid_arcs')
        instance.valid_arcs.assign(tf.constant(valid_arcs, dtype=tf.float32))
        return instance

@register_keras_serializable()
class HopfieldModel(Model):
    def __init__(self, n, distance_matrix, **kwargs):
        super(HopfieldModel, self).__init__(**kwargs)
        self.hopfield_layer = HopfieldLayer(n, distance_matrix)
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)
        self.cost_matrix = None

    def set_cost_matrix(self, cost_matrix):
        self.cost_matrix = cost_matrix
        
    def get_cost_matrix(self):
        return self.cost_matrix
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.hopfield_layer.energy()
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}
    
    def call(self, inputs, training=False):
        return self.hopfield_layer(inputs, training=training)
    
    def predict(self,source,destination):
        self.hopfield_layer.fine_tune_with_constraints(source, destination)
        state_matrix = tf.round(self.hopfield_layer.x).numpy()
        
        def extract_path(state_matrix):
            path = []
            current_node = source
            visited = set()
            while current_node != destination:
                path.append(current_node)
                visited.add(current_node)
                next_node = np.argmax(state_matrix[current_node])
                if next_node in visited or next_node == destination:
                    break
                current_node = next_node

            path.append(destination)
            return path

        return extract_path(state_matrix)
       
    def get_config(self):
        config = super(HopfieldModel, self).get_config()
        config.update({
            'n': self.hopfield_layer.n,
            'distance_matrix': self.hopfield_layer.distance_matrix.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def train_offline_model(adjacency_matrix_path: str) -> None:
    logger.info("Training offline model")
    logger.info("Adjacency matrix path: %s", adjacency_matrix_path)
    try:
        df = pd.read_csv(adjacency_matrix_path)
        if df.empty:
            raise ValueError("Adjacency matrix file is empty or invalid")
    except Exception as e:
        logger.error("Error loading adjacency matrix: %s", str(e))
        raise

    logger.info("Calculating cost matrix")
    try:
        cost_matrix = calculate_cost_matrix(adjacency_matrix_path)
    except Exception as e:
        logger.error("Error calculating cost matrix: %s", str(e))
        raise

    cost_matrix = np.array(df.pivot(index='origin', columns='destination', values='weight').fillna(1e6))
    cost_matrix[cost_matrix == np.inf] = 1e12
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)
    distance_matrix = cost_matrix_normalized.flatten()

    n = distance_matrix.shape[0]
    logger.info("Number of nodes: %d", n)

    logger.info("Creating model")
    model = HopfieldModel(n, distance_matrix)
    logger.info("Setting cost matrix")
    logger.debug(cost_matrix)
    model.set_cost_matrix(cost_matrix)
    logger.debug(model.get_cost_matrix())
    logger.info("Compiling model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

    m = int(sqrt(n))
    distance_matrix_tensor = tf.constant(distance_matrix, dtype=tf.float32)
    distance_matrix_tensor = tf.reshape(distance_matrix_tensor, (m, m))  
    distance_matrix_tensor = tf.reshape(distance_matrix_tensor, (1, m, m))  
    dummy_target = tf.zeros((1, n, n), dtype=tf.float32)

    logger.info("Training model")
    model(dummy_target)

    if 'PYTEST_CURRENT_TEST' in os.environ:
        logger.info("Function is being called by a pytest test")
        model_save_path = "../data/synthetic/tests/"
    else:
        logger.info("Function is being called during real execution")
        model_save_path = "../models/"

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    logger.info("Saving model to: %s", model_save_path)  
    model.save(model_save_path +'trained_model.keras')

    with open(model_save_path + 'cost_matrix.pkl', 'wb') as f:
        pickle.dump(cost_matrix, f)    
    logger.info("Cost matrix saved")

    with open(model_save_path + 'cost_matrix.pkl', 'rb') as f:
        loaded_cost_matrix = pickle.load(f)

    if np.array_equal(loaded_cost_matrix, cost_matrix):
        logger.info("Cost matrix has been correctly saved and verified.")
    else:
        logger.error("Mismatch in the saved cost matrix.")

    if os.path.exists(model_save_path):
        logger.info("Model successfully saved to %s", model_save_path)
    else:
        logger.error("Failed to save the model to %s", model_save_path)
