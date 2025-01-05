import pandas as pd
import numpy as np
import tensorflow as tf
from math import sqrt
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import register_keras_serializable

def calculate_cost_matrix(adjacency_matrix):   
    """
    Calculates the cost matrix for an adjacency matrix of a network given in a CSV file.

    The CSV file should have columns 'origin', 'destination', and 'weight'. The function
    will exit with an error message if the file is not found, is empty, or is
    invalid in any way.

    The cost matrix is a matrix where the element at row i and column j
    represents the cost of going from node i to node j. The diagonal of the
    matrix is set to 0, representing the cost of going from a node to itself.

    Args:
        adjacency_matrix (str): The path to the CSV file containing the adjacency matrix.
    Returns:
        cost_matrix: A 2D numpy array representing the cost matrix.

    """
    # Load data from CSV file
    try:
        df = pd.read_csv(adjacency_matrix, usecols=['origin', 'destination', 'weight'])
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        exit()
    except pd.errors.EmptyDataError:
        print("Error: The file is empty or invalid.")
        exit()

    # Identify unique nodes and create a mapping for indices
    nodos = pd.unique(df[['origin', 'destination']].values.ravel())
    node_to_index = pd.Series(range(len(nodos)), index=nodos)
    n = len(nodos)

    # Initialize the cost matrix with infinity
    cost_matrix = np.full((n, n), np.inf, dtype=float)

    # Set diagonal to 0 (self-costs)
    np.fill_diagonal(cost_matrix, 0)

    # Fill the cost matrix with the values from the CSV
    for _, row in df.iterrows():
        try:
            origen = row['origin']
            destino = row['destination']
            costo = float(row['weight'])
            cost_matrix[node_to_index[origen], node_to_index[destino]] = costo
        except KeyError:
            print("Error: Missing columns 'origen', 'destino', or 'costo'.")
            exit()
        except ValueError:
            print(f"Error: Invalid cost value on row {_}.")
            exit()
    return cost_matrix

# Define the Hopfield Neural Network layer
@register_keras_serializable()
class HopfieldLayer(Layer):
    def __init__(self, n, distance_matrix, **kwargs):
        '''
        Initializes the HopfieldLayer.
            
        Args:
            n (int): Number of nodes in the graph.
            distance_matrix (numpy array): Distance matrix of the graph.
            **kwargs: Additional keyword arguments for the parent class.
        '''
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
        # Valid arcs
        self.valid_arcs = tf.constant((self.distance_matrix.numpy() != 0).astype(np.float32), dtype=tf.float32)
                                         
    def energy(self):
        '''
        Computes the energy of the network, which is a function of the path cost, row and column constraints, binary constraint, and invalid arcs penalty.
            
        The energy is given by the sum of the following terms:
            
        1. Path Cost: The sum of the costs of the arcs in the path, where the cost of each arc is the product of the arc's weight and the corresponding element in the state matrix.
        2. Row Constraint: The sum of the squares of the differences between the sum of each row of the state matrix and 1.
        3. Column Constraint: The sum of the squares of the differences between the sum of each column of the state matrix and 1.
        4. Binary Constraint: The sum of the squares of the differences between the state matrix and its binary version (i.e., the state matrix where each element is 0 or 1).
        5. Invalid Arcs Penalty: The sum of the squares of the differences between the state matrix and the valid arcs matrix (i.e., the matrix where each element is 0 if the arc is invalid and 1 otherwise).
            
        The energy is used to fine-tune the state matrix with constraints.
        '''
        # Hyperparameters (weights for energy terms)
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
        '''
        Fine-tunes the state matrix with constraints, given a source and destination node, over a specified number of iterations.
            
        The energy of the network is computed with the addition of two extra terms, which are the sum of the squares of the differences between the sum of the outgoing arcs of the source node and 1, and the sum of the squares of the differences between the sum of the incoming arcs of the destination node and 1.
            
        The state matrix is then updated using gradient descent with the optimizer specified in the constructor, and the variables are clipped to [0, 1] to enforce valid range.
            
        The energy is printed every 100 iterations, and the final energy is printed at the end.
            
        Returns the optimized state matrix.
        '''
        print("Initial Energy:", self.energy().numpy())
        for i in range(iterations):
                with tf.GradientTape() as tape:
                    # Fine-tune with source and destination constraints
                    source_out = tf.reduce_sum(self.x[source, :]) - 1
                    term5 = (10.0 / 2) * tf.square(source_out)
                    dest_in = tf.reduce_sum(self.x[:, destination]) - 1
                    term6 = (10.0 / 2) * tf.square(dest_in)
                    energy = self.energy() + term5 + term6

                gradients = tape.gradient(energy, [self.x])
                self.optimizer.apply_gradients(zip(gradients, [self.x]))
                # Clip the variables to [0, 1] to enforce valid range
                self.x.assign(tf.clip_by_value(self.x, 0.0, 1.0))
                if i % 100 == 0:
                    print(f"Fine-Tuning Iteration {i}, Energy: {energy.numpy()}")
        print("Final Energy:", self.energy().numpy())
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
            instance.x.assign(tf.constant(x, dtype=tf.float32))  # Restore self.x from the configuration
            valid_arcs = config.pop('valid_arcs')
            instance.valid_arcs.assign(tf.constant(valid_arcs, dtype=tf.float32))  # Restore self.valid_arcs from the configuration
            return instance

#Integrate custom HopfieldLayer with Keras model
@register_keras_serializable()
class HopfieldModel(Model):
    def __init__(self, n, distance_matrix, **kwargs):
            '''
            Initializes the HopfieldModel.

            Args:
                n (int): Number of nodes in the graph.
                distance_matrix (numpy array): Distance matrix of the graph.
                **kwargs: Additional keyword arguments for the parent class.
            '''
            super(HopfieldModel, self).__init__(**kwargs)
            self.hopfield_layer = HopfieldLayer(n, distance_matrix)
            self.optimizer = tf.optimizers.Adam(learning_rate=0.01)

    def train_step(self, data):
            # Custom training logic
            """
            Performs a single training step for the HopfieldModel.

            This method utilizes TensorFlow's `GradientTape` to compute gradients of the energy function,
            which serves as the loss, with respect to the model's trainable variables. The computed gradients
            are then used to update the model's parameters via the optimizer.

            Args:
                data: Unused in this implementation, but typically a batch of input data.

            Returns:
                A dictionary containing the loss value under the key "loss".
            """

            with tf.GradientTape() as tape:
                # Compute the energy as the loss
                loss = self.hopfield_layer.energy()
            # Compute gradients and apply updates
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            # Return the loss
            return {"loss": loss}
    
    def call(self, inputs, training=False):
            # Forward pass
            return self.hopfield_layer(inputs, training=training)
    
    def predict(self,source,destination):
            '''
            Predicts the shortest path from a source node to a destination node using the Hopfield network.

            The method first fine-tunes the state matrix with constraints using the source and destination nodes. The optimized state matrix is then used to extract the shortest path by starting from the source node and iteratively finding the next node with the highest probability of being part of the shortest path.

            Args:
                source (int): The source node.
                destination (int): The destination node.

            Returns:
                A list of node indices representing the shortest path from the source node to the destination node.
            '''
            self.hopfield_layer.fine_tune_with_constraints(source, destination)
            # Retrieve the optimized state matrix
            state_matrix = tf.round(self.hopfield_layer.x).numpy()
            # Extract the shortest path based on the state matrix
            def extract_path(state_matrix):
                path = []
                current_node = source
                visited = set()
                while current_node != destination:
                    path.append(current_node)
                    visited.add(current_node)
                    # Find the next node
                    next_node = np.argmax(state_matrix[current_node])
                    if next_node in visited or next_node == destination:
                        break
                    current_node = next_node

                path.append(destination)
                return path

            # Extract and return the predicted path
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

def train_offline_model(adjacency_matrix,test=False):
    """
    Trains a Hopfield Neural Network model to solve the shortest path problem using a given distance matrix.
    The function performs the following steps:
    1. Loads, normalizes, and flattens the distance matrix, assigning large values to infinity values.
    2. Defines a custom HopfieldLayer class that computes the energy of the network and fine-tunes the state matrix with constraints.
    3. Integrates the custom HopfieldLayer with a Keras model (HopfieldModel) and defines custom training and prediction logic.
    4. Compiles and trains the model to minimize the energy function.
    5. Saves the trained model to a specified file path.
    Args:
        adjacency_matrix (str): The path to the CSV file containing the adjacency matrix.
    Returns:
        None
    """
    cost_matrix = calculate_cost_matrix(adjacency_matrix)
    df = pd.read_csv(adjacency_matrix)
    # Load, normalize and flatten the distance matrix, assigning a large value to infinity values
    cost_matrix = np.array(df.pivot(index='origin', columns='destination', values='weight').fillna(1e6))
    cost_matrix[cost_matrix == np.inf] = 1e12
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)
    cost_matrix_flat = cost_matrix_normalized.flatten()
    distance_matrix = cost_matrix_flat

    # Number of nodes
    n = distance_matrix.shape[0]
                     
    # Create the model
    model = HopfieldModel(n, distance_matrix)
    # Compile the model with a custom optimizer
    print("Compiling the model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    print("Model compiled.")

    # Ensure distance_matrix is a valid tensor and reshape it to match the expected input shape
    m = int(sqrt(n))
    distance_matrix_tensor = tf.constant(distance_matrix, dtype=tf.float32)
    distance_matrix_tensor = tf.reshape(distance_matrix_tensor, (m, m))  
    distance_matrix_tensor = tf.reshape(distance_matrix_tensor, (1, m, m))  
    # Create dummy target data as it is required by the fit method
    dummy_target = tf.zeros((1, n, n), dtype=tf.float32)
    # Train the model to minimize the energy function
    model(dummy_target)
    model.fit(dummy_target, epochs=1000)
    model.summary()
    # Save the trained model
    if not test:
        model.save("../models/trained_model_without_source_dest.keras")
