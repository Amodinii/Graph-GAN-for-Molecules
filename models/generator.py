import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

        self.latent_dim = config.LATENT_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.max_nodes = config.MAX_NODES
        self.node_dim = config.NODE_DIM
        self.edge_dim = config.EDGE_DIM

        # Define network layers
        self.dense1 = layers.Dense(self.hidden_dim, activation=tf.nn.relu)
        self.dense2 = layers.Dense(self.hidden_dim, activation=tf.nn.relu)

        # Graph-based decoding layers
        self.node_decoder = layers.Dense(self.max_nodes * self.node_dim)  
        self.edge_decoder = layers.Dense(self.max_nodes * self.max_nodes * self.edge_dim)

        # Dropout layer for regularization
        self.dropout = layers.Dropout(0.2)  

    def gumbel_softmax(self, logits, tau=1.0):
        """Differentiable sampling via Gumbel-Softmax."""
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1) + 1e-8) + 1e-8)
        y = logits + gumbel_noise
        return tf.nn.softmax(y / tau)

    def apply_valency_constraints(self, edge_probs, node_probs):
        """Ensure valency constraints are respected."""
        valid_edge_probs = edge_probs  # Keep it as a TensorFlow tensor

        # Allowed valencies for each atom type (C, N, O, F)
        max_valencies = tf.constant([4.0, 3.0, 2.0, 1.0], dtype=tf.float32)

        for i in range(self.config.MAX_NODES):
            atom_type = tf.argmax(node_probs[:, i, :], axis=-1)  # Get predicted atom type (batch_size,)

            # Use tf.gather to safely index tensor values
            max_valency = tf.gather(max_valencies, atom_type)  # Shape: (batch_size,)

            # Compute total bond order for each atom
            total_bond_order = tf.reduce_sum(valid_edge_probs[:, i, :, :], axis=(1, 2))  # Shape: (batch_size,)

            # Compute scaling factor to enforce valency constraints
            mask = total_bond_order > max_valency
            scaling_factor = tf.where(mask, max_valency / (total_bond_order + 1e-8), tf.ones_like(total_bond_order))

            # Corrected `expand_dims` calls (only single dimension at a time)
            scaling_factor = tf.expand_dims(scaling_factor, axis=1)  # (batch_size, 1)
            scaling_factor = tf.expand_dims(scaling_factor, axis=2)  # (batch_size, 1, 1)
            scaling_factor = tf.expand_dims(scaling_factor, axis=3)  # (batch_size, 1, 1, 1)

            valid_edge_probs *= scaling_factor  # Ensure valid valency

        return valid_edge_probs  # Return updated adjacency tensor

    def call(self, z, tau=1.0, training=True):
        """Forward pass of the generator with optional Gumbel-Softmax sampling."""
        
        # Add Gaussian noise for robustness
        if training:
            noise = tf.random.normal(shape=tf.shape(z), mean=0.0, stddev=0.2)
            z = z + noise  

        x = self.dense1(z)
        x = self.dense2(x)
        x = self.dropout(x, training=training)

        # **Gumbel-Softmax Sampling for Node Features**
        node_logits = tf.reshape(self.node_decoder(x), (-1, self.config.MAX_NODES, self.config.NODE_DIM))
        node_probs = self.gumbel_softmax(node_logits, tau)

        # **Thresholded Adjacency Matrix for Connectivity**
        edge_logits = tf.reshape(self.edge_decoder(x), (-1, self.config.MAX_NODES, self.config.MAX_NODES, self.config.EDGE_DIM))
        edge_probs = tf.nn.sigmoid(edge_logits)  # Ensure valid probability values

        # Apply valency constraints
        edge_probs = self.apply_valency_constraints(edge_probs, node_probs)

        return edge_probs, node_probs
