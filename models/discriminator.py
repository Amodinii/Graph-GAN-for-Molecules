import tensorflow as tf
from tensorflow.keras import layers
from utils.layers import GINConv  # Import new GIN layer

class Discriminator(tf.keras.Model):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.max_nodes = config.MAX_NODES
        self.node_dim = config.NODE_DIM
        self.edge_dim = config.EDGE_DIM
        self.noise_stddev = 0.1  # Gaussian noise for stability

        # GIN Layers (Replaces GCN)
        self.gin1 = GINConv(self.hidden_dim)
        self.gin2 = GINConv(self.hidden_dim)
        self.gin3 = GINConv(self.hidden_dim)

        # Fully Connected Layers (Classification)
        self.dense1 = layers.Dense(self.hidden_dim, activation=tf.nn.relu)
        self.dense2 = layers.Dense(1, activation=None)  # Raw score output (no activation for WGAN-GP)

        # Layer normalization for stability
        self.layer_norm = layers.LayerNormalization()

    def call(self, adjacency_tensor, node_tensor, training=True):
        """Forward pass for the Discriminator using GIN"""
        if training:
            node_tensor += tf.random.normal(tf.shape(node_tensor), mean=0.0, stddev=self.noise_stddev)

        # Apply GIN layers
        h = self.gin1(adjacency_tensor, node_tensor)
        h = self.gin2(adjacency_tensor, h)
        h = self.gin3(adjacency_tensor, h)

        # Global sum pooling to obtain graph-level representation
        graph_embedding = tf.reduce_sum(h, axis=1)

        # Fully connected layers for final prediction
        x = self.dense1(graph_embedding)
        x = self.layer_norm(x)  # Normalize activations
        realism_score = self.dense2(x)  # WGAN-GP requires raw scores (not probabilities)

        return realism_score  # Higher score = more realistic molecule
