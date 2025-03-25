import tensorflow as tf
from tensorflow.keras import layers

class Encoder(tf.keras.Model):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.hidden_dim = config.HIDDEN_DIM
        self.latent_dim = config.LATENT_DIM
        self.max_nodes = config.MAX_NODES

        # Graph Convolutional Layers
        self.gcn1 = layers.Dense(self.hidden_dim, activation=tf.nn.relu)
        self.gcn2 = layers.Dense(self.hidden_dim, activation=tf.nn.relu)

        # Fully Connected Layers to map to latent space
        self.fc_mu = layers.Dense(self.latent_dim)  # Mean
        self.fc_logvar = layers.Dense(self.latent_dim)  # Log variance for reparam trick

    def call(self, adjacency, nodes):
        """
        Forward pass of the encoder
        :param adjacency: Adjacency matrix of shape (batch_size, max_nodes, max_nodes)
        :param nodes: Node features of shape (batch_size, max_nodes, node_dim)
        :return: Latent vector (batch_size, latent_dim)
        """
        x = self.gcn1(nodes)  # Apply first GCN layer
        x = tf.matmul(adjacency, x)  # Aggregate neighbor features
        x = self.gcn2(x)  # Apply second GCN layer
        x = tf.matmul(adjacency, x)  # Aggregate again

        # Compute mean and log variance for latent representation
        mu = self.fc_mu(tf.reduce_mean(x, axis=1))  # Global pooling
        logvar = self.fc_logvar(tf.reduce_mean(x, axis=1))  # Global pooling

        # Reparameterization Trick for Variational Autoencoder (VAE)
        epsilon = tf.random.normal(shape=tf.shape(mu))
        latent_vector = mu + tf.exp(0.5 * logvar) * epsilon

        return latent_vector, mu, logvar
