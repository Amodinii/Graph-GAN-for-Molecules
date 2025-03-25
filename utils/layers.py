import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class GraphConvolution(Layer):
    """Graph Convolutional Layer (GCN) used in Discriminator & Reward Network"""
    def __init__(self, output_dim, activation=tf.nn.relu):
        super(GraphConvolution, self).__init__()
        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(feature_dim, self.output_dim),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        adjacency_matrix, node_features = inputs
        aggregated_features = tf.matmul(adjacency_matrix, node_features)
        transformed_features = tf.matmul(aggregated_features, self.kernel)
        return self.activation(transformed_features)

class GraphAggregation(Layer):
    """Global Sum Pooling for graph-level representation"""
    def __init__(self):
        super(GraphAggregation, self).__init__()

    def call(self, node_features):
        return tf.reduce_sum(node_features, axis=1)  # Sum over nodes

class TransposedGraphConvolution(Layer):
    """Transposed GCN layer used in Generator to create adjacency matrices"""
    def __init__(self, output_dim, activation=tf.nn.sigmoid):
        super(TransposedGraphConvolution, self).__init__()
        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(latent_dim, self.output_dim),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, latent_vector):
        adjacency_matrix = tf.matmul(latent_vector, self.kernel)
        adjacency_matrix = tf.matmul(adjacency_matrix, tf.transpose(latent_vector, perm=[0, 2, 1]))
        return self.activation(adjacency_matrix)

class MLP(Layer):
    """Fully Connected MLP used in all networks"""
    def __init__(self, units, activation=tf.nn.relu):
        super(MLP, self).__init__()
        self.dense = Dense(units, activation=activation)

    def call(self, inputs):
        return self.dense(inputs)

class RelationalGraphConvolution(Layer):
    """Graph Convolution Layer that incorporates edge types."""
    def __init__(self, output_dim, num_edge_types, activation=tf.nn.relu):
        super(RelationalGraphConvolution, self).__init__()
        self.output_dim = output_dim
        self.num_edge_types = num_edge_types
        self.activation = activation

    def build(self, input_shape):
        feature_dim = input_shape[1][-1]  # Node feature size
        self.kernels = [
            self.add_weight(
                shape=(feature_dim, self.output_dim),
                initializer="glorot_uniform",
                trainable=True
            ) for _ in range(self.num_edge_types)
        ]

    def call(self, inputs):
        adjacency_tensor, node_features = inputs  # (B, N, N, E), (B, N, F)
        outputs = tf.zeros_like(node_features)

        for i in range(self.num_edge_types):
            adj_matrix = adjacency_tensor[:, :, :, i]  # Edge type i
            aggregated = tf.matmul(adj_matrix, node_features)
            transformed = tf.matmul(aggregated, self.kernels[i])
            outputs += transformed

        return self.activation(outputs)
    
class EdgeFeatureProcessing(Layer):
    """
    Edge Feature Processing Layer.
    This layer updates edge features based on node embeddings and adjacency.
    """

    def __init__(self, edge_dim, activation=tf.nn.relu):
        super(EdgeFeatureProcessing, self).__init__()
        self.edge_dim = edge_dim  # Edge feature size
        self.activation = activation

    def build(self, input_shape):
        node_feature_dim = input_shape[1][-1]  # Get node feature size (F)
        
        # Weight matrix to transform edge features
        self.edge_transform = self.add_weight(
            shape=(node_feature_dim, self.edge_dim),
            initializer="glorot_uniform",
            trainable=True
        )

        # Bias term for transformation
        self.bias = self.add_weight(
            shape=(self.edge_dim,),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        adjacency_tensor, node_features, edge_features = inputs  
        # adjacency_tensor: (B, N, N, E), node_features: (B, N, F), edge_features: (B, N, N, E)

        # Compute transformed node features
        transformed_nodes = tf.matmul(node_features, self.edge_transform)  # (B, N, edge_dim)

        # Expand dimensions to match adjacency structure
        transformed_nodes_i = tf.expand_dims(transformed_nodes, axis=1)  # (B, 1, N, edge_dim)
        transformed_nodes_j = tf.expand_dims(transformed_nodes, axis=2)  # (B, N, 1, edge_dim)

        # Compute edge updates
        edge_updates = adjacency_tensor * (transformed_nodes_i + transformed_nodes_j)  # (B, N, N, E)

        # Apply activation function
        updated_edges = self.activation(edge_updates + self.bias)  # (B, N, N, E)

        return updated_edges
    
class GINConv(Layer):
    """Graph Isomorphism Network (GIN) Convolutional Layer"""
    def __init__(self, output_dim, epsilon=0.1):
        super(GINConv, self).__init__()
        self.output_dim = output_dim
        self.epsilon = epsilon  # Learnable parameter for message passing
        self.mlp = Dense(output_dim, activation=tf.nn.relu)  # MLP inside GIN

    def call(self, adjacency_matrix, node_features):
        """Apply GIN update rule"""
        # Aggregate neighborhood information
        neighbor_sum = tf.matmul(adjacency_matrix, node_features)  # Sum of neighbors

        # Apply GIN update: (1 + epsilon) * node_features + aggregated_neighbors
        updated_features = (1 + self.epsilon) * node_features + neighbor_sum

        # Pass through MLP
        return self.mlp(updated_features)
