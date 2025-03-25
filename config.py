import tensorflow as tf
import os

class Config:
    """
    Configuration settings for the MolGAN model.
    """

    # Data parameters
    DATASET_PATH = "data/qm91/qm91/"  # Path to QM9 SDF file
    PREPROCESSED_DATA_PATH = "data/preprocessed_qm9.pkl"  # Path to save/load processed graph data
    SMILES_PATH = "data/smiles.txt"  # Path to save SMILES strings
    NUM_NODE_TYPES = 4  # Number of atom types in QM9
    NUM_EDGE_TYPES = 4  # Number of bond types in QM9
    MAX_NODES = 9  # Maximum number of nodes in a molecule

    # Model hyperparameters
    LATENT_DIM = 56  # Dimensionality of latent space
    NODE_DIM = NUM_NODE_TYPES  # Node feature dimension
    EDGE_DIM = NUM_EDGE_TYPES  # Edge feature dimension
    HIDDEN_DIM = 64  # Hidden dimension for GCN
    NUM_GCN_LAYERS = 3  # Number of GCN layers

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    INITIAL_LEARNING_RATE = 5e-4  
    MIN_LEARNING_RATE = 1e-5
    NUM_EPOCHS = 20
    FEATURE_MATCHING = True
    DISCRIMINATOR_STEPS = 2

    # Loss function weights
    LAMBDA_ADV = 1.0  # Adversarial loss weight
    LAMBDA_FM = 12.0  # Feature matching loss weight
    LAMBDA_REWARD = 7.0  # Reward network loss weight

    # RL Reward Weights
    LAMBDA_QED = 1.0  
    LAMBDA_DIVERSITY = 2.2  
    LAMBDA_VALIDITY = 1.2  # Higher weight for generating valid molecules
    LAMBDA_UNIQUENESS = 1.0  
    LAMBDA_RL = 5.0  # Scaling factor for reinforcement learning

    LAMBDA_SIMILARITY = 0.1  # Controls Tanimoto penalty strength
    LAMBDA_ENTROPY = 0.07  # Strength of entropy regularization

    # Gumbel-Softmax parameters
    GUMBEL_TAU = 0.8  # Initial temperature
    GUMBEL_MIN_TAU = 0.5  # Minimum temperature for annealing
    GUMBEL_DECAY = 0.995  # Decay rate per epoch

    #Reward
    REWARD_LR = 0.0001
    REWARD_SAVE_PATH = "checkpoints/reward_network"


    # Miscellaneous
    SEED = 42  # Random seed for reproducibility
    DEVICE = "cuda" if tf.config.list_physical_devices('GPU') else "cpu"
    SAVE_MODEL_PATH = "checkpoints/molgan"

    # Ensure directories exist
    os.makedirs(os.path.dirname(PREPROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

# Initialize configuration
config = Config()