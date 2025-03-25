import os
import tensorflow as tf
import numpy as np
from config import Config
from data.preprocess import QM9Preprocessor
from models.generator import Generator
from models.discriminator import Discriminator
from models.reward import RewardNetwork
from models.encoder import Encoder  # Import the Encoder
from optimizers.gan import GraphGANOptimizer
from training.train import MolGANTrainer as train
from evaluation.metrics import MolGANMetrics

def main():
    # Load configuration
    config = Config()

    # Ensure directories exist
    os.makedirs(os.path.dirname(config.PREPROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(config.SAVE_MODEL_PATH, exist_ok=True)

    # Load and preprocess QM9 dataset
    print("Loading QM9 dataset...")
    preprocessor = QM9Preprocessor(config)
    adjacency_tensor, node_tensor, smiles_list = preprocessor.load_qm9_data()

    # Define models
    print("Initializing models...")
    generator = Generator(config)
    discriminator = Discriminator(config)
    reward_network = RewardNetwork(config)
    encoder = Encoder(config)  # Initialize the Encoder

    # Define optimizer with Encoder
    optimizer = GraphGANOptimizer(
        generator=generator,
        discriminator=discriminator,
        reward_network=reward_network,
        encoder=encoder,  # Pass Encoder here
        config=config
    )

    # Reset optimizers before training (important)
    optimizer.reset_optimizers()

    # Train model
    print("Starting training...")
    train(config=config)

    # Evaluate model
    print("Evaluating model...")
    metrics = MolGANMetrics(config, generator).evaluate(num_samples=1000)
    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
