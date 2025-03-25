import os
import tensorflow as tf
from config import Config
from data.preprocess import QM9Preprocessor
from models.generator import Generator
from models.discriminator import Discriminator
from models.reward import RewardNetwork
from models.encoder import Encoder
from optimizers.gan import GraphGANOptimizer
from evaluation.metrics import MolGANMetrics

class MolGANTrainer:
    def __init__(self, config):
        self.config = config

        # Load dataset
        self.preprocessor = QM9Preprocessor(config)
        self.adjacency_tensor, self.node_tensor, _ = self.preprocessor.load_qm9_data()

        # Initialize models
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.reward_network = RewardNetwork(config)
        self.encoder = Encoder(config)

        # Define optimizer with Encoder
        self.optimizer = GraphGANOptimizer(
            generator=self.generator,
            discriminator=self.discriminator,
            reward_network=self.reward_network,
            encoder=self.encoder,
            config=config
        )

        # Evaluation Metrics
        self.metrics = MolGANMetrics(config, self.generator)

        # Prepare checkpointing
        self.checkpoint_dir = config.SAVE_MODEL_PATH
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            reward_network=self.reward_network,
            encoder=self.encoder,
            optimizer=self.optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=5)

        # Separate checkpointing for reward network
        self.reward_checkpoint = tf.train.Checkpoint(reward_network=self.reward_network)
        self.reward_checkpoint_manager = tf.train.CheckpointManager(self.reward_checkpoint, config.REWARD_SAVE_PATH, max_to_keep=5)

        # Prepare logging
        self.log_dir = "logs/molgan"
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        # Load checkpoint if exists
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restored from checkpoint: {self.checkpoint_manager.latest_checkpoint}")

        if self.reward_checkpoint_manager.latest_checkpoint:
            self.reward_checkpoint.restore(self.reward_checkpoint_manager.latest_checkpoint)
            print(f"Restored reward network from checkpoint: {self.reward_checkpoint_manager.latest_checkpoint}")

    def train(self):
        print("Starting training...")
        dataset = tf.data.Dataset.from_tensor_slices((self.adjacency_tensor, self.node_tensor))
        dataset = dataset.shuffle(len(self.adjacency_tensor)).batch(self.config.BATCH_SIZE)

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            for step, (adj_batch, node_batch) in enumerate(dataset):
                losses = self.optimizer.train_step(adj_batch, node_batch, epoch)

                # Logging losses
                with self.summary_writer.as_default():
                    tf.summary.scalar("Discriminator Loss", losses["d_loss"], step=epoch)
                    tf.summary.scalar("Generator Loss", losses["g_loss"], step=epoch)
                    tf.summary.scalar("RL Reward", losses["rl_reward"], step=epoch)
                    tf.summary.scalar("Reward Prediction Loss", losses["reward_loss"], step=epoch)  # Log Reward Loss
                    tf.summary.scalar("Gumbel Temperature", losses["gumbel_tau"], step=epoch)  # Log Gumbel temperature

                # Print status
                if step % 10 == 0:
                    print(f"Epoch {epoch} Step {step} | D Loss: {losses['d_loss']:.4f} | "
                          f"G Loss: {losses['g_loss']:.4f} | RL Reward: {losses['rl_reward']:.4f} | "
                          f"Reward Loss: {losses['reward_loss']:.4f} | Gumbel τ: {losses['gumbel_tau']:.4f}")

            # Evaluate model every few epochs
            if epoch % self.config.EVAL_INTERVAL == 0:
                eval_results = self.metrics.evaluate(num_samples=1000)
                print(f"\nEpoch {epoch} - Evaluation Results:")
                for key, value in eval_results.items():
                    print(f"   {key}: {value:.4f}")

                # Log evaluation metrics
                with self.summary_writer.as_default():
                    for key, value in eval_results.items():
                        tf.summary.scalar(f"Metrics/{key}", value, step=epoch)

            # Save model periodically
            if epoch % self.config.SAVE_INTERVAL == 0:
                self.checkpoint_manager.save()
                self.reward_checkpoint_manager.save()  # Save reward network separately
                print(f"Checkpoint saved at epoch {epoch}.")

        print("Training complete.")

if __name__ == "__main__":
    config = Config()
    trainer = MolGANTrainer(config)
    trainer.train()
