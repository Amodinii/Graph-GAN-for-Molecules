import tensorflow as tf
from optimizers.losses import wasserstein_loss, gradient_penalty, feature_matching_loss, reward_loss
from models.encoder import Encoder  # Import Encoder

class GraphGANOptimizer(tf.Module):
    def __init__(self, generator, discriminator, reward_network, encoder, config):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.reward_network = reward_network
        self.encoder = encoder  
        self.config = config

        # Initialize optimizers
        self.reset_optimizers()

    def reset_optimizers(self):
        """Reset optimizers for reproducibility."""
        self.g_optimizer = tf.keras.optimizers.Adam(self.config.INITIAL_LEARNING_RATE, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(self.config.INITIAL_LEARNING_RATE, beta_1=0.5, beta_2=0.9)
        self.r_optimizer = tf.keras.optimizers.Adam(self.config.INITIAL_LEARNING_RATE, beta_1=0.5, beta_2=0.9)
        self.e_optimizer = tf.keras.optimizers.Adam(self.config.INITIAL_LEARNING_RATE, beta_1=0.5, beta_2=0.9)  

    def train_step(self, adjacency_real, nodes_real, epoch):
        batch_size = tf.shape(adjacency_real)[0]

        # Encode real molecules
        encoded_latent = self.encoder(adjacency_real, nodes_real)

        # Gumbel-Softmax Temperature Annealing
        tau = max(self.config.GUMBEL_TAU * (self.config.GUMBEL_DECAY ** epoch), 0.1)

        # Noisy Latent Fusion (Concatenation + Gating Mechanism)
        noise = tf.random.normal([batch_size, self.config.LATENT_DIM])
        latent_input = tf.concat([encoded_latent, noise], axis=-1)
        latent_input = tf.keras.layers.Dense(self.config.LATENT_DIM, activation=tf.nn.sigmoid)(latent_input)

        # Train Discriminator More Often (5 Steps per G Update)
        for _ in range(self.config.DISCRIMINATOR_STEPS):
            with tf.GradientTape() as d_tape:
                adjacency_fake, nodes_fake = self.generator(latent_input, tau=tau)

                # Compute Discriminator Loss
                real_pred = self.discriminator(adjacency_real, nodes_real)
                fake_pred = self.discriminator(adjacency_fake, nodes_fake)

                d_loss = wasserstein_loss(tf.ones_like(real_pred), real_pred) + \
                        wasserstein_loss(-tf.ones_like(fake_pred), fake_pred)

                # Gradient Penalty (WGAN-GP)
                gp = gradient_penalty(self.discriminator, adjacency_real, nodes_real, adjacency_fake, nodes_fake)
                d_loss += self.config.LAMBDA_GP * gp

            # Apply gradients to Discriminator
            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train Generator Once
        with tf.GradientTape(persistent=True) as g_tape:
            adjacency_fake, nodes_fake = self.generator(latent_input, tau=tau)

            # **Compute RL-Based Rewards**
            qed_rewards, diversity_rewards, validity_rewards, uniqueness_rewards = \
                self.reward_network.compute_reward(adjacency_fake, nodes_fake)

            # **Mask invalid molecules by setting their rewards to zero**
            valid_mask = tf.cast(validity_rewards > 0, tf.float32)
            qed_rewards *= valid_mask
            diversity_rewards *= valid_mask
            uniqueness_rewards *= valid_mask

            total_reward = (self.config.LAMBDA_QED * qed_rewards +
                            self.config.LAMBDA_DIVERSITY * diversity_rewards +
                            self.config.LAMBDA_UNIQUENESS * uniqueness_rewards)

            policy_loss = -tf.reduce_mean(total_reward)

            # Generator Adversarial Loss
            fake_pred = self.discriminator(adjacency_fake, nodes_fake)
            g_loss = wasserstein_loss(tf.ones_like(fake_pred), fake_pred)

            # Feature Matching Loss
            if self.config.FEATURE_MATCHING:
                real_features = self.discriminator.extract_features(adjacency_real, nodes_real)
                fake_features = self.discriminator.extract_features(adjacency_fake, nodes_fake)
                g_loss += self.config.LAMBDA_FM * feature_matching_loss(real_features, fake_features)

            # Entropy Regularization (Encourage Exploration)
            entropy_loss = -tf.reduce_mean(nodes_fake * tf.math.log(nodes_fake + 1e-8))

            # **Final Generator Loss**
            g_total_loss = (g_loss + self.config.LAMBDA_RL * policy_loss +
                            self.config.LAMBDA_ENTROPY * entropy_loss)

            # **Encoder Reconstruction Loss**
            reconstructed_latent = self.encoder(adjacency_fake, nodes_fake)
            e_loss = tf.reduce_mean(tf.keras.losses.MSE(encoded_latent, reconstructed_latent))

        # Apply gradients to Generator, Reward Network, and Encoder
        g_grads = g_tape.gradient(g_total_loss, self.generator.trainable_variables)
        r_grads = g_tape.gradient(policy_loss, self.reward_network.trainable_variables)
        e_grads = g_tape.gradient(e_loss, self.encoder.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.r_optimizer.apply_gradients(zip(r_grads, self.reward_network.trainable_variables))
        self.e_optimizer.apply_gradients(zip(e_grads, self.encoder.trainable_variables))

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "rl_reward": tf.reduce_mean(total_reward),
            "entropy_loss": entropy_loss,
            "e_loss": e_loss
        }
