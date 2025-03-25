import tensorflow as tf

def wasserstein_loss(y_true, y_pred):
    """Wasserstein loss for WGAN"""
    return -tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator, adj_real, nodes_real, adj_fake, nodes_fake, lambda_gp=12):
    """Computes gradient penalty for WGAN-GP."""
    batch_size = tf.shape(adj_real)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)

    # Interpolated inputs
    adj_interpolated = alpha * adj_real + (1 - alpha) * adj_fake
    nodes_interpolated = alpha * nodes_real + (1 - alpha) * nodes_fake

    with tf.GradientTape() as tape:
        tape.watch([adj_interpolated, nodes_interpolated])
        pred_interpolated = discriminator(adj_interpolated, nodes_interpolated)

    grads = tape.gradient(pred_interpolated, [adj_interpolated, nodes_interpolated])
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads[0])) + tf.reduce_sum(tf.square(grads[1])) + 1e-8)

    return lambda_gp * tf.reduce_mean((grad_norm - 1.0) ** 2)


def feature_matching_loss(real_features, fake_features):
    """Feature Matching Loss"""
    return tf.reduce_mean(tf.abs(tf.reduce_mean(real_features, axis=0) - tf.reduce_mean(fake_features, axis=0)))

def reward_loss(reward_pred):
    """Reinforcement Learning-based Reward Loss"""
    return -tf.reduce_mean(reward_pred)