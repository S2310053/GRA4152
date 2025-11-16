import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from encoder import Encoder
from decoder import Decoder
from bicoder import BiCoder


class VAE(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x, color=False):
        """
        Forward pass of the VAE.
        Returns x̂, μ, log σ²
        """
        if color:
            mu, log_var = self.encoder.getEncoderCNN(x)
        else:
            mu, log_var = self.encoder.getEncoderMLP(x)

        eps = tf.random.normal(shape=tf.shape(mu))
        z   = mu + tf.exp(0.5 * log_var) * eps

        xhat = self.decoder.getDecoderCNN(z) if color else self.decoder.getDecoderMLP(z)
        return xhat, mu, log_var

    @staticmethod
    def kl_divergence(mu, log_var):
        """
        KL(q(z|x) || p(z)) for diagonal Gaussian.
        """
        return 0.5 * tf.reduce_sum(
            tf.square(mu) + tf.exp(log_var) - log_var - 1,
            axis=1
        )

    @staticmethod
    def recon_log_likelihood(x, xhat, sigma2=1.0):
        """
        Gaussian log-likelihood log p(x|z)
        """
        return -0.5 * tf.reduce_sum(
            (x - xhat)**2 / sigma2 + tf.math.log(2.0 * np.pi * sigma2),
            axis=1
        )

    def elbo_loss(self, x, xhat, mu, log_var):
        """
        Computes the NEGATIVE ELBO:
        L = - E_q[ log p(x|z) ] + KL(q||p)
        """
        log_px_z = self.recon_log_likelihood(x, xhat)
        kl = self.kl_divergence(mu, log_var)
        return tf.reduce_mean(-log_px_z + kl)  # negative ELBO

    @tf.function
    def train_step(self, x, optimizer, color=False):
        """
        Single gradient update step that explicitly optimizes Negative ELBO.
        """
        with tf.GradientTape() as tape:
            xhat, mu, log_var = self.call(x, color=color)
            loss = self.elbo_loss(x, xhat, mu, log_var)

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss
