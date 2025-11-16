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

    def call(self, x=None, priorIsotropic=False, color=False):
        """
        Forward pass of the VAE.
        - If priorIsotropic=True: sample z ~ N(0, I) and decode (generation mode).
        - Else: encode x -> (mu, log_var), sample z, decode, and compute ELBO loss.
        """

        # --- PURE GENERATION FROM PRIOR ---
        if priorIsotropic:
            latent_dim = BiCoder._latentDimensionColor if color else BiCoder._latentDimensionBlackWhite

            # if x is provided, match batch size, otherwise use n=1
            if x is not None:
                batch_size = tf.shape(x)[0]
            else:
                batch_size = 1

            z = self.encoder.getEncoderIsotropic(latent_dim, n=batch_size)
            xhat = self.decoder.getDecoderCNN(z) if color else self.decoder.getDecoderMLP(z)
            # no loss in pure generation mode
            return xhat

        # --- ENCODE ---
        if color:
            mu, log_var = self.encoder.getEncoderCNN(x)   # (batch, 50)
        else:
            mu, log_var = self.encoder.getEncoderMLP(x)   # (batch, 20)

        # --- REPARAMETERIZATION TRICK: z = mu + sigma * eps ---
        eps = tf.random.normal(shape=tf.shape(mu))
        z   = mu + tf.exp(0.5 * log_var) * eps

        # --- DECODE ---
        xhat = self.decoder.getDecoderCNN(z) if color else self.decoder.getDecoderMLP(z)

        # --- RECONSTRUCTION LOSS (MSE) ---
        recon_loss = tf.reduce_mean(tf.square(x - xhat))

        # --- KL DIVERGENCE ---
        kl_loss = 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=1)
        )

        # store total loss (negative ELBO)
        self.vae_loss = recon_loss + kl_loss

        return xhat

    # -------- Optional: helper formulas for Gaussian log-likelihood and KL --------
    @staticmethod
    def log_diag_mvn(x, mu, log_var):
        """
        Log density of a diagonal multivariate normal N(mu, diag(exp(log_var))).
        """
        sum_axes = tf.range(1, tf.rank(mu))
        k        = tf.cast(tf.reduce_prod(tf.shape(mu)[1:]), x.dtype)
        logp     = -0.5 * k * tf.math.log(2.0 * np.pi) \
                   - 0.5 * tf.reduce_sum(log_var, axis=sum_axes) \
                   - 0.5 * tf.reduce_sum(tf.square(x - mu) / tf.exp(log_var), axis=sum_axes)
        return logp

    @staticmethod
    def kl_divergence(mu, log_var):
        """
        KL divergence between q(z|x) = N(mu, diag(exp(log_var))) and p(z) = N(0, I).
        """
        return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=1)

    def lossFunction(self, x, xhat, mu, log_var):
        """
        Alternative loss using the exact Gaussian log-likelihood (negative ELBO).
        Not used in train(), but kept for reference.
        """
        recon_loss = -tf.reduce_mean(self.log_diag_mvn(x, xhat, log_var))
        kl_loss    = tf.reduce_mean(self.kl_divergence(mu, log_var))
        total_loss = recon_loss + kl_loss
        return total_loss

    @tf.function
    def train(self, x, optimizer, color=False):
        """
        One training step: forward + backward.
        """
        with tf.GradientTape() as tape:
            # Forward pass: computes xhat and stores self.vae_loss
            self.call(x, color=color)
            loss = self.vae_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


