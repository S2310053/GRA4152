import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from encoder import Encoder
from decoder import Decoder
from bicoder import BiCoder


# vae.py

class VAE(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x, color=False):
        """Forward pass. Returns xHat, mu, logVar."""
        if color:
            mu, logVar = self.encoder.getEncoderCNN(x)          # x: (B, 28, 28, 3)
        else:
            # Flatten BW input for MLP encoder
            xFlat = tf.reshape(x, [tf.shape(x)[0], 28 * 28])     # x: (B, 28, 28, 1)
            mu, logVar = self.encoder.getEncoderMLP(xFlat)

        eps = tf.random.normal(shape=tf.shape(mu))
        z   = mu + tf.exp(0.5 * logVar) * eps

        if color:
            xHat = self.decoder.getDecoderCNN(z)                # (B, 28, 28, 3) in [0,1]
        else:
            xHat = self.decoder.getDecoderMLP(z)                # (B, 784) in [0,1]
            xHat = tf.reshape(xHat, [-1, 28, 28, 1])

        return xHat, mu, logVar

    @staticmethod
    def klDivergence(mu, logVar):
        """KL(q(z|x) || p(z)) for diagonal Gaussian."""
        return 0.5 * tf.reduce_sum(
            tf.square(mu) + tf.exp(logVar) - logVar - 1,
            axis=1
        )

    @staticmethod
    def reconLogLikelihood(x, xHat, color=False):
        """
        Gaussian log-likelihood log p(x|z).
        Pixels are in [0,1]. Use a small sigma^2 to sharpen.
        """
        sigma2 = 0.1 if color else 0.01
        axes = list(range(1, len(x.shape)))  # sum over H,W,C

        return -0.5 * tf.reduce_sum(
            (x - xHat) ** 2 / sigma2 + tf.math.log(2.0 * np.pi * sigma2),
            axis=axes
        )

    def elboLoss(self, x, xHat, mu, logVar, color=False):
        """Negative ELBO = -E_q[log p(x|z)] + KL(q||p)."""
        logPxZ = self.reconLogLikelihood(x, xHat, color=color)
        kl     = self.klDivergence(mu, logVar)
        return tf.reduce_mean(-logPxZ + kl)

    @tf.function
    def train(self, x, optimizer, color=False):
        """One gradient step."""
        with tf.GradientTape() as tape:
            xHat, mu, logVar = self.call(x, color=color)
            loss = self.elboLoss(x, xHat, mu, logVar, color=color)

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    # ---------- Latent space ----------

    def visualizeLatent(self, dataset, color=False, limit=5000, savefig="latent_space.pdf"):
        """Collect latent means and project to 2D with TSNE."""
        zs = []
        nSamples = 0

        for batch in dataset:
            _, mu, _ = self.call(batch, color=color)
            zs.append(mu.numpy())
            nSamples += mu.shape[0]
            if nSamples >= limit:
                break

        Z = np.concatenate(zs, axis=0)[:limit]
        Z2 = TSNE(n_components=2, init="random", learning_rate="auto").fit_transform(Z)

        plt.figure(figsize=(8, 6))
        plt.scatter(Z2[:, 0], Z2[:, 1], s=4, alpha=0.6)
        plt.title("Latent Space Visualization (TSNE)")
        plt.savefig(savefig)
        plt.close()

    # ---------- Image grids ----------

    def plotGrid(self, images, color=False, N=10, C=10, figsize=(18, 18), name="grid"):
        """Plot an N x C grid of images. Input images are in [0,1]."""
        # Scale to [0,255] and convert to uint8 for plotting
        images = tf.clip_by_value(255.0 * images, 0, 255)
        images = tf.cast(images, tf.uint8).numpy()

        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(N, C), axes_pad=0)

        for ax, im in zip(grid, images):
            if color:
                ax.imshow(im)
            else:
                ax.imshow(im.squeeze(), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.savefig(f"{name}.pdf")
        plt.close()
        print(f"Saved: {name}.pdf")

    # ---------- Generation ----------

    def generateFromPrior(self, n=100, color=False):
        """Sample z ~ N(0, I) and decode."""
        latentDim = BiCoder._latentDimensionColor if color else BiCoder._latentDimensionBlackWhite
        z = tf.random.normal((n, latentDim))
        xHat = self.decoder.getDecoderCNN(z) if color else self.decoder.getDecoderMLP(z)

        if not color:
            xHat = tf.reshape(xHat, (-1, 28, 28, 1))

        return xHat  # still in [0,1]

    def generateFromPosterior(self, dataset, n=100, color=False):
        """Sample z from q(z|x) for real x and decode."""
        images = []
        for batch in dataset:
            xHat, mu, logVar = self.call(batch, color=color)
            eps = tf.random.normal(tf.shape(mu))
            z   = mu + tf.exp(0.5 * logVar) * eps
            xHat = self.decoder.getDecoderCNN(z) if color else self.decoder.getDecoderMLP(z)

            if not color:
                xHat = tf.reshape(xHat, (-1, 28, 28, 1))

            images.append(xHat.numpy())

            if sum(b.shape[0] for b in images) >= n:
                break

        return np.concatenate(images, axis=0)[:n]  # still in [0,1]
