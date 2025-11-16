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
            axis= 1)


    @staticmethod
    def recon_log_likelihood(x, xhat, sigma2_bw=1.0, sigma2_color=25.0, color=False):
        axes = list(range(1, len(x.shape)))
        sigma2 = sigma2_color if color else sigma2_bw
    
        return -0.5 * tf.reduce_sum(
            (x - xhat)**2 / sigma2 + tf.math.log(2.0 * np.pi * sigma2),
            axis=axes
        )

   
      
    def elbo_loss(self, x, xhat, mu, log_var, color=False):
          """
            Computes the NEGATIVE ELBO:
            L = - E_q[ log p(x|z) ] + KL(q||p)
            """
        log_px_z = self.recon_log_likelihood(x, xhat, color=color)
        kl = self.kl_divergence(mu, log_var)
        return tf.reduce_mean(-log_px_z + kl)


    @tf.function
    def train(self, x, optimizer, color=False):
        with tf.GradientTape() as tape:
            xhat, mu, log_var = self.call(x, color=color)
            loss = self.elbo_loss(x, xhat, mu, log_var, color=color)
    
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


    ### Latent space fplot

    def visualize_latent(self, dataset, color=False, limit=5000, savefig="latent_space.pdf"):
        
        """
        Collects latent means (mu) from the encoder and visualizes them with TSNE.
        Works for both BW and Color datasets.
        """
        zs = []
        n = 0
    
        for batch in dataset:
            _, mu, log_var = self.call(batch, color=color)
            zs.append(mu.numpy())
            n += mu.shape[0]
            if n >= limit:
                break
    
        Z = np.concatenate(zs, axis=0)[:limit]
    
        print(f"[LATENT] Running TSNE on {Z.shape[0]} samples...")
        Z2 = TSNE(n_components=2, init="random", learning_rate="auto").fit_transform(Z)
    
        plt.figure(figsize=(8, 6))
        plt.scatter(Z2[:, 0], Z2[:, 1], s=4, alpha=0.6)
        plt.title("Latent Space Visualization (TSNE)")
        plt.savefig(savefig)
        plt.close()
        print(f"[LATENT] Saved latent visualization to {savefig}")






   # image gird plot
    def plot_grid(self, images, color=False, N=10, C=10, figsize=(18,18), name="grid"):
        if not color:  # BW in [0,1] → scale to [0,255]
            images = tf.clip_by_value(255 * images, 0, 255).numpy().astype(np.uint8)
        else:          # Color already [0,255]
            images = tf.clip_by_value(images, 0, 255).numpy().astype(np.uint8)
    
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(N, C), axes_pad=0)
    
        for ax, im in zip(grid, images):
            if not color:
                ax.imshow(im, cmap="gray")
            else:
                ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
    
        plt.savefig(f"{name}.pdf")
        plt.close()
        print(f"Saved: {name}.pdf")



    # generate from prio plot
    def generate_from_prior(self, n=100, color=False):
        latent_dim = BiCoder._latentDimensionColor if color else BiCoder._latentDimensionBlackWhite
        z = tf.random.normal((n, latent_dim))
        xhat = self.decoder.getDecoderCNN(z) if color else self.decoder.getDecoderMLP(z)
        return xhat



    def generate_from_posterior(self, dataset, n=100, color=False):
        images = []
        
        for batch in dataset:
            xhat, mu, log_var = self.call(batch, color=color)
    
            eps = tf.random.normal(tf.shape(mu))
            z = mu + tf.exp(0.5 * log_var) * eps
    
            xhat = self.decoder.getDecoderCNN(z) if color else self.decoder.getDecoderMLP(z)
            batch_imgs = tf.clip_by_value(255 * xhat, 0, 255).numpy().astype(np.uint8)
    
            images.append(batch_imgs)
    
            if sum(len(b) for b in images) >= n:
                break
    
        images = np.concatenate(images, axis=0)[:n]
        return images
