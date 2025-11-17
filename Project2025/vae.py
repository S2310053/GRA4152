##
#  This module defines the VAE class
#  It combines the encoder, decoder, and variational inference machinery.
#
#  Some useful information:
#    - Inheritance: derives from tf.keras.Model to integrate naturally with
#      TensorFlow’s training loop, gradient tracking, and variable management.
#    - Overriding: the call() method defines the forward pass of the model.
#    - Decorators: @tf.function is used for compilation and execution speed
#      in the training step.
#    - Polymorphism: supports two data modes (BW or color) through the same
#      forward interface.
#
#  Notes:
#    - The VAE optimizes the Evidence Lower Bound (ELBO):
#          ELBO = E_q[log p(x|z)] − KL(q(z|x) || p(z))
#      Minimizing −ELBO encourages accurate reconstruction while regularizing
#      the latent space via KL divergence.
#    - Reconstruction uses a Gaussian log-likelihood with small variance to
#      yield sharper and more numerically stable training signals.
#    - TSNE projects latent means into 2D to reveal clustering structure in
#      the learned manifold.
#
#  @module Encoder, Decoder  architecture components
#  @module BiCoder           latent dimension access
#  @library tensorflow       tensors, ops, model class
##
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  =  "3"

import tensorflow            as tf
from sklearn.manifold        import TSNE
import matplotlib.pyplot      as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from encoder                 import Encoder
from decoder                 import Decoder
from bicoder                 import BiCoder


##
#  Variational Autoencoder: combines encoder + decoder + ELBO objective.
#
class VAE(tf.keras.Model):

    ##
    #  Constructor.
    #
    #  Overriding:
    #    - Extends tf.keras.Model by defining internal encoder and decoder.
    #
    def __init__(self):
        super().__init__()
        self.encoder   =  Encoder()
        self.decoder   =  Decoder()


    ##
    #  Forward pass: encodes x → (mu, log_var) → samples z → decodes x̂.
    #
    #  Polymorphism:
    #    - Same call() for BW and color; distinct internal paths selected
    #      implicitly based on the boolean flag.
    #
    #  @param x      input images
    #  @param color  whether using the CNN architecture
    #
    #  @return xHat, mu, logVar
    #
    def call(self, x, color=False):

        if color:
            mu, logVar   =  self.encoder.getEncoderCNN(x)
        else:
            xFlat        =  tf.reshape(x, [tf.shape(x)[0], 28 * 28])
            mu, logVar   =  self.encoder.getEncoderMLP(xFlat)

        eps             =  tf.random.normal(tf.shape(mu))
        z               =  mu + tf.exp(0.5 * logVar) * eps

        if color:
            xHat         =  self.decoder.getDecoderCNN(z)
        else:
            xHat         =  self.decoder.getDecoderMLP(z)
            xHat         =  tf.reshape(xHat, [-1, 28, 28, 1])

        return xHat, mu, logVar


    ##
    #  KL divergence term: KL(q(z|x) || p(z))
    #
    @staticmethod
    def klDivergence(mu, logVar):

        return 0.5 * tf.reduce_sum(
            tf.square(mu) + tf.exp(logVar) - logVar - 1,
            axis = 1
        )


    ##
    #  Gaussian log-likelihood for reconstruction.
    #
    #  *Note:
    #    - A small sigma squared (variance) sharpens reconstructions while keeping the term
    #      numerically stable
    #
    @staticmethod
    def reconLogLikelihood(x, xHat, color=False):

        sigma2   = 0.1 if color else 0.01
        axes     = list(range(1, len(x.shape)))

        return -0.5 * tf.reduce_sum(
            (x - xHat)**2 / sigma2 + tf.math.log(2.0 * np.pi * sigma2),
            axis = axes
        )


    ##
    #  Negative ELBO loss: −E[log p(x|z)] + KL.
    #
    def elboLoss(self, x, xHat, mu, logVar, color=False):

        logPxZ   =  self.reconLogLikelihood(x, xHat, color=color)
        kl       =  self.klDivergence(mu, logVar)

        return tf.reduce_mean(-logPxZ + kl)


    ##
    #  One training step compiled with @tf.function for speed.
    #
    #  Decorator:
    #    - Compiles the gradient computation and forward pass to optimize
    #      performance during training.
    #
    @tf.function
    def train(self, x, optimizer, color=False):

        with tf.GradientTape() as tape:
            xHat, mu, logVar  =  self.call(x, color=color)
            loss              =  self.elboLoss(x, xHat, mu, logVar, color=color)

        grads   =  tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss


    ##
    #  Projects latent means into 2D using TSNE for visualization.
    #
    def visualizeLatent(self, dataset, color=False, limit=5000, savefig="latent_space.pdf"):

        zs        =  []
        count     =  0

        for batch in dataset:
            _, mu, _   =  self.call(batch, color=color)
            zs.append(mu.numpy())
            count     +=  mu.shape[0]

            if count >= limit:
                break

        Z          =  np.concatenate(zs, axis=0)[:limit]
        Z2         =  TSNE(n_components=2, init="random",
                           learning_rate="auto").fit_transform(Z)

        plt.figure(figsize=(8, 6))
        plt.scatter(Z2[:, 0], Z2[:, 1], s=4, alpha=0.6)
        plt.title("Latent Space (TSNE)")
        plt.savefig(savefig)
        plt.close()
        print(f"Saved: latent_space.pdf")


    ##
    #  Plots an N×C grid of images.
    #
    def plotGrid(self, images, color=False, N=10, C=10, figsize=(18, 18), name="grid"):

        images   =  tf.clip_by_value(255.0 * images, 0, 255)
        images   =  tf.cast(images, tf.uint8).numpy()

        fig      =  plt.figure(figsize=figsize)
        grid     =  ImageGrid(fig, 111, nrows_ncols=(N, C), axes_pad=0)

        for ax, img in zip(grid, images):
            if color:
                ax.imshow(img)
            else:
                ax.imshow(img.squeeze(), cmap="gray")

            ax.set_xticks([])
            ax.set_yticks([])

        plt.savefig(f"{name}.pdf")
        plt.close()
        print(f"Saved: {name}.pdf")


    ##
    #  Generates samples from the prior p(z) = N(0, I).
    #
    def generateFromPrior(self, n=100, color=False):

        dim      =  BiCoder._latentDimColor if color else BiCoder._latentDimBW
        z        =  tf.random.normal((n, dim))

        xHat     =  self.decoder.getDecoderCNN(z) if color else self.decoder.getDecoderMLP(z)

        if not color:
            xHat  =  tf.reshape(xHat, (-1, 28, 28, 1))

        return xHat


    ##
    #  Generates samples from q(z|x) using real dataset images.
    #
    def generateFromPosterior(self, dataset, n=100, color=False):

        out       =  []

        for batch in dataset:
            _, mu, logVar  =  self.call(batch, color=color)

            eps           =  tf.random.normal(tf.shape(mu))
            z             =  mu + tf.exp(0.5 * logVar) * eps

            xHat          =  self.decoder.getDecoderCNN(z) if color else self.decoder.getDecoderMLP(z)

            if not color:
                xHat      =  tf.reshape(xHat, (-1, 28, 28, 1))

            out.append(xHat.numpy())

            if sum(arr.shape[0] for arr in out) >= n:
                break

        return np.concatenate(out, axis=0)[:n]

