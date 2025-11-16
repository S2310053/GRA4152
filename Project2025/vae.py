##
#  This module defines the VAE class 
#  Contains the behavior in VAEs
#

## Load necessary libraries and packages
#  @library numpy for vectorization and data handling
#  @library os, os.environ 3 to just include tensorflow error messages
#  @library tensorflow let us train our model, VAE inherits it to prevent crash 
#  @module  TSNE from sklearn.manifold for manifold learning
#  @module  plt from mathplotlib.pyplot for plotting
#  @module  ImageGrid form mlp_toolkits.axes_grid1 displays images (multi fixed)
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#from mlp_toolkits.axes_grid1 import ImageGrid
from encoder import Encoder
from decoder import Decoder
from bicoder import BiCoder

## Variational Autoencoders are models in generative artificial intelligence
#  They combine deep learning and probabilistic modeling
#  Which enables generative modelling and representation learning
#
class VAE(tf.keras.Model):
    ## The constructor declares the objects which methods
    #  include the computations for encoders and decoders and sample
    #  retrieves Encoder and Decoder Classes
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    ## Defines the loss function which becomes the objective function
    #  That we maximize in practice through maximum likelihood estimation
    #  @param data x from our data sets
    #  @param prior true when data z is sampled from prior distribution
    #  @param color to identify between the black and white and color images
    #  @returns new images (xhat) depending the data methods
    #
    def call(self, x = None,  priorIsotropic = False, color = False):
        if priorIsotropic:
            z = self.encoder.getEncoderIsotropic()
            xhat = self.decoder.getDecoderCNN(z) if color else self.decoder.getDecoderMLP(z)
            return xhat
        if color:
            mu, log_var = self.encoder.getEncoderCNN(x)
            latent_dim = BiCoder._latentDimensionColor
        else:
            mu, log_var = self.encoder.getEncoderMLP(x)
            latent_dim = BiCoder._latentDimensionBlackWhite
            # Reparameterization trick
            eps = tf.random.normal(shape=tf.shape(mu))
            z = mu + tf.exp(0.5 * log_var) * eps
            xhat = self.decoder.getDecoderCNN(z) if color else self.decoder.getDecoderMLP(z)
            recon_loss = tf.reduce_mean(tf.square(x - xhat))
            # KL divergence
            kl_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=1))
            # Store total loss as attribute
            self.vae_loss = recon_loss + kl_loss

            return xhat


   ## Creates a helper function to computes the log function
   #  The objective function of our model by two components
   #  Log density and likelihood plus KL divergence term
   # Formulas retrieved from the examp resources
   #  @param x data from x hat
   #  @param mean
   # @param logVariance
   #
    def log_diag_mvn(x, mu, log_var):
        sum_axes = tf.range(1, tf.rank(mu))
        k        = tf.cast(tf.reduce_prod(tf.shape(mu)[1:]), x.dtype)
        logp     = -0.5 * k * tf.math.log(2 * np.pi) \
                  - 0.5 * log_var \
                  - 0.5 * tf.reduce_sum(tf.square(x - mu) / tf.math.exp(log_var), axis=sum_axes)
        return logp

    def kl_divergence(mu, log_var):
        return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis = 1)

    def lossFunction(self, x, xhat, mu, log_var):
       # Uses log_diag_mvn formula:
       #   log p(x|z) = -½ [ k log(2π) + 2 logσ + ((x - μ)² / exp(2 logσ)) ]
        recon_loss = -tf.reduce_mean(log_diag_mvn(x, xhat, log_var))

       # Uses KL formula:
       #   D_KL = ½ Σ ( μ² + e^{logσ²} - logσ² - 1 )
        kl_loss = tf.reduce_mean(kl_divergence(mu, log_var))

       # total (negative ELBO) ---
        total_loss = recon_loss + kl_loss                                  

        return total_loss

    ## Train method to update VAE trainable parameters
    #  @param x our train data set (black and white or colored images)
    #  @param optimizer adam to improve efficiency in convergence
    #  @return loss as an accuracy measure of the model from the objective function
    #          all optimizers always minimize so the expression becomes negative
    #
    @tf.function
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.call(x)          # xhat computed, loss stored
            loss = self.vae_loss         # retrieve full ELBO loss
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    ## Generates and visualizes latent space z
    #  @param latent space z generated in encoder
    #  @return scatter plot
    #
    #def visualizeLatent(self, data):

    ## Generates new images
    #  @ param xhat predicted value generated in decoder
    #  @ param distribution call the name of distribution
    #  @ return generated images
    #def generateImages(self,distribution, xhat):

