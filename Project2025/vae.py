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
import mathplotlib.pyplot as plt
from mlp_toolkits.axes_grid1 import ImageGrid
from encoder import Encoder
from decoder import Decoder

## Variational Autoencoders are models in generative artificial intelligence
#  They combine deep learning and probabilistic modeling
#  Which enables generative modelling and representation learning
#
class VAE(tf):
    ## Train method to update VAE trainable parameters
    #  @param x our train data set (black and white or colored images)
    #  @param optimizer adam to improve efficiency in convergence
    #  @return loss as an accuracy measure of the model from the objective function
    #          all optimizers always minimize so the expression becomes negative
    #
    @tf.function
    def train(self,x, optimizer):
        with tf.GradientTape() as tape:
            loss      = self.call(x)
            gradients = tape.gradients(self.vae_loss,self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            return loss

    ## Generates and visualizes latent space z
    #  @param latent space z generated in encoder
    #  @return scatter plot
    #
    def visualizeLatent(self, data):

    ## Generates new images
    #  @ param xhat predicted value generated in decoder
    #  @ param distribution call the name of distribution
    #  @ return generated images
    #def generateImages(self,distribution, xhat):

