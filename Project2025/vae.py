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

## Variational Autoencoders are models in generative artificial intelligence
#  They combine deep learning and probabilistic modeling
#  Which enables generative modelling and representation learning
#
class VAE(tf):
    ## Train method to update VAE trainable parameters
    #  @param x our train data set (black and white or colored images)
    #  @param optimizer adam to improve efficiency in convergence
    #  @return loss as an accuracy measure of the model
    #
    @tf.function
    def train(self,x, optimizer):

    ## Generates and visualizes latent space z
    #  @param latent space z generated in encoder
    #  @return scatter plot
    #
    #def visualizeLatent(self, visualize = True):

    ## Generates new images
    #  @ param xhat predicted value generated in decoder
    #  @ param distribution call the name of distribution
    #  @ return generated images
    #def generateImages(self,distribution, xhat):

