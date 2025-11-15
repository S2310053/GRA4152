##
#  This module defines the VAE class 
#  Contains the behavior in VAEs
#

## Load necessary libraries and packages
#  @library numpy for vectorization and data handling
#  @library os, os.environ 3 to just include tensorflow error messages
#  @library tensorflow let us train our model, VAE inherits it to prevent crash 
#
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = 
import tensorflow as tf

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

    ## Generates latent space z and visualizes it
    #  @param posteriorDistribution p
    #  @return plot of the latent space from posterior distribution

