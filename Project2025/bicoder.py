##
#  This module defines the BiCoder class
#  Contains the common behavior in encoders and decoders
#  

## Load necessary libraries and packages
#  @library numpy as np
#  @library os, os.environ 3 to just include tensorflow error messages
#  @module layers from tensorflow.keras reusable when stating weights NN
#  @module activations from tensorflow.keras adds non-linearity to model
#  @module Sequential from tensorflow.keras stacks layers
#
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential

## The bicoder captures the direct relationship between encoders and decoders
#  We know that encoders learn a latent representation z and are used as
#  input to sample new data and get xhat in the decoder
#  There's a direct mechanism to get this values
#
class BiCoder(layers.Layer):
    # Set default parameters that are common in
    # encoder and decoders as static variables

    # Both encoder and decoders for black and white and color images
    _activation                = "relu"
                                               
    # Encoder and decoder black and white images
    _latentDimensionBlackWhite = 20 

    # Encoder and decoder color images
    _filtersColor              = 32
    _latentDimensionColor      = 50
    _stridesColor              = 2
    _kernelSizeColor           = 3
    _paddingColor              = "same"
   
    ## Computes the z encoder from a posterior q(z|x) distribution
    #  @param output encoder from the Black and White MLP or Color Convolutional Neural Network
    #  @param latentDimension either from Black and White or Color images
    #  @return z as encoder
    #
    @staticmethod
    def _calculateZPosteriorDistribution(output, latentDimension):

    ## Computes the xhat decoder where z is used as input
    #  @param mean of decoder from the Black and White MLP or Color Convolutional Neural Network
    #  @param standardDeviation fixed as it is challenging
    #  @return xhat decoder as sample of new data
    #
    @staticmethod
    def _calculateXhatPosteriorDistribution(mean, standardDeviation = 0.75):

