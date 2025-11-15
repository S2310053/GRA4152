## 
#  This module defines the Encoder class 
#  Contains specific behavior in encoders
#

## Load necessary libraries and modules
#  @library os, os.environ 3 to just include tensorflow error messages
#  @module layers from tensorflow.keras reusable when stating weights NN
#  @module activations from tensorflow.keras adds non-linearity to model
#  @module Sequential from tensorflow.keras stacks layers
#  @module BiCoder retrieves general formulas for z(decoder) and xhat(decoder)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
from bicoder import BiCoder

## Probabilistic encoders take the input of data x and learn a latent
#  representation z used as input of the generative model p(x|z) (decoder)
#
class Encoder(layers.Layer, BiCoder):
    # Default parameters specific for this class
    # Set as static variables

    # Parameters specific for the black and white images MLP model encoder
    _inputShapeBlackWhite = (28 * 28, )
    _unitsBlackWhite      = 400

    # Parameter specific for the color images convolutional neural network model encoder
    _inputShapeColor      = (28, 28, 3)

    ## Define the constructor and set default values
    #
    def __init__(self):

    ## Generate the encoder z for black and white images
    #  Gaussian encoder for vectorized images
    #  MLP models with one hidden layer
    #  @decorator BiCoder._calculateZPosteriorDistribution transform data with equation used to get z from posterior distribution
    #  @param data x (black and white images) from the dataset
    #  @return output parameters (Gaussian distribution) of the MLP model
    #
    @Bicoder._calculateZPosteriorDistribution
    def getEncoderMLP(self, data):

    ## Generate the encoder z for the color images
    #  Convolutional neural network encoder
    #  @decorator BiCoder._calculateZPosteriorDistribution transform output with equation z from posterior distribution
    #  @param data x (color images)
    #  @return output parameters (Gaussian distribution) of convolutional neural network model
    #
    @Bicoder._calculateZPosteriorDistribution
    def getEncoderCNN(self, data):
