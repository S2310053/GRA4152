##
#  This module defines the Decoder class
#  Contains specific behavior in decoders
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

## Probabilistic decoders take the input z of data generated from the encoder
#  and sample new data x using the location scale approach
#
class Decoder(layers.Layer, BiCoder):
        # Default parameters specific for this class
        # Set as static variables

        # Parameters specific for the black and white images MLP model decoder
        _outputDimensionBlackWhite = (28 * 28)

        # Parameter specific for the color images convolutional neural network model decoder
        _targetShapeColor   = (4, 4, 128)
        _channelOutputColor = 3
        _unitsColor         = np.prod(_targetShapeColor)

        ## Initialize class and set default parameters
        #
        def __init__(self):
   
        ## Generate the decoder xhat for black and white images
        #  Gaussian decoder for vectorized images
        #  MLP models with one hidden layer
        #  @decorator BiCoder._calculateXhatPosteriorDistribution
        #             transform data with equation used 
        #             to get x from posterior distribution
        #  @param     data x (black and white images) from encoder
        #  @return    mean (Gaussian distribution) of the MLP model
        #
        @BiCoder._calculateXhatPosteriorDistribution
        def getDecoderMLP(self,newdata):


        ## Generate the decoder z for the color images
        #  Convolutional neural network decoder
        #  @decorator BiCoder._calculateXPosteriorDistribution transform 
        #             output with equation x from posterior distribution
        #  @param     data x (color images) from encoder
        #  @return    mean (Gaussian distribution) of convolutional neural network model
        #
        @BiCoder._calculateXhatPosteriorDistribution
        def getDecoderCNN(self, newdata):

