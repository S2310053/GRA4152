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
import numpy as np
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
        #  Always initialize for best practice
        def __init__(self):
            super().__init__()

            #  Define the MLP decoder ONCE
             self.decoder_mlp = Sequential([
                     layers.InputLayer(input_shape=(BiCoder._latentDimensionBlackWhite,)),
                     layers.Dense(BiCoder._unitsBlackWhite, activation=BiCoder._activation),
                     layers.Dense(self._outputDimensionBlackWhite),
             ])

        # Define the CNN decoder ONCE
             self.decoder_cnn = Sequential([
                     layers.InputLayer(input_shape=(BiCoder._latentDimensionColor,)),
                     layers.Dense(units=self._unitsColor, activation=BiCoder._activation),
                     layers.Reshape(target_shape=self._targetShapeColor),
                     layers.Conv2DTranspose(
                             filters=2 * BiCoder._filtersColor,
                             kernel_size=BiCoder._kernelSizeColor,
                             strides=BiCoder._stridesColor,
                             padding=BiCoder._paddingColor,
                             output_padding=0,
                             activation=BiCoder._activation),
                     layers.Conv2DTranspose(
                             filters=BiCoder._filtersColor,
                             kernel_size=BiCoder._kernelSizeColor,
                             strides=BiCoder._stridesColor,
                             padding=BiCoder._paddingColor,
                             output_padding=1,
                             activation=BiCoder._activation),
                     layers.Conv2DTranspose(
                             filters=self._channelOutputColor,
                             kernel_size=BiCoder._kernelSizeColor,
                             strides=BiCoder._stridesColor,
                             padding=BiCoder._paddingColor,
                             output_padding=1),
                     layers.Activation("linear", dtype="float32"),
        ])
        
   
        ## Generate the decoder xhat for black and white images
        #  Gaussian decoder for vectorized images
        #  MLP models with one hidden layer
        #  recall BiCoder._calculateXhatPosteriorDistribution
        #             transform data with equation used 
        #             to get x from posterior distribution
        #  @param     dataZ (black and white images) from encoder
        #  @return    mean (Gaussian distribution) of the MLP model
        #

        def getDecoderMLP(self, z):
                output = self.decoder_mlp(z)
                return BiCoder._calculateXhatPosteriorDistribution(output)

        ## Generate the decoder z for the color images
        #  Convolutional neural network decoder
        #  @recall BiCoder._calculateXhatPosteriorDistribution transform 
        #             output with equation x from posterior distribution
        #  @param     dataZ (color images) from encoder
        #  @return    mean (Gaussian distribution) of convolutional neural network model
        #
        def getDecoderCNN(self, z):
                output = self.decoder_cnn(z)
                return BiCoder._calculateXhatPosteriorDistribution(output)
