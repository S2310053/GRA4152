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
# decoder.py  (only __init__ modified)

class Decoder(layers.Layer, BiCoder):

    _outputDimensionBlackWhite = (28 * 28)
    _targetShapeColor   = (4, 4, 128)
    _channelOutputColor = 3
    _unitsColor         = np.prod(_targetShapeColor)

    def __init__(self):
        super().__init__()

        # MLP decoder for black-and-white images
        self.decoder_mlp = Sequential([
            layers.InputLayer(input_shape=(BiCoder._latentDimensionBlackWhite,)),
            layers.Dense(BiCoder._unitsBlackWhite, activation=BiCoder._activation),
            # Output in [0,1] to match normalized pixels
            layers.Dense(self._outputDimensionBlackWhite, activation="sigmoid"),
        ])

        # CNN decoder for color images
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
            # Final probabilities in [0,1]
            layers.Activation("sigmoid", dtype="float32"),
        ])

    def getDecoderMLP(self, z):
        output = self.decoder_mlp(z)
        return BiCoder._calculateXhatPosteriorDistribution(output)

    def getDecoderCNN(self, z):
        output = self.decoder_cnn(z)
        return BiCoder._calculateXhatPosteriorDistribution(output)
