## 
#  This module defines the Encoder class 
#  Contains specific behavior in encoders
#

## Load necessary libraries and modules
#  @library os, os.environ 3 to just include tensorflow error messages
#  @library tensorflow to generate isotropic p(z) distribution
#  @library tensorflow_probability to generate isotropic correlation
#  @module layers from tensorflow.keras reusable when stating weights NN
#  @module activations from tensorflow.keras adds non-linearity to model
#  @module Sequential from tensorflow.keras stacks layers
#  @module BiCoder retrieves general formulas for z(decoder) and xhat(decoder)
#
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
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

    # Parameter specific for the black and white images MLP model encoder
    _inputShapeBlackWhite      = (28 * 28, )
    _latentDimensionBlackWhite = 20
    _unitsBlackWhite           = 400

    # Parameter specific for the color images convolutional neural network model encoder
    _inputShapeColor           = (28, 28, 3)
    _latentDimensionColor      = 50 

    ## Defines the constructor and set default values
    #
    def __init__(self):

    ## Computes the z encoder value from a prior p(z) distribution
    #  loc as mean value for the first dimension and second dimension
    #  with a variance 1 property, dimensions 2 property
    #  and covariance matri has 0 correlation
    #  Improves efficiency parametrizing by a Cholesky factor of the covariance matrix
    #  Generates 10,000 samples as in Monte Carlo methods
    #  @return prior z from and isotropic Gaussian distribution N(0,I)
    #  
    def getEncoderIsotropic(self):
        _loc                         = [0.0, 0.5]
        _variance                    = 1.0
        _dimensions                  = 2
        _covarianceMatrix            = tf.eye(_dimensions)  * _variance
        _scaleLowerTriangular        = tf.linalg.cholesky(_covarianceMatrix)
        _isotropicNormalDistribution = tfd.MultivariateNormalTriL(loc = loc,
                                                                  scale_tril = scaleLowerTriangular)
        return isotropicNormalDistribution.sample(10_000)

    ## Generates the encoder z for black and white images
    #  Gaussian encoder for vectorized images
    #  MLP models with one hidden layer
    #  @decorator BiCoder._calculateZPosteriorDistribution transform data with equation used to get z from posterior distribution
    #  @param data x (black and white images) from the dataset
    #  @return output parameters (Gaussian distribution) of the MLP model
    #
    @BiCoder._calculateZPosteriorDistribution
    def getEncoderMLP(self, data):
       _encoderMLP = Sequential(
                               [
                               layers.InputLayer(input_shape = _inputShapeBlackWhite)
                               layers.Dense(_unitsBlackWhite)
                               layers.Dense(2 * _latentDimensionBlackWhite)
                               ]
                               )
        return _encoderMLP(data), _latentDimensionBlackWhite


    ## Generate the encoder z for the color images
    #  Convolutional neural network encoder
    #  @decorator BiCoder._calculateZPosteriorDistribution transform output with equation z from posterior distribution
    #  @param data x (color images)
    #  @return output parameters (Gaussian distribution) of convolutional neural network model
    #
    @BiCoder._calculateZPosteriorDistribution
    def getEncoderCNN(self, data):
        _encoderCNN = Sequential(
                                [
                                layers.InputLayer(input_shape = _inputShapeColor),
                                layers.Conv2D(
                                    filters     = 1 * BiCoder._filtersColor,
                                    kernel_size = BiCoder._kernelSizeColor,
                                    strides     = BiCoder._stridesColor,
                                    activation  = BiCoder._activation,
                                    padding     = BiCoder._paddingColor),
                                layers.Conv2D(
                                    filters     = 2 * BiCoder._filtersColor,
                                    kernel_size = BiCoder._kernelSizeColor,
                                    strides     = BiCoder._stridesColor,
                                    activation  = BiCoder._activation,
                                    padding     = BiCoder._paddingColor),
                                layers.Conv2D(
                                    filters     = 4 * BiCoder._filtersColor,
                                    kernel_size = BiCoder._kernelSizeColor,
                                    strides     = BiCoder._stridesColor,
                                    activation  = BiCoder._activation,
                                    padding     = BiCoder._paddingColor)
                                layers.Flatten(),
                                layers.Dense(2 * _latentDimensionColor)
                                ]
                                )
        return _encoderCNN(data), _latentDimensionColor
