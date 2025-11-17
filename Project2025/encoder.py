##
#  This module defines the Encoder class
#  It provides the probabilistic encoders for both black–white and color MNIST
#  The encoder maps input x into the latent Gaussian parameters (mu, log_var).
#
#  Some useful information
#    - Inheritance: the class derives from keras.Layer to integrate naturally
#      with TensorFlow’s variable management, and from BiCoder to reuse shared
#      posterior-distribution formulas.
#    - Overriding: the constructor extends the parent class by defining the
#      trainable components necessary for MLP and CNN encoders.
#    - Polymorphism: different encoder paths (MLP or CNN) are selected based on
#      the input structure while preserving a unified interface.
#
#  @library tensorflow               for tensors and ops
#  @library tensorflow_probability   for Gaussian prior utilities
#  @module  layers                   Keras layers
#  @module  Sequential               sequential model container
#  @module  BiCoder                  shared latent utilities
##
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  =  "3"

import tensorflow                as tf
import tensorflow_probability    as tfp
tfd  =  tfp.distributions

from tensorflow.keras            import layers
from tensorflow.keras.models     import Sequential
from bicoder                     import BiCoder


##
#  Encoder maps input images to latent Gaussian parameters.
#  Two architectures are supported:
#      - MLP for black–white images
#      - CNN for color images
#
class Encoder(layers.Layer, BiCoder):

    # Input shapes for the two encoder types
    _inputBW        =  (28 * 28,)
    _inputColor     =  (28, 28, 3)

    ##
    #  Constructor
    #
    #  Overriding:
    #    - Extends keras.Layer by building internal trainable modules.
    #
    def __init__(self):
        super().__init__()

        ##
        #  Black–white encoder (MLP)
        self._encoderBW    =  Sequential([
            layers.InputLayer(input_shape=self._inputBW),

            layers.Dense(
                BiCoder._unitsBW,
                activation = BiCoder._activationBW
            ),

            # Produces concatenated (mu, log_var)
            layers.Dense(2 * BiCoder._latentDimBW),
        ])

        ##
        #  Color encoder (CNN)
        self._encoderColor =  Sequential([
            layers.InputLayer(input_shape=self._inputColor),

            layers.Conv2D(
                filters     = 1 * BiCoder._filtersColor,
                kernel_size = BiCoder._kernelColor,
                strides     = BiCoder._stridesColor,
                padding     = BiCoder._paddingColor,
                activation  = BiCoder._activationBW
            ),

            layers.Conv2D(
                filters     = 2 * BiCoder._filtersColor,
                kernel_size = BiCoder._kernelColor,
                strides     = BiCoder._stridesColor,
                padding     = BiCoder._paddingColor,
                activation  = BiCoder._activationBW
            ),

            layers.Conv2D(
                filters     = 4 * BiCoder._filtersColor,
                kernel_size = BiCoder._kernelColor,
                strides     = BiCoder._stridesColor,
                padding     = BiCoder._paddingColor,
                activation  = BiCoder._activationBW
            ),

            layers.Flatten(),
            layers.Dense(2 * BiCoder._latentDimColor),
        ])


    ##
    #  Samples from the standard Gaussian prior p(z) = N(0, I).
    #
    #  @param latentDim integer
    #  @param n         number of samples
    #
    #  @return samples from p(z)
    def getEncoderIsotropic(self, latentDim, n=1):

        if latentDim <= 0:
            raise ValueError("Latent dimensionality must be positive.")

        return tf.random.normal(shape=(n, latentDim), dtype=tf.float32)


    ##
    #  Encodes black–white images using the MLP encoder.
    #
    #  Polymorphism:
    #    - One unified interface, different internal behavior depending on data type.
    #
    #  @param x tensor of shape (B, 28*28)
    #  @return  (mu, log_var)
    def getEncoderMLP(self, x):

        output    =  self._encoderBW(x)
        L         =  BiCoder._latentDimBW

        mu        =  output[:, :L]
        log_var   =  output[:, L:]

        return mu, log_var


    ##
    #  Encodes color images using the CNN encoder.
    #
    #  @param x tensor of shape (B, 28, 28, 3)
    #  @return  (mu, log_var)
    def getEncoderCNN(self, x):

        output    =  self._encoderColor(x)
        L         =  BiCoder._latentDimColor

        mu        =  output[:, :L]
        log_var   =  output[:, L:]

        return mu, log_var


