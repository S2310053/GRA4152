##
#  This module defines the BiCoder mix-in class.
#  It provides shared latent-distribution utilities for both the Encoder
#  and the Decoder, keeping common probabilistic operations in one place.
#
#  Some useful information
#    - Inheritance: BiCoder acts as a mix-in, supplying shared behavior
#      to Encoder and Decoder without defining a full hierarchy.
#    - Decorators: static-method decorators are used because these
#      numerical transformations do not depend on instance state.
#
#  @library tensorflow  for tensor operations and sampling
##
import tensorflow as tf


##
#  BiCoder supplies formulas used by both encoders and decoders:
#    - z posterior sampling (reparameterization)
#    - x̂ posterior mean output
#
#  The class stores latent dimensionalities as shared attributes.
#
class BiCoder:

    # Shared architecture hyperparameters
    _activationBW              =  "relu"
    _latentDimBW               =   20
    _unitsBW                   =  400

    _filtersColor              =   32
    _latentDimColor            =   50
    _stridesColor              =    2
    _kernelColor               =    3
    _paddingColor              =  "same"

    ##
    #  Static utility: computes the posterior distribution q(z|x)
    #  via the reparameterization trick.
    #
    #  Decorator:
    #    - Static method, as this transformation does not rely on instance state.
    #
    #  @param output tensor containing concatenated (mu, log_var)
    #  @param latentDim number of latent dimensions
    #
    #  @return (z, mu, log_var)
    #
    @staticmethod
    def _calculateZPosterior(output, latentDim):

        mu       =  output[:, :latentDim]
        log_var  =  output[:, latentDim:]

        std      =  tf.exp(0.5 * log_var)
        eps      =  tf.random.normal(tf.shape(mu))

        return mu + std * eps, mu, log_var


    ##
    #  Static utility: returns the mean of p(x|z).
    #  No sampling is used for stability.
    #
    #  Decorator:
    #    - Static method since this is a simple deterministic mapping.
    #
    #  @param mean reconstructed mean tensor
    #  @return mean reconstruction
    #
    @staticmethod
    def _calculateXhatPosterior(mean):

        return mean



