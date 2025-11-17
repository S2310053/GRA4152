##
#  This module defines the Decoder class.
#  It reconstructs images from latent variables z using either:
#      - an MLP decoder for black–white images
#      - a CNN decoder for color images
#
#  Some useful information
#    - Inheritance: derives from keras.Layer to integrate naturally with
#      TensorFlow’s training machinery, and from BiCoder to reuse posterior
#      utilities shared with the encoder.
#    - Overriding: the constructor extends the parent class by defining the
#      trainable decoding components.
#    - Polymorphism: two decoding paths (MLP or CNN) coexist under one public
#      interface, selected implicitly through the latent dimensionality.
#
#  @library tensorflow.keras.layers     decoder layers
#  @module  Sequential                  sequential container
#  @module  BiCoder                     shared x̂ posterior utility
#  @library numpy                       for shape computations
##
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  =  "3"

from tensorflow.keras            import layers
from tensorflow.keras            import activations
from tensorflow.keras.models     import Sequential
from bicoder                     import BiCoder


##
#  Decoder reconstructs images from latent representations z.
#
class Decoder(layers.Layer, BiCoder):

    _outputDimBW       =  (28 * 28)
    _targetColorShape  =  (4, 4, 128)
    _colorChannels     =    3
    _unitsColor        =  np.prod(_targetColorShape)

    ##
    #  Constructor.
    #
    #  Overriding:
    #    - Extends keras.Layer by defining the internal MLP and CNN decoders
    #
    #  Notes:
    #    - BW decoder: a single hidden layer (400 units inherited) followed
    #      by a sigmoid output. Since x ∈ [0,1], sigmoid is a natural choice
    #      to model since we look for assymetry in the data
    #    - Color decoder: starts from a Dense layer that expands z into a
    #      spatial tensor (4×4×128), then three transposed convolutions
    #      progressively upsample it back to 28×28×3
    #    - Strides and kernel sizes mirror the encoder’s structure, ensuring
    #      symmetrical upsampling.
    #    - The final sigmoid constrains reconstructed pixels to [0,1]
    #
    def __init__(self):
        super().__init__()

        ##
        #  MLP decoder for black–white images
        self._decoderBW   =  Sequential([
            layers.InputLayer(
                input_shape = (BiCoder._latentDimBW,)
            ),

            layers.Dense(
                BiCoder._unitsBW,
                activation = BiCoder._activationBW
            ),

            layers.Dense(
                self._outputDimBW,
                activation = "sigmoid"
            ),
        ])

        ##
        #  CNN decoder for color images
        self._decoderColor =  Sequential([
            layers.InputLayer(
                input_shape = (BiCoder._latentDimColor,)
            ),

            layers.Dense(
                units      = self._unitsColor,
                activation = BiCoder._activationBW
            ),

            layers.Reshape(
                target_shape = self._targetColorShape
            ),

            layers.Conv2DTranspose(
                filters        = 2 * BiCoder._filtersColor,
                kernel_size    = BiCoder._kernelColor,
                strides        = BiCoder._stridesColor,
                padding        = BiCoder._paddingColor,
                output_padding = 0,
                activation     = BiCoder._activationBW
            ),

            layers.Conv2DTranspose(
                filters        = 1 * BiCoder._filtersColor,
                kernel_size    = BiCoder._kernelColor,
                strides        = BiCoder._stridesColor,
                padding        = BiCoder._paddingColor,
                output_padding = 1,
                activation     = BiCoder._activationBW
            ),

            layers.Conv2DTranspose(
                filters        = self._colorChannels,
                kernel_size    = BiCoder._kernelColor,
                strides        = BiCoder._stridesColor,
                padding        = BiCoder._paddingColor,
                output_padding = 1
            ),

            layers.Activation("sigmoid", dtype="float32"),
        ])


    ##
    #  Decodes latent vectors for black–white images
    #
    #  Polymorphism:
    #    - The method shares its name with the CNN counterpart, differing
    #      internally in architecture while maintaining a unified interface.
    #
    #  @param z latent tensor
    #
    def getDecoderMLP(self, z):

        output   =  self._decoderBW(z)
        return BiCoder._calculateXhatPosterior(output)


    ##
    #  Decodes latent vectors for color images.
    #
    #  @param z latent tensor
    #
    def getDecoderCNN(self, z):

        output   =  self._decoderColor(z)
        return BiCoder._calculateXhatPosterior(output)

