##
#  This module defines the Decoder class
#  Contains the specific behavior in decoders
#

# Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential


#####################################
#
# Decoder Network for black and white images
# Both encoder and decoder are MLP models
# with one hidden layer
# Output: mu parameter of the Gaussian distribution
# treat sigma = 0.75 (fixed) as known to make things easier
# x = mu + std*eps
#
#####################################

# Parameters
output_dim = 28 * 28

# Gaussian decoder for B&W images
decoder_mlp = Sequential(
                        [
                        layers.InputLayer(input_shape = laten_dim), #latent_dim parameter from bw encoder
                        layers.Dense(units, activation = activation), #units and activation params from bw encoder
                        layers.Dense(output_dim)
                        ]
                        )

# Output
mu  = decoder_mlp(z) #z from bw encoder
std = 0.75 #we don't need to learn sigma as it is challenging, we assume this value
eps = tf.random.normal(mu.shape) #we don't retrieve as random behave separately
x   = mu + std*eps

#####################################
#
# Decoder Network for the color MNIST images
# Architecture based on convolutional neural networks
# Output: mu parameter of the Gaussian distribution
# treat sigma = 0.75 (fixed) as known to make things easier
# x = mu + std*eps
#
#####################################

# Parameters
target_shape = (4, 4, 128)
channel_out  = 3
units        = np.prod(target_shape)

# Convolutional neural network decoder for color MNIST images
decoder_conv = Sequential(
                         [
                         layers.InputLayer(input_shape = (latent_dim, )), #latent_dim from encoder
                         layers.Dense(units = units, activation = activation), #activation from encoder
                         layers.Reshape(target_shape = target_shape),
                         layers.Conv2DTranspose(
                             filters = 2 * filters, kernel_size = kernel_size, strides = 2, padding = "same", output_padding = 0,
                             activation = activation), #all object parameters from encoder
                         layers.Conv2DTranspose(
                             filters = filters, kernel_size = kernel_size, strides = 2, padding = "same", output_padding = 1,
                             activation = activation), #all object parameters from encoder
                         layers.Conv2DTranspose(
                             filters = channel_out, kernel_size = kernel_size, strides = 2, padding = "same", output_padding = 1),
                         layers.Activation("linear", dtype = "float32"), #all object parameters are from encoder **channel_out not
                         ] #look that strides= has a number here and padding= same situation but same values as in encoder
                         )
# Output
mu = decoder_conv(z) #z from encoder
# We dont' need to learn sigma bc it is challenging
# Assume a fixed value
std = 0.75 #just as decoder for bw images
eps = tf.random.normal(mu.shape)
x   = mu + std*eps
