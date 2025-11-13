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
#####################################
