##
#  This module defines the Decoder class
#  Contains the specific behavior in encoders
#

#####################################
#
# Encoder network for black and white images
# Both encoder and decoder are MLP models
# with one hidden layer
# Output: parameters of the Gaussian distribution
# z = mu + std*epsilon
#
#####################################

#Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential

# Parameters
input_shape = (28*28, )
units       = 400
activation  = "relu"
latent_dim  = 20

# Gaussian encoder for vectorized images
encoder_mlp = Sequential(
                        [
                        layers.InputLayer(input_shape = input_shape),
                        layers.Dense(units,activation = activation),
                        layers.Dense(2 * latent_dim)
                        ]
                        )  
# Output
out     = encoder_mpl(x)       #missing data x
mu      = out[:, :latent_dim]
log_var = out[:, latent_dim:]
std     = tf.math.exp(0.5 * log_var)

# stopped at output_dim = 28*28


#####################################
#
# Encoder network for color MNIST images
# Architecture based on convolutional neural networks
# Output: parameters of the Gaussian distribution
# z = mu + std*epsilon
#
#####################################

# Parameters
input_shape  = (28, 28, 3)
filters      = 32
kernel_size  = 3
strides      = 2
activation   = "relu"
latent_dim   = 50

# Convolutional neural network encoder for color MNIST images
encoder_conv = Sequential(
                         [
                         layers.InputLayer(input_shape = input_shape),
                         layers.Conv2D(
                             filters = filters, kernel_size = kernel_size, strides = strides, activation = activation, padding = "same"),
                         layers.Conv2D(
                             filters = 2 * filters, kernel_size = kernel_size, strides = strides, activation = activation, padding = "same"),
                         layers.Conv2D(
                             filters = 4 * filters, kernel_size = kernel_size, strides = strides, activation = activation, padding = "same"),
                         layers.Flatten(),
                         layers.Dense(2 * latent_dim)
                         ]
                         )

# Output
out     = encoder_conv(x)
mu      = out[:, :latent_dim]
log_var = out[:, latent_dim:]
std     = tf.math.exp(0.5 * log_var)
eps     = tf.random.normal(mu.shape)
z       = mu + std*eps

# Stopped at target_shape = (4,4,128)


