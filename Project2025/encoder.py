##
#  This module defines the Decoder class
#  Contains the specific behavior in encoders
#

#####################################

# Encoder network for bw

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
out     = encoder_mpl(x)
mu      = out[:,:latent_dim]
log_var = out[:,latent_dim:]
std     = tf.math.exp(0.5 * log_var)

# stopped at output_dim = 28*28


#####################################

# Encoder network for color

#####################################
