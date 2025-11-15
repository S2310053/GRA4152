##
#  This module defines the BiCoder class
#  Contains the common behavior in encoders and decoders
#  

## Load necessary libraries and packages
#  @library os, os.environ 3 to just include tensorflow error messages
#  @module layers from tensorflow.keras reusable when stating weights NN
#  @module activations from tensorflow.keras adds non-linearity to model
#  @module Sequential from tensorflow.keras stacks layers
#
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential

## The bicoder captures the direct relationship between encoders and decoders
#  We know that encoders learn a latent representation z and are used as
#  input to sample new data and get xhat in the decoder
#

#class BiCoder(layers.Layer):
