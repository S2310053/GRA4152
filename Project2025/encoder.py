## 
#  This module defines the Encoder class 
#  Contains specific behavior in encoders
#

## Load necessary libraries and modules
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

## Probabilistic encoders take the input of data x and learn a latent
#  representation z used as input of the generative model p(x|z) (decoder)
#
class Encoder(layers.Layer, BiCoder)::

