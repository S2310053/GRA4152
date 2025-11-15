##
#  This module defines the VAE class 
#  Contains the behavior in VAEs
#

## Load necessary libraries and packages
#  @library numpy for vectorization and data handling
#  @library os, os.environ 3 to just include tensorflow error messages
#  @library tensorflow let us train our model  
#  @module tensorflow.keras for MLP encoder and decoder models
#  
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential

## Variational Autoencoders are models in generative artificial intelligence
#  They combine deep learning and probabilistic modeling
#  Which enables generative modelling and representation learning
#
class VAE:
