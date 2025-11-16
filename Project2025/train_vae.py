import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#from mlp_toolkits.axes_grid1 import ImageGrid
from encoder import Encoder
from decoder import Decoder
from dataloader import DataLoader
from vae import VAE


# This PROGRAM represents the pseoudocode for the trian_vae.py
# 

# i) Load the dataset given args arguments
my_data_loader = DataLoader("mnist_bw")

# ii) Initialize the VAE model
model = VAE() # Use default values as prof, or pass arguments with argpase

# iii) Set the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

# iv) Invoke the method train using mini batch from my_data_loader
tr_data = my_data_loader.loadData("mnist_bw")

# Complete number of epochs pass through entire training set 
# (one forward pass  + one backward pass)
# forward take inputs, makes predictions and computes errors
# backward adjusts weights based on the errors to improve future predictions
for e in range(20):
        for i, tr_batch in enumerate(tr_data):
                    loss = model.train(tr_batch, optimizer)

