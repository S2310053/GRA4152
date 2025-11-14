##
#  This module defines the DataLoader class
#  Processes and loads the data
#

# Libraries
import numpy as np
import pickle


## Load and process mnist_bw data
#  Needs:
#  Train  <- mnist_bw.npy
#  Test   <- mnist_bw_te.npy
#  Labels <- mnist_bw_y_te.npy

# Load data
filenameDataTrainBlackWhite = "mnist_bw.npy"
filenameDataTestBlackWhite  = "mnist_bw_te.npy"
filenameDataLabelsBlackWhite = "mnist_bw_y_te.npy"

dataTrainBlackWhite  = np.load(filenameDataTrainBlackWhite)
dataTestBlackWhite   = np.load(filenameDataTestBlackWhite)
dataLabelsBlackWhite = np.load(filenameDataTestBlackWhite)

## Transform data
#  Scalling: making the image be in interval 0 and 1
#  normalizing by dividing data(test and train) / 255
#  Vectorization: each image is vector (shape) 28 * 28 = 784 dimensions
#  reshaping training set by shape (60000, 784)
#  reshaping test set by shape (10000, 784)

# Scalling
dataTrainBlackWhite = dataTrainBlackWhite.astype("float32") / 255
dataTestBlackWhite  = dataTestBlackWhite.astype("float32")  / 255
                                                
# Vectorization
dataTrainBlackWhite = dataTrainBlackWhite.reshape((60000, 28 * 28))
dataTestBlackWhite  = dataTestBlackWhite.reshape(( 10000, 28 * 28))

#print(dataTrainBlackWhite.shape)
#print(dataTestBlackWhite.shape)

## Load (no further processing) mnist_color data
#  Needs:
#  Train  <- mnist_color.pkl
#  Test   <- mnist_color_te.pkl
#  Labels <- mnist_color_y_te.npy

# Load data 
filenameDataTrainColor  = "mnist_color.pkl"
filenameDataTestColor   = "mnist_color_te.pkl"
filenameDataLabelsColor = "mnist_color_y_te.npy"

with open(filenameDataTrainColor, "rb") as file:
    data = pickle.load(file)
