##
#  This module defines the DataLoader class
#  Processes and loads the data
#

# Libraries
import numpy as np


## Load and process mnist_bw data
#  Needs:
#  Train  <- mnist_bw.npy
#  Test   <- mnist_bw_te.npy
#  Labels <- mnist_bw_y_te.npy

filenameDataTrainBlackWhite = "mnist_bw.npy"
filenameDataTestBlackWhite  = "mnist_bw_te.npy"
filenameDataLabelsBlackWhite = "mnist_bw_y_te.npy"

dataTrainBlackWhite  = np.load(filenameDataTrainBlackWhite)
dataTestBlackWhite   = np.load(filenameDataTestBlackWhite)
dataLabelsBlackWhite = np.load(filenameDataTestBlackWhite)

## Load (no further processing) mnist_color data
#  Needs:
#  Train  <- mnist_color.pkl
#  Test   <- mnist_color_te.pkl
#  Labels <- mnist_color_y_te.npy
