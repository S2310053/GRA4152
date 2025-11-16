##
#  This module defines the DataLoader class
#  Downloads, loads and transforms data sets
#

## Loads necessary libraries and packages
#  @library numpy for vectorization and data handling
#  @library os, os.environ 3 to just include tensorflow error messages
#  @library tensorflow let us handle the data
#  @library subprocess when determining commands used by wget when downloading
import subprocess
import pickle
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import tensorflow as tf


## A loading class unzips data, tranforms black and white images set
#  by scalling and vectorization and loads and returns processsed data
#
class DataLoader():

    urlTrainBlackWhite  = "https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0"
    urlTrainColor       = "https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0"

    def __init__(self, datasetName):
        self.datasetName = datasetName

    def downloadData(self, datasetName):
        if datasetName == "mnist_bw":
            subprocess.run(["wget", "-O", "mnist_bw.npy", self.urlTrainBlackWhite])
            print("Downloaded mnist_bw.npy")

        elif datasetName == "mnist_color":
            subprocess.run(["wget", "-O", "mnist_color.pkl", self.urlTrainColor])
            print("Downloaded mnist_color.pkl")

        else:
            print("Not a valid dataset")

    # NEW: ensure data exists before loading
    def ensureDataFiles(self):
        if self.datasetName == "mnist_bw":
            if not os.path.exists("mnist_bw.npy"):
                print("mnist_bw.npy not found, downloading...")
                self.downloadData("mnist_bw")

        elif self.datasetName == "mnist_color":
            if not os.path.exists("mnist_color.pkl"):
                print("mnist_color.pkl not found, downloading...")
                self.downloadData("mnist_color")

    def loadData(self, datasetName, keyColor="m4"):
        # Ensure files exist before loading
        self.ensureDataFiles()

        if datasetName == "mnist_bw":
            _raw = np.load("mnist_bw.npy")  
            _data = (_raw.astype("float32") / 255.0).reshape(60_000, 28, 28, 1)

        elif datasetName == "mnist_color":
            with open("mnist_color.pkl", "rb") as f:
                raw = pickle.load(f)

            if keyColor not in raw:
                raise ValueError(f"Invalid color key '{keyColor}'. Available: {list(raw.keys())}")

            _data = raw[keyColor].astype("float32") / 255.0

        else:
            raise ValueError(f"Unknown dataset '{datasetName}'.")

        return tf.data.Dataset.from_tensor_slices(_data).batch(32)





