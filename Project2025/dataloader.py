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
    ## Identify static variables like the urls of the dropbox files
    #

    # Train set url for black and white images
    urlTrainBlackWhite  = "https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0"

    # Labels url for black and white images
    urlLabelsBlackWhite = "https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0"

    # Train set url for color images
    urlTrainColor       = "https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0"

    # Labels url for color images
    urlLabelsColor      = "https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0"

    ## Initializes constructor of the class
    #  Here we know that data could be different
    def __init__(self, datasetName):
        self.datasetName = datasetName

    ## Downloads data from urls directly in our current directory
    #  @param mnist_bw or mnist_color depending on black and white or color dataset choice
    #  return confirmation of downloading or assertion of files already in directroy
    #
    def downloadData(self, datasetName):
        if datasetName == "mnist_bw":

            _downloadTrain  = subprocess.run(["wget", "-O", "mnist_bw.npy", self.urlTrainBlackWhite])
           #_downloadLabels = subprocess.run(["wget", "-O", "mnist_bw_y_te.npy" ,self.urlLabelsBlackWhite])
            result = "Successful"

        elif datasetName == "mnist_color":
            _downloadTrain  = subprocess.run(["wget", "-O", "mnist_color.pkl" , self.urlTrainColor])
            #_downloadLabels = subprocess.run(["wget", "-O", "mnist_color_y_te.npy", self.urlLabelsColor]) 
            result = "Successful"
        else:
            result = "Not a valid dataset"

        return  print(result)

    ## Loads the data
    #  If the data set belongs to mnist_bw
    #  Transforms the black and white data set
    #  Scales by making the image interval 0 and 1
    #  Normalizes by diving each image with vector 28 * 28 = 784 dimensions
    #  Reshapes the training set (60_000, 784)
    #  If the data set belongs to mnist_color, chooses default dictionary value
    #  
    def loadData(self, datasetName, keyColor = "m0"):
        if datasetName == "mnist_bw":
            _rawDataTrain = np.load("mnist_bw.npy")
           # _labels       = np.load("mnist_bw_y_te.npy")
            _dataTrain    = (_rawDataTrain.astype("float32") / 255).reshape(60_000, 28 * 28)
            data          = tf.data.Dataset.from_tensor_slices(_dataTrain).batch(32)

        elif datasetName == "mnist_color":
           # _labels       = np.load("mnist_color_y_te.npy")
            with open("mnist_color.pkl", "rb") as file:
                _rawDataTrain = pickle.load(file)
            _dataTrain   = np.array(_rawDataTrain[keyColor], dtype = "float32")
            data         = tf.data.Dataset.from_tensor_slices(_dataTrain).batch(32)
       
        return data


# Test class
mydata = DataLoader("mnist_color")
#download = mydata.downloadData("mnist_color")
data     = mydata.loadData("mnist_color")
