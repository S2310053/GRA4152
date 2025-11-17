##
#  This module defines the DataLoader class.
#  It downloads, loads, validates, standardizes, and batches MNIST data.
#
#  Useful information:
#    - Private instance variables (underscore prefix) help encapsulate internal state
#    - Private helper methods structure the workflow and reduce external exposure
#    - Behavior adapts depending on dataset type (black–white or color)
#
#  Debugging tools included:
#    - Exceptions and targeted exception handlers
#    - Assertions for input/output assumptions
#    - Defensive programming checks during standardization and shape validation
#
#  @library numpy      for array handling
#  @library os         to check file existence and suppress TF logs
#  @library tensorflow for creating the batched Dataset
#  @library subprocess to execute wget when needed
##
import subprocess
import pickle
import numpy     as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  =  "3"
import tensorflow as tf


##
#  DataLoader encapsulates the full MNIST preparation pipeline.
#
#  Private instance variables:
#      _datasetName  selected dataset name
#      _urlBW        URL for black–white dataset
#      _urlColor     URL for color dataset
#
#  Public method:
#      loadData()
#
#  Private helper methods:
#      _downloadData(), _standardize(), _validateShape(), _ensureFiles()
class DataLoader:

    ##
    #  Constructor.
    #
    #  @param datasetName string: "mnist_bw" or "mnist_color"
    def __init__(self, datasetName):

        valid  =  ["mnist_bw", "mnist_color"]
        if datasetName not in valid:
            raise ValueError(f"Dataset must be one of {valid}")

        self._datasetName  =  datasetName

        self._urlBW        =  (
            "https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/"
            "mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0"
        )

        self._urlColor     =  (
            "https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/"
            "mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0"
        )


    ##
    #  Private: converts pixel arrays into float32 and normalizes to [0,1].
    #  
    #  Pixel ranges may differ across datasets (not always 0–255).
    #  Normalizing to [0,1] using (x - min) / (max - min) adapts to the true
    #  dynamic range of the data. This prevents degenerate cases where all
    #  values collapse toward zero (e.g., incorrect division by 255), which
    #  would lead to black images and unstable VAE training.
    #
    #  @param arr numpy array with pixel values in [0,255]
    #  @return    standardized array in float32
    def _standardize(self, arr):

        try:
            arr  =  arr.astype("float32")
            mn   =  arr.min()
            mx   =  arr.max()
            arr  =  (arr - mn) / (mx - mn + 1e-8)
        except Exception:
            raise ValueError("Standardization failed: non-numeric data.")

        assert arr.min() >= 0.0 and arr.max() <= 1.0, \
            "Standardization failed: values not within [0,1]."

        return arr


    ##
    #  Private: validates the shape of loaded image data.
    #
    #  @param arr numpy array
    #  @param expectedChannels number of color channels
    def _validateShape(self, arr, expectedChannels):

        assert arr.ndim       == 4, \
               "Images must be 4D: (N, H, W, C)."

        assert arr.shape[-1]  == expectedChannels, \
               f"Expected {expectedChannels} channels but got {arr.shape[-1]}."

        assert arr.shape[1]   == 28 and arr.shape[2] == 28, \
               "Image resolution must be 28 x 28."


    ##
    #  Private: downloads a dataset via wget.
    #
    #  Polymorphism:
    #    - Behavior adapts to dataset type through a unified interface.
    #
    #  @param name dataset name
    def _downloadData(self, name):

        if name == "mnist_bw":
            subprocess.run(["wget", "-O", "mnist_bw.npy",    self._urlBW])

        elif name == "mnist_color":
            subprocess.run(["wget", "-O", "mnist_color.pkl", self._urlColor])

        else:
            raise ValueError("Invalid dataset name for download.")


    ##
    #  Private: ensures required data files exist in the directory.
    def _ensureFiles(self):

        if self._datasetName == "mnist_bw":
            if not os.path.exists("mnist_bw.npy"):
                print("mnist_bw.npy not found — downloading...")
                self._downloadData("mnist_bw")

        elif self._datasetName == "mnist_color":
            if not os.path.exists("mnist_color.pkl"):
                print("mnist_color.pkl not found — downloading...")
                self._downloadData("mnist_color")


    ##
    #  Public: loads and returns a batched TensorFlow Dataset.
    #
    #  @param name      dataset name ("mnist_bw" or "mnist_color")
    #  @param keyColor  dictionary key for the color dataset
    #  @param batchSize batch size
    #
    #  @return batched tf.data.Dataset
    def loadData(self, name, keyColor="m4", batchSize=32):

        self._ensureFiles()

        if name == "mnist_bw":

            try:
                raw   =  np.load("mnist_bw.npy")
            except Exception:
                raise IOError("Could not load mnist_bw.npy")

            data  =  self._standardize(raw).reshape(-1, 28, 28, 1)
            self._validateShape(data, expectedChannels=1)

        elif name == "mnist_color":

            try:
                with open("mnist_color.pkl", "rb") as f:
                    rawDict  =  pickle.load(f)
            except Exception:
                raise IOError("Could not load mnist_color.pkl")

            if keyColor not in rawDict:
                available  =  list(rawDict.keys())
                raise ValueError(
                    f"Invalid keyColor '{keyColor}'. Available keys: {available}"
                )

            data  =  self._standardize(rawDict[keyColor])
            self._validateShape(data, expectedChannels=3)

        else:
            raise ValueError(f"Unknown dataset '{name}'.")

        return tf.data.Dataset.from_tensor_slices(data).batch(batchSize)







