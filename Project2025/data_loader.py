##
#  This module defines the DataLoader class
#  Downloads, processes and loads the data
#

# How to use wget with subprocess.run() link
# https://www.webscrapingapi.com/effortlessly-download-web-pages-and-files-with-python-and-wget

# Libraries
import subprocess


##########################################
#
# Adress to get the data set mnist_bw
# file type: npy
# images size: (28,28,1)
#
##########################################

# Train dset
'https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0'

# Test det dset
'https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=0'

# Labels, which are useful to color scatters of the latent space
'https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0'

##########################################
#
# Adress to get the data set mnist_color
# file type: pkl
# images size: (28,28,3)
# labels fyle type: npy
#
###########################################

# Train dset
'https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0'

# Test dset
'https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=0'

# Labels, which are useful to color scatters of the latent space
'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0'



###########################################
# DRAFT CODE STARTS HERE
###########################################

# Think of this as instance variables from calls that are going to be called to other class

# Get the train dataset mnist_bw
mnistbwDatasetTrain = subprocess.run(["wget", 'https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0'])
