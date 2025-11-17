##
#  This Program demonstrates the functionalilty of the Variational Autoencoder
#
#  Notes:
#    - The training loop is intentionally simple for clarity
#    - Each optional feature can be called independently from the terminal.
#    - Argparse is used to expose core functionality through concise flags.
#
#  @library argparse         command-line interface
#  @module  DataLoader       loads MNIST data
#  @module  VAE              variational autoencoder model
#  @library tensorflow       optimizer
##
import argparse
from dataloader   import DataLoader
from vae          import VAE
import tensorflow as tf
import numpy      as np


##
#  Command-line arguments.
#  The help strings reflect the expected usage and supported tasks.
#
parser  = argparse.ArgumentParser(
    description = "Train and evaluate the Variational Autoencoder."
)

parser.add_argument(
    "--dset",
    type    = str,
    default = "mnist_bw",
    help    = "Dataset to load: mnist_bw or mnist_color."
)

parser.add_argument(
    "--epochs",
    type    = int,
    default = 10,
    help    = "Number of training epochs."
)

parser.add_argument(
    "--visualize_latent",
    action  = "store_true",
    help    = "Plot a 2D TSNE visualization of the latent space."
)

parser.add_argument(
    "--generate_from_prior",
    action  = "store_true",
    help    = "Generate images by sampling z ~ N(0, I)."
)

parser.add_argument(
    "--generate_from_posterior",
    action  = "store_true",
    help    = "Generate images from q(z|x) using samples from the dataset."
)

args    = parser.parse_args()


##
#  Load dataset.
#
loader    =  DataLoader(args.dset)
train_ds  =  loader.loadData(args.dset)

model     =  VAE()
optimizer =  tf.keras.optimizers.Adam(1e-3)


##
#  Training loop.
#  The flag (color = isColor) selects the correct architecture:
#  MLP for black–white, CNN for color
#  It must be passed explicitly to avoid shape mismatches.
isColor   =  (args.dset == "mnist_color")
for epoch in range(args.epochs):
    for batch in train_ds:
        loss  =  model.train(batch, optimizer, color=isColor)

    print(f"epoch {epoch+1}, loss = {loss.numpy():.4f}")


##
#  Optional tasks.
#
if args.visualize_latent:
    model.visualizeLatent(train_ds, color=isColor)

if args.generate_from_prior:
    imgs = model.generateFromPrior(color=isColor)
    model.plotGrid(imgs, color=isColor, name="generated_prior")

if args.generate_from_posterior:
    imgs = model.generateFromPosterior(train_ds, color=isColor)
    model.plotGrid(imgs, color=isColor, name="generated_posterior")


