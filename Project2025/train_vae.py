import argparse
from dataloader import DataLoader
from vae import VAE
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dset", type=str, default="mnist_bw")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--visualize_latent", action="store_true")
parser.add_argument("--generate_from_prior", action="store_true")
parser.add_argument("--generate_from_posterior", action="store_true")
args = parser.parse_args()

# train_vae.py

loader   = DataLoader(args.dset)
train_ds = loader.loadData(args.dset)

model     = VAE()
optimizer = tf.keras.optimizers.Adam(1e-3)
isColor   = (args.dset == "mnist_color")

for epoch in range(args.epochs):
    for batch in train_ds:
        loss = model.train(batch, optimizer, color=isColor)
    print(f"Epoch {epoch+1}, loss={loss.numpy():.4f}")

if args.visualize_latent:
    model.visualizeLatent(train_ds, color=isColor)

if args.generate_from_prior:
    imgs = model.generateFromPrior(color=isColor)
    model.plotGrid(imgs, color=isColor, name="generated_prior")

if args.generate_from_posterior:
    imgs = model.generateFromPosterior(train_ds, color=isColor)
    model.plotGrid(imgs, color=isColor, name="generated_posterior")

