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

# Load data
loader = DataLoader(args.dset)
train_ds = loader.loadData(args.dset)

# Init model & optimizer
model = VAE()
optimizer = tf.keras.optimizers.Adam(1e-3)

# Train
for epoch in range(args.epochs):
    for batch in train_ds:
        loss = model.train(batch, optimizer, color=(args.dset=="mnist_color"))
    print(f"Epoch {epoch+1}, loss={loss.numpy():.4f}")

# Latent visualization
if args.visualize_latent:
    model.visualize_latent(train_ds, color=(args.dset=="mnist_color"))

# Generation from prior
if args.generate_from_prior:
    imgs = model.generate_from_prior(color=(args.dset=="mnist_color"))
    print(np.min(imgs), np.max(imgs))
    model.plot_grid(imgs, name="generated_prior")

# Generation from posterior
if args.generate_from_posterior:
    imgs = model.generate_from_posterior(train_ds, color=(args.dset=="mnist_color"))
    model.plot_grid(imgs, name="generated_posterior")
