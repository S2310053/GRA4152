from dataloader import DataLoader
from vae import VAE
import tensorflow as tf

loader = DataLoader("mnist_bw")
train_ds = loader.loadData("mnist_bw")   # (batch, 784)

model = VAE()
optimizer = tf.keras.optimizers.Adam(1e-3)

for epoch in range(10):
    for step, batch in enumerate(train_ds):
        loss = model.train_step(batch, optimizer, color=False)
    print(f"Epoch {epoch+1}, loss = {loss.numpy():.4f}")

