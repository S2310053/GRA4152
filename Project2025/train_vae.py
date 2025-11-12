## 
#  This PROGRAM represents the pseoudocode for the trian_vae.py
# 

# Load the dataset given args arguments
my_data_loader = DataLoader(dseet=args.dset)

# Initialize the VAE model
model = VAE() # Use default values as prof, or pass arguments with argpase

# Set the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

# Invoke the method train using mini batch from my_data_loader
tr_data = my_data_loader.get_training_data

# Complete number of epochs pass through entire training set 
# (one forward pass  + one backward pass)
# forward take inputs, makes predictions and computes errors
# backward adjusts weights based on the errors to improve future predictions
for e in range(args.epochs):
    for i, tr_batch in enumerate(tr_data):
        loss = model.train(tr_batch, optimizer)
