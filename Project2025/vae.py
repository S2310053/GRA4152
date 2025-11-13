##
#  This module defines the VAE class
#  Contains the behaviour in VAEs
#

# Inherite libraries
import tensorflow as tf

# Retrieves information from the encoder,decoder,bicoder classes

##########################################
#
# Data has been normalized between [0,1]
# From previous data loader data(x)
# Before plot, scale output decoder (x_hat)
# To range [0 255]
# And change dtype to unit 8. Use this function
#
##########################################

img = tf.clip_by_value(255 * x_hat, clip_value_min, clip_value_max = 255).numpy().astype(np.uint8)

# Adam optimizer is the default choice
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4) #1e-3 in train_vae.py

##########################################
#
# Train function to update the VAE
# trainable variables
#
##########################################

@tf.function
def train(self, x, optimizer):
    with tf.GradientTape() as tape:
        loss = self.call(x)           #epecial tf method, input only (data), used to specify all steps to arrive to the objective funtion to be optimized
    # slef.vae_los = - ELBO -> all optimizers always minimze so is negative to maximize
    # ELBO is given in Equation 1
    gradients = tape.gradient(self.vae_loss, self.trainable_variables)
    optimzer.apply_gradients(zip(gradients, self.trainable_variables))

    return loss

