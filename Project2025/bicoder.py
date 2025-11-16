class BiCoder:
    # Shared architecture hyperparameters
    _activation                = "relu"

    _latentDimensionBlackWhite = 20
    _unitsBlackWhite           = 400

    _filtersColor              = 32
    _latentDimensionColor      = 50
    _stridesColor              = 2
    _kernelSizeColor           = 3
    _paddingColor              = "same"

    @staticmethod
    def _calculateZPosteriorDistribution(output, latentDimension):
        """Reparameterization trick: z = mu + sigma * eps"""
        mu        = output[:, :latentDimension]
        log_var   = output[:, latentDimension:]
        std       = tf.exp(0.5 * log_var)
        eps       = tf.random.normal(tf.shape(mu))
        return mu + std * eps, mu, log_var

    @staticmethod
    def _calculateXhatPosteriorDistribution(mean):
        """Return mean of p(x|z). No sampling for training stability."""
        return mean

