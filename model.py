# Function to create an encoder model that takes an input of shape input_dim and maps it to a latent space of size latent_dim
def create_encoder(input_dim, latent_dim):
    # Define input layer
    input_layer = Input(input_dim)
    # Reshape the input data
    layer = Reshape((input_dim[0], input_dim[1]), name="reshape_encoder")(input_layer)

    # Add convolutional layers with increasing number of filters
    for i in range(5):
        layer = Conv1D(filters=2**(i+4), kernel_size=1, padding='same', activation='tanh')(layer)
        layer = ReLU()(layer)
        layer = BatchNormalization()(layer)
        
    # Add a max pooling layer
    layer = MaxPooling1D(pool_size=input_dim[0])(layer)

    # Output the mean and log variance of the distribution in the latent space
    output1 = Dense(units=latent_dim, name='mean_z')(layer)
    output1 = Reshape((latent_dim,))(output1)
    output2 = Dense(units=latent_dim, name='log_variance_z', kernel_initializer=tf.keras.initializers.Zeros())(layer)
    output2 = Reshape((latent_dim,))(output2)

    # Return the keras model
    return Model(input_layer, [output1, output2])


# Function to create a decoder model that takes an input of size latent_dim and maps it to the original data shape of input_dim
def create_decoder(input_dim, latent_dim):
    # Define input layer
    input_layer = Input(shape=(latent_dim,), name='z_sampling')
    
    # Add dense layers to decode the latent space to the original data shape
    layer = Dense(units = 24, activation = "tanh")(input_layer)
    layer = Dense(units = 48, activation = "tanh")(layer) 
    layer = Dense(units = input_dim[0]*input_dim[1])(layer)
    
    # Reshape the output to the original data shape
    output = Reshape((input_dim[0], input_dim[1]), name="reshape_decoder")(layer)
         
    return Model(input_layer, output)


# Layer class to implement sampling from a Gaussian distribution parameterized by the mean and log variance
class Sampling_log(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5*z_var) * epsilon
    

# Function to create a custom loss function for the earth mover's distance
def create_earth_mover_distance():
    def earth_mover_distance(p,q):
        x = tf.sqrt(tf.reduce_sum(tf.square(p - q), 1)) #car dim 2
        y = tf.cumsum(x)
        return tf.reduce_sum(tf.abs(y))
    return earth_mover_distance
        

# Implementation of the beta VAE that will link the encoder part, the reparameterization trick, and the decoder part
class VAE(Model):
    def __init__(self, kl_rate, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Create the encoder model
        self.shape_encoder = create_encoder(input_dim, latent_dim)
        # Create the decoder model
        self.decoder = create_decoder(input_dim, latent_dim)
        # Set the value of the hyperparameter for the KL loss
        self.kl_rate = kl_rate
      
# Definition os the call methos of the VAE class
    def call(self, x):
        # Set shape to be the input data
        shape = x

        # EPS is a small value used to avoid numerical instability in KL divergence calculation
        EPS = K.epsilon()

        # Encoding part
        # Pass input data through the shape encoder to obtain mu_shape and log_var_shape
        mu_shape, log_var_shape = self.shape_encoder(shape)

        # Reparameterization trick
        # Sample a point in the latent space using mu_shape and log_var_shape
        sample_shape = Sampling_log()([mu_shape, log_var_shape])

        # Decoding part
        # Pass the sampled point through the decoder to obtain the reconstructed shape
        decoded_shape = self.decoder(sample_shape)

        # Reconstruction loss
        # Use custom earth mover's distance loss to compute the difference between the input shape and the decoded shape
        custom_loss = create_earth_mover_distance()
        emd_loss = custom_loss(shape, decoded_shape)

        # KL divergence between the Gaussian distribution predicted by the encoder and a Gaussian reduced multivariate normal distribution. 
        # Compute the KL divergence between the learned distribution and a standard normal distribution
        kl_loss = K.sum(K.square(mu_shape) + K.exp(log_var_shape) - 1 - log_var_shape + EPS)

        # Final cost function
        # Combine the reconstruction loss and the KL divergence loss using a weighting factor
        loss = self.kl_rate*kl_loss + emd_loss

        # Add the cost function and metrics to the model
        self.add_loss(loss)
        self.add_metric(tf.reduce_sum(emd_loss), name="emd_loss", aggregation='mean')
        self.add_metric(tf.reduce_sum(kl_loss), name="kl_loss", aggregation='mean')

        # Return the decoded data by default
        return decoded_shape

    #Object in order to build a VAE, compile and plot the architecture
    def build_vae(kl_rate, input_dim, latent_dim, verbose = True):
        vae = VAE(kl_rate, input_dim, latent_dim)
        # Compile the model with Adam optimizer
        vae.compile(optimizer="Adam")
        # If verbose is True, print a summary of the shape encoder and decoder models
        if verbose:
            vae.shape_encoder.summary()
            print('-'*50)
            vae.decoder.summary()
        # Return the compiled VAE model
        return vae
