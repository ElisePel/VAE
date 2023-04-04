#predict a gaussian distribution caracterized as mean and variance
def create_encoder(input_dim, latent_dim):
    input_layer = Input(input_dim)
    layer = Reshape((input_dim[0], input_dim[1]), name="reshape_encoder")(input_layer)

    for i in range(5):
        layer = Conv1D(filters=2**(i+4), kernel_size=1, padding='same', activation='tanh')(layer)
        layer = ReLU()(layer)
        layer = BatchNormalization()(layer)
        
    layer = MaxPooling1D(pool_size=input_dim[0])(layer)

    output1 = Dense(units=latent_dim, name='mean_z')(layer)
    output1 = Reshape((latent_dim,))(output1)
    output2 = Dense(units=latent_dim, name='log_variance_z', kernel_initializer=tf.keras.initializers.Zeros())(layer)
    output2 = Reshape((latent_dim,))(output2)

    # Return the keras model
    return Model(input_layer, [output1, output2])


def create_decoder(input_dim, latent_dim):
    input_layer = Input(shape=(latent_dim,), name='z_sampling')
    
    layer = Dense(units = 24, activation = "tanh")(input_layer)
    layer = Dense(units = 48, activation = "tanh")(layer) 
    layer = Dense(units = input_dim[0]*input_dim[1])(layer)
    
    output = Reshape((input_dim[0], input_dim[1]), name="reshape_decoder")(layer)
         
    return Model(input_layer, output)


#Sampling within a distribution characterizing the latent space
class Sampling_log(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5*z_var) * epsilon
    
def create_earth_mover_distance():
    def earth_mover_distance(p,q):
        x = tf.sqrt(tf.reduce_sum(tf.square(p - q), 1)) #car dim 2
        y = tf.cumsum(x)
        return tf.reduce_sum(tf.abs(y))
    return earth_mover_distance
        

#Implementation of the beta VAE that will linked the encoder part, the reparametrization trick and the decoder part
class VAE(Model):
    
    def __init__(self, kl_rate, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.shape_encoder = create_encoder(input_dim, latent_dim)
        self.decoder = create_decoder(input_dim, latent_dim)
        self.kl_rate = kl_rate
        
    def call(self, x):
    
        shape = x
        EPS = K.epsilon()
        
        #Encoding part
        mu_shape, log_var_shape = self.shape_encoder(shape)
        
        #Reparamterization trick
        sample_shape = Sampling_log()([mu_shape, log_var_shape])
        
        #Decoding part
        decoded_shape = self.decoder(sample_shape)
        
        #Reconstruction loss
        custom_loss = create_earth_mover_distance()
        emd_loss = custom_loss(shape, decoded_shape)
        
        #KL divergence between the Gaussian distribution predicted by the coder and a Gaussian reduced multivariate normal distribution. 
        kl_loss = K.sum(K.square(mu_shape) + K.exp(log_var_shape) - 1 - log_var_shape + EPS)

        #Final cost function
        loss = self.kl_rate*kl_loss + emd_loss

        #Adding the cost funtion and metrics to the model
        self.add_loss(loss)
        self.add_metric(tf.reduce_sum(emd_loss), name="emd_loss", aggregation='mean')
        self.add_metric(tf.reduce_sum(kl_loss), name="kl_loss", aggregation='mean')

        #return the decoded data by default
        return decoded_shape

#Object in order to build a VAE, compile and plot the architecture
def build_vae(kl_rate, input_dim, latent_dim, verbose = True):
    vae = VAE(kl_rate, input_dim, latent_dim)
    vae.compile(optimizer="Adam")
    if verbose:
        vae.shape_encoder.summary()
        print('-'*50)
        vae.decoder.summary()
    return vae
