import tensorflow as tf
from tensorflow import keras


def encoding_conv(in_feats, out_feats, stride):
    """Define bespoke encoding convolution."""
    return keras.layers.Conv2D(filters=out_feats,
                               kernel_size=(5, 5),
                               input_shape=(None, None, in_feats),
                               strides=(stride, stride),
                               padding='same',
                               data_format='channels_last',
                               activation='relu',
                               )


def decoding_conv(in_feats, out_feats, stride, activation='relu'):
    """Define bespoke decoding convolution."""
    return keras.layers.Conv2DTranspose(filters=out_feats,
                                        kernel_size=(5, 5),
                                        input_shape=(None, None, in_feats),
                                        strides=(stride, stride),
                                        padding='same',
                                        data_format='channels_last',
                                        activation=activation,
                                        )


class Autoencoder(keras.Model):
    """Defines the encoding-decoding paradigm."""

    def __init__(self, autoencoder_lr: float = 0.001):
        super(Autoencoder, self).__init__()
        self.encoder_tup = (encoding_conv(1, 128, 1),
                            keras.layers.BatchNormalization(),
                            encoding_conv(128, 64, 2),
                            keras.layers.BatchNormalization(),
                            encoding_conv(64, 32, 2),
                            keras.layers.BatchNormalization(),
                            encoding_conv(32, 8, 2),
                            keras.layers.BatchNormalization(),
                            )
        self.decoder_tup = (decoding_conv(8, 8, 2),
                            keras.layers.BatchNormalization(),
                            decoding_conv(8, 16, 2),
                            keras.layers.BatchNormalization(),
                            decoding_conv(16, 32, 2),
                            keras.layers.BatchNormalization(),
                            decoding_conv(32, 64, 1),
                            keras.layers.BatchNormalization(),
                            decoding_conv(64, 128, 1),
                            keras.layers.BatchNormalization(),
                            decoding_conv(128, 1, 1, activation='sigmoid'),
                            keras.layers.BatchNormalization(),
                            )
        self.lr = keras.optimizers.schedules.ExponentialDecay(
            autoencoder_lr, decay_steps=1000, decay_rate=0.75, staircase=True
        )
        self.optimiser = keras.optimizers.Adam(learning_rate=self.lr)

    def call_encoder(self, input_state, training=False):
        state = input_state
        for i in range(len(self.encoder_tup)):
            # if i % 2 == 0: print('in '+str(i)+' enc_state: ', state.shape)
            state = self.encoder_tup[i](state, training=training)
            # if i % 2 == 0: print('out '+str(i)+' enc_state: ', state.shape)
        return state

    def call_decoder(self, input_state, training=False):
        state = input_state
        for i in range(len(self.decoder_tup)):
            # if i % 2 == 0: print('in '+str(i)+' dec_state: ', state.shape)
            state = self.decoder_tup[i](state, training=training)
            # if i % 2 == 0: print('out '+str(i)+' dec_state: ', state.shape)
        return state

    def call(self, input_state, training=False):
        encoded = self.call_encoder(input_state, training=training)
        decoded = self.call_decoder(encoded, training=training)
        return decoded

    def optimise_autoencoder(self, x, y, loss_func):
        with tf.GradientTape() as g:
            enc = self.call_encoder(x, training=True)
            x_predict = self.call_decoder(enc, training=True)
            loss = tf.math.reduce_mean(loss_func(y, x_predict))
        self.optimiser.minimize(
            loss, [layer.trainable_weights for layer in
                   self.encoder_tup + self.decoder_tup], tape=g
        )
        return loss


@tf.function
def ae_train_step(data, model):
    images, masks = data
    return model.optimise_autoencoder(images, masks, keras.losses.MSE)


@tf.function
def ae_test_step(data, model):
    images, masks = data
    evals = model(images)
    loss = tf.math.reduce_mean(keras.losses.MSE(evals, masks), axis=1)
    loss = tf.math.reduce_mean(loss, axis=1)
    # print ('evals.shape: ', evals.shape)
    # print ('images.shape: ', images.shape)
    # print ('loss: ', loss)
    return evals, loss, images, masks
