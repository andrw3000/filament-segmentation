from tensorflow import keras
from tensorflow.keras import layers


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


def get_autoencoder_model(image_size, num_colour_channels=1, num_classes=2):
    """Instantiate a Keras Autoencoder model."""

    # Input shape
    inputs = keras.Input(shape=image_size + (num_colour_channels,))

    # Encoding block
    x = encoding_conv(1, 128, 1)(inputs)
    x = layers.BatchNormalization()(x)

    x = encoding_conv(128, 64, 2)(x)
    x = layers.BatchNormalization()(x)

    x = encoding_conv(64, 32, 2)(x)
    x = layers.BatchNormalization()(x)

    x = encoding_conv(32, 8, 2)(x)
    x = layers.BatchNormalization()(x)

    # Decoding block
    x = decoding_conv(8, 8, 2)(x)
    x = layers.BatchNormalization()(x)

    x = decoding_conv(8, 16, 2)(x)
    x = layers.BatchNormalization()(x)

    x = decoding_conv(16, 32, 2)(x)
    x = layers.BatchNormalization()(x)

    x = decoding_conv(32, 64, 1)(x)
    x = layers.BatchNormalization()(x)

    x = decoding_conv(64, 128, 1)(x)
    x = layers.BatchNormalization()(x)

    # Add a per-pixel classification layer
    if num_classes == 1:
        final_activation = "sigmoid"
    else:
        final_activation = "softmax"

    x = decoding_conv(128, num_classes, 1, activation=final_activation)(x)
    # outputs = layers.BatchNormalization()(x)
    outputs = x

    # Define the model
    model = keras.Model(inputs, outputs)
    return model
