import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class UNet(keras.Model):
    """Defines the U-Net paradigm."""

    def __init__(self, unet_lr: float = 0.001):
        super(UNet, self).__init__()
        self.lr = keras.optimizers.schedules.ExponentialDecay(
            unet_lr, decay_steps=1000, decay_rate=0.75, staircase=True
        )

        self.optimiser = keras.optimizers.Adam(learning_rate=self.lr)

    def call(self, x, training=False):

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Second half of the network: upsampling inputs
        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        x = layers.Conv2D(
            1, 3, activation="softmax", padding="same"
        )(x)

        return x

    def optimise_unet(self, x, y, loss_func):
        with tf.GradientTape() as g:
            y_pred = self.__call__(x, training=True)
            loss = tf.math.reduce_mean(loss_func(y, y_pred))
        self.optimiser.minimize(
            loss, var_list=[self.trainable_weights], tape=g
        )
        return loss


@tf.function
def unet_train_step(data, model):
    images, masks = data
    return model.optimise_unet(images, masks, keras.losses.MSE)


@tf.function
def unet_test_step(data, model):
    images, masks = data
    evals = model(images)
    loss = tf.math.reduce_mean(keras.losses.MSE(evals, masks), axis=1)
    loss = tf.math.reduce_mean(loss, axis=1)
    # print ('evals.shape: ', evals.shape)
    # print ('images.shape: ', images.shape)
    # print ('loss: ', loss)
    return evals, loss, images, masks
