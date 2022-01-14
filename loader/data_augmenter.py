import math
import tensorflow as tf
import tensorflow.keras.backend as kbe


def get_trans_mat(rotation,
                  shear,
                  height_zoom,
                  width_zoom,
                  height_shift,
                  width_shift,
                  ):
    """Returns a 3x3 transform matrix to transforms indicies.

    See: https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96
    """

    # Degrees to radians
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    # Rotation matrix
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    rotation_matrix = tf.reshape(
        tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0),
        [3, 3],
    )

    # Shear matrix
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(
        tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0),
        [3, 3],
    )

    # Zoom matrix
    zoom_matrix = tf.reshape(
        tf.concat([one / height_zoom, zero, zero, zero,
                   one / width_zoom, zero, zero, zero, one,
                   ], axis=0),
        [3, 3],
    )

    # Shift matrix
    shift_matrix = tf.reshape(
        tf.concat([one, zero, height_shift, zero,
                   one, width_shift, zero, zero, one,
                   ], axis=0),
        [3, 3],
    )

    return kbe.dot(kbe.dot(rotation_matrix, shear_matrix),
                   kbe.dot(zoom_matrix, shift_matrix),
                   )


def random_transform(image, mask):
    """Randomly transforms image and corresponding mask. """
    if image.shape[0] != image.shape[1]:
        raise ValueError('Square images only for transforming')

    dim = image.shape[0]
    xdim = dim % 2  # Adjustment for size 331

    # Get transformation matrix
    mat = get_trans_mat(
        rotation=15. * tf.random.normal([1], dtype='float32'),
        shear=5. * tf.random.normal([1], dtype='float32'),
        height_zoom=1.0 + tf.random.normal([1], dtype='float32') / 10.,
        width_zoom=1.0 + tf.random.normal([1], dtype='float32') / 10.,
        height_shift=16. * tf.random.normal([1], dtype='float32'),
        width_shift=16. * tf.random.normal([1], dtype='float32'),
    )

    # List destination pixel indices
    x = tf.repeat(tf.range(dim // 2, -dim // 2, -1), dim)
    y = tf.tile(tf.range(-dim // 2, dim // 2), [dim])
    z = tf.ones([dim * dim], dtype='int32')
    idx = tf.stack([x, y, z])

    # Rotate destination pixels onto origin pixels
    idx2 = kbe.dot(mat, tf.cast(idx, dtype='float32'))
    idx2 = kbe.cast(idx2, dtype='int32')
    idx2 = kbe.clip(idx2, -dim // 2 + xdim + 1, dim // 2)

    # Find original pixel values
    idx3 = tf.stack([dim // 2 - idx2[0], dim // 2 - 1 + idx2[1]])
    new_image = tf.gather_nd(image, tf.transpose(idx3))
    new_mask = tf.gather_nd(mask, tf.transpose(idx3))

    return (tf.reshape(new_image, [dim, dim, 1]),
            tf.reshape(new_mask, [dim, dim, 1]),
            )


def one_hot_mask(image, mask):
    """Converts integer valued binary mask to a one-hot encoded mask."""
    assert mask.shape[-1] == 1, "Last mask channel should be length 1."
    ohmask = tf.stack((tf.squeeze(mask), 1. - tf.squeeze(mask)), axis=-1)
    return image, ohmask


def augment_data(train_imgs,
                 train_msks,
                 valid_imgs,
                 valid_msks,
                 batch_size,
                 one_hot=False,
                 ):
    train_set = tf.data.Dataset.from_tensor_slices((train_imgs, train_msks))
    train_set = train_set.shuffle(512)
    train_set = train_set.map(random_transform)
    train_set = train_set.batch(batch_size)
    valid_set = tf.data.Dataset.from_tensor_slices((valid_imgs, valid_msks))
    valid_set = valid_set.shuffle(512).batch(batch_size)
    if one_hot:
        train_set = train_set.map(one_hot_mask)
        valid_set = valid_set.map(one_hot_mask)
    return train_set, valid_set

# def transform_data(images, masks):
#    random_angles = tf.random.uniform(
#        shape=(), minval=0, maxval=4, dtype=tf.int32,
#    )
#    return (tf.image.rot90(images, random_angles),
#            tf.image.rot90(masks, random_angles),
#            )
