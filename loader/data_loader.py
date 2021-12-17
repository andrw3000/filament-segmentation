# Locate and augment data
import os
import glob
import numpy as np
from skimage import transform, io

def get_data(path_train_imgs,
             path_train_msks,
             path_valid_imgs,
             path_valid_msks,
             train_frac,
             valid_frac,
             image_size,
             num_images_per_original,
             ):
    """Get, split, and arrange the dataset."""

    # Get training data
    data_tra_imgs_filenames = glob.glob(path_train_imgs)
    data_tra_imgs = []
    data_tra_msks = []
    for img_file in data_tra_imgs_filenames:
        msk_file = os.path.join(path_train_msks,
                                img_file.split('/')[-1][:-4] + '.png')
        img = io.imread(img_file).astype(float) / 255.
        img = transform.resize(img, (2 * image_size[0], 2 * image_size[1]))
        msk = (io.imread(msk_file) > 0).astype(float)
        msk = transform.resize(msk, (img.shape[0], img.shape[1]))
        msk = (msk > 0.5).astype(float)
        #print('msk.shape: ', msk.shape)
        #print('img.shape: ', img.shape)

        for i in range(num_images_per_original):
            x_rand = np.random.randint(0, img.shape[1] - image_size[1])
            y_rand = np.random.randint(0, img.shape[0] - image_size[0])
            img_cropped = img[y_rand:y_rand + image_size[0],
                              x_rand:x_rand + image_size[1]]
            msk_cropped = msk[y_rand:y_rand + image_size[0],
                              x_rand:x_rand + image_size[1]]
            data_tra_imgs.append(np.expand_dims(img_cropped, -1))
            data_tra_msks.append(np.expand_dims(msk_cropped, -1))

    # Get validation/testing data
    if len(path_valid_imgs):
        data_validation_filenames = glob.glob(path_valid_imgs)
        data_val_imgs = []
        data_val_msks = []
        for img_file in data_validation_filenames:
            msk_file = os.path.join(path_valid_msks,
                                    img_file.split('/')[-1][:-4] + '.png')
            img = io.imread(img_file).astype(float) / 255.
            img = transform.resize(img, (2 * image_size[0], 2 * image_size[1]))
            img = img[:image_size[0], :image_size[1]]
            msk = (io.imread(msk_file) > 0).astype(float)
            msk = msk[:image_size[0], :image_size[1]]
            data_val_imgs.append(np.expand_dims(img, -1))
            data_val_msks.append(np.expand_dims(msk, -1))
        # Split the validation data into two
        ndata_val = len(data_val_imgs)
        data_tes_imgs = data_val_imgs[int(ndata_val / 2):]
        data_tes_msks = data_val_msks[int(ndata_val / 2):]
        data_val_imgs = data_val_imgs[:int(ndata_val / 2)]
        data_val_msks = data_val_msks[:int(ndata_val / 2)]

    else:
        ndata_tot = len(data_tra_imgs)
        data_val_imgs = data_tra_imgs[int(train_frac * ndata_tot):
                                      int((train_frac + valid_frac) * ndata_tot)]
        data_val_msks = data_tra_msks[int(train_frac * ndata_tot):
                                      int((train_frac + valid_frac) * ndata_tot)]
        data_tes_imgs = data_tra_imgs[int((train_frac + valid_frac) * ndata_tot):]
        data_tes_msks = data_tra_msks[int((train_frac + valid_frac) * ndata_tot):]
        data_tra_imgs = data_tra_imgs[:int(train_frac * ndata_tot)]
        data_tra_msks = data_tra_msks[:int(train_frac * ndata_tot)]

    return (data_tra_imgs,
            data_tra_msks,
            data_val_imgs,
            data_val_msks,
            data_tes_imgs,
            data_tes_msks,
            )
