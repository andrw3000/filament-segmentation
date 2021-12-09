import os
import numpy as np

from skimage import io
from skimage.transform import hough_line
from skimage.transform import hough_line_peaks
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny

import matplotlib.pyplot as plt
from matplotlib import cm


# Import test image
root_dir = '/Users/Holmes/Research/IDSAI/PROOF/filament-segmentation'
data_dir = os.path.join(root_dir, 'data/masks-tf1/semantic')
image_file = 'tf1_001.png'
image = io.imread(os.path.join(data_dir, image_file))
if len(image.shape) > 2:
    image = image[:, :, 0]
print('image shape: ', image.shape)
print('image max: ', image.max())
print('image min: ', image.min())

# Classic straight-line Hough transform
# Set a precision of .5 degrees.
edges = canny(image, 2, 1, 25)
print('edges shape: ', edges.shape)

tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(image, theta=tested_angles)

print('h.shape: ', h.shape)
print('theta.shape: ', theta.shape)
print('d.shape: ', d.shape)

# Generating figure 1
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(image, cmap=cm.gray)
ax[1].set_ylim((image.shape[0], 0))
ax[1].set_xlim((0, image.shape[1]))
ax[1].set_title('Detected lines')

for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=50)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax[1].axline((x0, y0), slope=np.tan(angle + np.pi/2))

plt.tight_layout()
plt.show()
