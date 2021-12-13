# Instance segmentation from semantic masks

import os
import numpy as np
from skimage import io

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

from lines import get_line_instances

# Import test image
root_dir = '/Users/Holmes/Research/IDSAI/PROOF/filament-segmentation'
data_dir = os.path.join(root_dir, 'data/masks-tf1/semantic')
image_file = 'tf1_090.png'
image = io.imread(os.path.join(data_dir, image_file))
if len(image.shape) > 2:
    image = image[:, :, 0]
print('image shape: ', image.shape)


# Compute instances from semantic mask
full_lines, instances, line_ends = get_line_instances(image,
                                                      hough_line_dist=100,
                                                      pixel_width=200)

nlines = len(instances)
print('Number of lines identified: ', nlines)

# Check line ends
for idx, line_end in enumerate(line_ends):
    print('Line endings on line {}: '.format(idx), *line_end)

# Plot output of `line_instances`

# Plot all instances
fig = plt.figure(constrained_layout=True, figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, figure=fig)

# Plot original image
gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
ax0 = fig.add_subplot(gs0[0])
ax0.imshow(image, cmap=cm.gray)
ax0.set_title('Input image')

# Plot proposed instances
grid_width = nlines//4 + int(np.ceil((nlines % 4) / 4))
gs1 = gridspec.GridSpecFromSubplotSpec(4, grid_width, subplot_spec=gs[1])
for ii in range(nlines):
    ax1 = fig.add_subplot(gs1[ii])
    ax1.imshow(instances[ii], cmap=cm.gray)
    ax1.set_ylim((image.shape[0], 0))
    ax1.set_xlim((0, image.shape[1]))
    ax1.set_axis_off()

#plt.tight_layout()
plt.show()
