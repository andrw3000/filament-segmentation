# Instance segmentation from semantic masks

import os
import numpy as np
from skimage import io

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

from lines import get_line_instances

# Import test mask
root_dir = '/Users/Holmes/Research/IDSAI/PROOF/filament-segmentation'
data_dir = os.path.join(root_dir, 'data/masks-tf1/semantic')
mask_file = 'tf1_002.png'
mask = io.imread(os.path.join(data_dir, mask_file))
if len(mask.shape) > 2:
    mask = mask[:, :, 0]
print('mask shape: ', mask.shape)


# Compute instances from semantic mask
instances, line_ends = get_line_instances(mask,
                                          length_tol=0.2,
                                          line_pixel_width=160,
                                          hough_line_sep=40,
                                          )

nlines = len(instances)
print('Number of lines identified: ', nlines)

# Check line ends
for idx, line_end in enumerate(line_ends):
    print('Line endings on line {}: '.format(idx), *line_end)

# Plot output of `line_instances`

# Plot all instances
fig = plt.figure(constrained_layout=True, figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, figure=fig)

# Plot original mask
gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
ax0 = fig.add_subplot(gs0[0])
ax0.imshow(mask, cmap=cm.gray)
ax0.set_title('Input mask')

# Plot proposed instances
grid_width = nlines//4 + int(np.ceil((nlines % 4) / 4))
gs1 = gridspec.GridSpecFromSubplotSpec(4, grid_width, subplot_spec=gs[1])
for ii in range(nlines):
    ax1 = fig.add_subplot(gs1[ii])
    ax1.imshow(instances[ii], cmap=cm.gray)
    ax1.set_ylim((mask.shape[0], 0))
    ax1.set_xlim((0, mask.shape[1]))
    ax1.set_axis_off()

#plt.tight_layout()
plt.show()
