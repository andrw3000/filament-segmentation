# Instance segmentation from semantic masks

import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import colors
from lines import get_line_instances


class MaskColours:
    """Process colour map to make RGB values available."""

    def __init__(self, cmap_name='winter', start_val=0., stop_val=1.):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

# Import test mask
root_dir = '/Users/Holmes/Research/IDSAI/PROOF/filament-segmentation'
data_dir = os.path.join(root_dir, 'data/masks-tf1/semantic')
mask_file = 'tf1_002.png'
mask = io.imread(os.path.join(data_dir, mask_file))
print(mask.shape)
print(mask.shape[0])
print(mask.shape[1])
print(mask.shape[2])
if len(mask.shape) > 2:
    mask = mask[:, :, 0]
print('mask shape: ', mask.shape)


# Compute instances from semantic mask
instances, line_ends = get_line_instances(mask,
                                          length_tol=0.2,
                                          line_pixel_width=160,
                                          hough_line_sep=40,
                                          )

ninst = len(instances)
print('Number of lines identified: ', ninst)

# Check line ends
for idx, line_end in enumerate(line_ends):
    print('Line endings on line {}: '.format(idx), *line_end)

#################################
# Plot output of `line_instances`
#################################

# Plot all instances
fig = plt.figure(constrained_layout=True, figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, figure=fig)

# Plot original mask
gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
ax0 = fig.add_subplot(gs0[0])
ax0.imshow(mask, cmap=cm.gray)
ax0.set_title('Input mask')

# Plot proposed instances
grid_width = ninst//4 + int(np.ceil((ninst % 4) / 4))
gs1 = gridspec.GridSpecFromSubplotSpec(4, grid_width, subplot_spec=gs[1])
for ii in range(ninst):
    ax1 = fig.add_subplot(gs1[ii])
    ax1.imshow(instances[ii], cmap=cm.gray)
    ax1.set_ylim((mask.shape[0], 0))
    ax1.set_xlim((0, mask.shape[1]))
    ax1.set_axis_off()

#plt.tight_layout()
plt.show()

#################################
# Plot overlay of instances
#################################

#################################
# Plot overlay of instances
#################################

lines = ['Readme', 'How to write text files in Python']
with open(file_name + 'coordinates.txt', 'w') as f:
    f.write('\n'.join(lines))