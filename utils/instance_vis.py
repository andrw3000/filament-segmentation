# Instance segmentation from semantic masks

import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import colors


def label_colours(ncolours, cmap_name: str = 'tab20'):
    """Generates a list of length ncolours of RGB values over the cmap.

    Colour maps reference:
    https://matplotlib.org/stable/gallery/color/colormap_reference.html
    """
    cmap = plt.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=0., vmax=1.)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    indices = np.linspace(0., 1., ncolours)
    return color.rgba2rgb(scalar_map.to_rgba(indices))


def shade_instances(
        name, image, instances, line_ends=None, cmap_name: str = 'tab20',
):
    """Plots image (greyscale) with instance masks overlayed in colour."""

    # Sum instance masks
    instance_labels = np.zeros((image.shape[0], image.shape[1]), dtype=float)
    for n, instance in enumerate(instances):
        instance_labels[instance > 0] = float(n+1)

    coloured_imag = color.label2rgb(instance_labels,
                                    image=image,
                                    colors=label_colours(len(instances),
                                                         cmap_name=cmap_name,
                                                         ),
                                    alpha=0.4,
                                    kind='overlay',
                                    saturation=0,
                                    )

    # Plot mask and image overlay instances
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(coloured_imag)
    ax.set_title('Coloured image {}'.format(name))
    if line_ends:
        for end_pairs in line_ends:
            for ends in end_pairs:
                e1 = ends[0]
                e2 = ends[1]
                ax.plot([e1[1], e2[1]], [e1[0], e2[0]], ls="", marker="o")

    return fig


def grid_display_masks(name, semantic_mask, instances):
    """Plots semantic mast and corresponding seperate instance masks."""

    # Plot all instances
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # Plot original mask
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
    ax0 = fig.add_subplot(gs0[0])
    ax0.imshow(semantic_mask, cmap=cm.gray)
    ax0.set_title('Input mask')

    # Plot proposed instances
    nlines = len(instances)
    grid_width = nlines // 4 + int(np.ceil((nlines % 4) / 4))
    gs1 = gridspec.GridSpecFromSubplotSpec(4, grid_width, subplot_spec=gs[1])
    for ii in range(nlines):
        ax1 = fig.add_subplot(gs1[ii])
        ax1.imshow(instances[ii], cmap=cm.gray)
        ax1.set_ylim((semantic_mask.shape[0], 0))
        ax1.set_xlim((0, semantic_mask.shape[1]))
        ax1.set_axis_off()

    return fig
