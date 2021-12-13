import numpy as np
import skimage
import matplotlib.pyplot as plt


def get_line_ends(
    mask: np.ndarray, plot: bool = False, dpi: int = 500
) -> np.ndarray:
    # how to determine if we have a line end after skeletonization:
    # - look in a 3x3 window centred on a pixel
    # - if we only have one pixel connected to it then it must be a line end

    # copy the mask and remove border pixels
    mask = np.copy(mask)
    mask[0, :] = mask[:, 0] = mask[-1, :] = mask[:, -1] = False

    rr, cc = np.where(mask)  # rows, cols indices of the marked pixels
    pixel_inds = np.stack([rr, cc]).T

    # offsets of the neighbours of a central pixel (i.e. [0, 0])
    neighbour_inds = np.array(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    )

    # get all the neighbours of the pixels -- shape (npixels, 8, 2)
    pixel_neighbour_inds = (
        pixel_inds[:, np.newaxis, :] + neighbour_inds[np.newaxis, :, :]
    )

    # extract the pixels from the mask -- shape (npixels, 8)
    pixel_neighbours = mask[
        pixel_neighbour_inds[:, :, 0], pixel_neighbour_inds[:, :, 1]
    ]

    # count the number of neighbours -- shape (npixels, )
    pixel_neighbour_counts = np.count_nonzero(pixel_neighbours, axis=1)

    # pixels at the end of a line will have only 1 neighbour
    end_pixels = pixel_inds[pixel_neighbour_counts == 1, :]

    # pair up the end points.

    # if we've only got 2 end points, job done
    if end_pixels.shape[0] == 2:
        end_pairs = np.array([[end_pixels[0], end_pixels[1]]])

    # else we label each pixel in the marked up mask based on its
    # connected components -- so lines will all be the same label
    # we can exploit the fact that, because we've already found the end
    # coordinates, we can just index the label mask for each end point
    # and get their corresponding labels.
    else:
        # labels go from [1, nlabels+1] inclusive (0 is background)
        mask_labels, nlabels = skimage.measure.label(  # type: ignore
            mask, return_num=True, connectivity=2
        )

        # build up a list of end pixel pairs such that the i'th pair of points
        # is end_pairs[i] = [[r0, c0], [r1, c1]] -- [row, col]
        end_pairs = np.zeros((nlabels, 2, 2))

        # pointer to know if we've already seen a point in that
        # connected component or yet (for indexing correctly!)
        e = np.zeros(nlabels, dtype="int")

        for r, c in end_pixels:
            # get the index of the pair to select (-1 because starting at 1)
            idx = mask_labels[r, c] - 1

            # store it
            end_pairs[idx, e[idx], :] = r, c

            # increment pointer
            e[idx] += 1

        # set the plotting image
        mask = mask_labels

    if plot:
        _, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=dpi)
        ax.imshow(mask, cmap="hot")
        for [[r0, c0], [r1, c1]] in end_pairs:
            ax.plot([c0, c1], [r0, r1], ls="", marker="x")
        plt.show()

    return end_pairs
