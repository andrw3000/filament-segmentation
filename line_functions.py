import numpy as np
import skimage
from skimage.transform import hough_line
from skimage.transform import hough_line_peaks
from skimage.draw import line


def boundary_intersections(nrows, ncols, angle, dist):
    """Converts Hough angle + distance to image boundary instercepts.

    Args:
        nrows: number of rows in image; image.shape[0]
        ncols: number of rows in image; image.shape[1]
        angle: measured from the x-axis anti-clockwise valued in [-pi/2, pi/2]
        dist: min distance between the line and the origin (0,0) in top-left

    Returns: list [(r1, c1), (r2, c2))] where Hough line crosses the boundary.
    """
    ymin, ymax = 0, nrows - 1
    xmin, xmax = 0, ncols - 1
    x0 = dist * np.cos(angle)
    y0 = dist * np.sin(angle)
    grad = np.tan(angle + np.pi/2)  # Assuming theta anti-clockwise from x-axis

    # Compute boundary intersections with  exceptional `tol` cases.
    bpoints = []
    tol = 1e-8

    if grad < tol:
        # Approximately vertical, y = y0 with grad = 0.
        bpoints.append((np.rint(y0).astype(np.int), xmin))
        bpoints.append((np.rint(y0).astype(np.int), xmax))

    elif grad > 1/tol:
        # Approximately horizontal, x = x0 with grad = inf.
        bpoints.append((ymin, np.rint(x0).astype(np.int)))
        bpoints.append((ymax, np.rint(y0).astype(np.int)))

    else:
        # Line y = grad * x + yint
        yint = y0 - grad * x0

        # Intersections
        with_ymin = -yint/grad
        with_ymax = (ymax - yint) / grad
        with_xmin = yint
        with_xmax = grad * xmax + yint

        # Check the intersections hit the pixel image boundary
        if with_ymin < xmin or with_ymin > xmax:
            with_ymin = None
        else:
            with_ymin = (ymin, np.rint(with_ymin).astype(np.int))
        if with_ymax < xmin or with_ymax > xmax:
            with_ymax = None
        else:
            with_ymax = (ymax, np.rint(with_ymax).astype(np.int))
        if with_xmin < ymin or with_xmin > ymax:
            with_xmin = None
        else:
            with_xmin = (np.rint(with_xmin).astype(np.int), xmin)
        if with_xmax < ymin or with_xmax > ymax:
            with_xmax = None
        else:
            with_xmax = (np.rint(with_xmax).astype(np.int), xmax)

        for intercept in (with_ymin, with_ymax, with_xmin, with_xmax):
            if intercept is not None:
                bpoints.append(intercept)

    if not len(bpoints) == 2:
        raise ValueError('Only {} boundary intersections'.format(len(bpoints)))

    return bpoints[0], bpoints[1]


def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return yy, xx, val

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1, 1)
          + np.arange(-thickness-1, thickness+2).reshape(1, -1))
    xx = np.repeat(x, yy.shape[1])

    # Compute trapez
    y0 = y.reshape(-1, 1)
    trapez = np.clip(
        np.minimum(yy + 1 + w / 2 - y0, -yy + 1 + w / 2 + y0), 0, 1
    )
    vals = trapez.flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return yy[mask].astype(int), xx[mask].astype(int), vals[mask]


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


def get_line_instances(semantic_mask: np.ndarray,
                       length_tol: float = 0.2,
                       line_pixel_width: int = 1,
                       hough_line_sep: int = 20,
                       ):
    """Traces straight lines through semantic filament segmentations.

    Args:
        semantic_mask: Input mask with data type uint8.
        length_tol: Fraction of longest edge for minimum length.
        line_pixel_width: Bespoke maximum pixel width of instance masks.
        hough_line_sep: Minimum distance between proposed hough lines.

    Returns:
        full_lines: Lines spanning the boundaries os the semantic_mask.
        instances: Proposed instance masks.
        line_ends: corresponding endpoints of instance masks.
    """

    # Classic straight-line Hough transform with .5 degree precision
    test_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, angles, dists = hough_line(semantic_mask, theta=test_angles)

    # Function outputs
    instances = []
    line_ends = []

    # Iterate over Hough lines
    for _, angle, dist in zip(*hough_line_peaks(
            h, angles, dists, min_distance=hough_line_sep, min_angle=10,
    )):
        # Intercepts of Hough line with pixel border
        int1, int2 = boundary_intersections(nrows=semantic_mask.shape[0],
                                            ncols=semantic_mask.shape[1],
                                            angle=angle,
                                            dist=dist,
                                            )

        # Compute thin line:
        thin_line = np.zeros(shape=semantic_mask.shape, dtype=np.uint8)
        rr, cc = line(int1[0], int1[1], int2[0], int2[1])
        rr[rr >= semantic_mask.shape[0]] = semantic_mask.shape[0] - 1  # Trim r
        cc[cc >= semantic_mask.shape[1]] = semantic_mask.shape[1] - 1  # Trim c
        thin_line[rr, cc] = 255

        # Take intersection of thin line and semantic mask
        thin_comp = (thin_line.astype(float) +
                     semantic_mask.astype(float) > 255).astype(int) * 255
        end_pairs = get_line_ends(thin_comp, dpi=72)

        # Discard small fragments from instance
        sq_dists = []
        long_ends = []
        long_lines = []
        for ends in end_pairs:
            e1 = ends[0]
            e2 = ends[1]
            sq_dists.append((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2)
            if np.sqrt(sq_dists[-1]) > length_tol * max(semantic_mask.shape):
                long_ends.append(ends)
                wide_line = np.zeros(shape=semantic_mask.shape, dtype=np.uint8)
                rrw, ccw, _ = weighted_line(
                    e1[0], e1[1], e2[0], e2[1], line_pixel_width,
                )
                rrw[rrw >= semantic_mask.shape[0]] = semantic_mask.shape[0] - 1
                ccw[ccw >= semantic_mask.shape[1]] = semantic_mask.shape[1] - 1
                wide_line[rrw, ccw] = 255
                long_lines.append(wide_line.astype(float))

        if long_lines:
            # Save endpoints
            line_ends.append(long_ends)

            # Add fragments together onto single instance mask
            instance = (sum(long_lines) > 0).astype(int) * 255

            # Take intersection with semantic mask
            instances.append(
                (instance.astype(float) +
                 semantic_mask.astype(float) > 255).astype(int) * 255
            )

    else:
        pass

    return instances, line_ends
