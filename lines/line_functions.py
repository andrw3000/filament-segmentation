import numpy as np
from skimage.transform import hough_line
from skimage.transform import hough_line_peaks
from skimage.draw import line
from .get_line_endings import get_line_ends


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


def get_line_instances(semantic_mask: np.ndarray,
                       hough_line_dist: int = 100,
                       pixel_width: int = 1):
    """Traces straight lines through semantic filament segmentations."""

    # Classic straight-line Hough transform with .5 degree precision
    test_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, angles, dists = hough_line(semantic_mask, theta=test_angles)

    # Function outputs
    full_lines = []
    instances = []
    line_ends = []

    # Iterate over Hough lines
    for _, angle, dist in zip(*hough_line_peaks(
            h, angles, dists, min_distance=hough_line_dist, min_angle=10
    )):
        # Intercepts of Hough line with pixel border
        int1, int2 = boundary_intersections(nrows=semantic_mask.shape[0],
                                            ncols=semantic_mask.shape[1],
                                            angle=angle,
                                            dist=dist,
                                            )

        # Compute line
        rr, cc, vals = weighted_line(
            int1[0], int1[1], int2[0], int2[1], pixel_width
        )

        # Trim weighted line
        rr[rr >= semantic_mask.shape[0]] = semantic_mask.shape[0] - 1
        cc[cc >= semantic_mask.shape[1]] = semantic_mask.shape[1] - 1

        # Record total line
        full_line = np.zeros(shape=semantic_mask.shape, dtype=np.uint8)
        full_line[rr, cc] = 255
        full_lines.append(full_line)

        # Take complement with semantic mask
        float_line = full_line.astype(float)
        float_mask =
        instance = (full_line.astype(float) + semantic_mask > 255).astype(int) * 255
        instances.append(instance)

        # Compute thin line:
        rr0, cc0 = line(int1[0], int1[1], int2[0], int2[1])
        rr0[rr0 >= semantic_mask.shape[0]] = semantic_mask.shape[0] - 1
        cc0[cc0 >= semantic_mask.shape[1]] = semantic_mask.shape[1] - 1

        # Take compliment with thin line
        thin_line = np.zeros(shape=semantic_mask.shape, dtype=np.uint8)
        thin_line[rr0, cc0] = 255
        line_ends.append(get_line_ends(thin_line, dpi=72))

    return full_lines, instances, line_ends
