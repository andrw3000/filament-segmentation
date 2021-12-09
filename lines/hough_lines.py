import numpy as np
from skimage.transform import hough_line
from skimage.transform import hough_line_peaks
from skimage.draw import line


def boundary_intersections(angle, dist):
    """Converts Hough angle + distance to (x, y) axis instercepts.

    Returns: [tuple] (y intercept, x intercept)
    """
    x0 = np.rint(dist / np.cos(angle))
    y0 = np.rint(dist / np.sin(angle))
    return (0, y0), (x0, 0)


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


def mask2lines(mask: np.ndarray):
    """Traces straight lines through semantic filament segmentations."""

    # Classic straight-line Hough transform with .5 degree precision
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, angles, dists = hough_line(mask, theta=tested_angles)
    xint = []
    yint = []
    for _, angle, dist in zip(*hough_line_peaks(
            h, angles, dists, min_distance=100, min_angle=10
    )):
        xaxis, yaxis = boundary_intersections(angle, dist)
        xint.append(xaxis)
        yint.append(yaxis)


    # rr, cc = line(1, 1, 8, 8)

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))