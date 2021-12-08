import numpy as np
from skimage.transform import hough_line
from skimage.transform import hough_line_peaks
from skimage.draw import line

def mask2lines(mask: np.ndarray):
    """Traces straight lines through semantic filament segmentations."""

    # Classic straight-line Hough transform with .5 degree precision
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(mask, theta=tested_angles)

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]

    # rr, cc = line(1, 1, 8, 8)

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))