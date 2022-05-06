# -*- coding: utf-8 -*-
"""
@Project: psychic-memory
@File: segmentation_counts.py

@Description: Extract connected components of a binary image.
filter the components per size. Return list of components, with list of
corresponding pixel indices.

"""

import numpy as np
from skimage.measure import label
# demo!
from scalespace import gaussian_scalespace
import matplotlib.pyplot as plt

__author__ = "François Lauze, University of Copenhagen"
__date__ = "5/5/22"
__version__ = "0.0.1"


def get_segment_components(f: np.ndarray, k:int, size_threshold=0) -> list:
    """
    Extract connected components which size in pixel is larger than a threshold, from a segment.

    Args:
        f: int numpy array
            segmentation image size mxn, with value in {0...K-1}, K is number of segments.
        k: int
            integer in {0...K-1} specifying the segment number.
        size_threshold: int, optional
            threshold for minimal connected component size, the default is 0, i.e.,
            no threshold.
    Returns:
        list of connected components of size  at least size_size_threshold. Each entry contains
        - the connected component number,
        - the list of pixel indices for this component.
    """
    Rk = (f == k).astype(int)
    labels, n_components = label(Rk, connectivity=1, return_num=True)

    # runs over all the labels, keep the components larger than size_threshold
    # 0-value in labels means background.
    component_list = []
    for c in range(1, n + 1):
        lc = (labels == c).astype(int)
        if lc.sum() >= size_threshold:
            idx = np.where(lc == 1)
            # len(idx[0]) = len(idx[1]) = lc.sum()!
            component_list.append((c, idx))

    return component_list


def indices_to_image(idx: list, size: int) -> np.ndarray:
    """
    Make a binary image by setting to 1 pixels in the idx list.

    Args:
        idx: list
            idx = (array_x, array_y) each array being a 1D numpy array of
            coordinates.
        size: tuple
            size (m, n) of the image to create.
    Returns:
        a (m,n) image with 1 at idx pixels, 0 elsewhere.
    """
    u = np.zeros((m, n), dtype=int)
    u[idx] = 1
    return u


if __name__ == "__main__":
    # as in skimage example....
    m = n = 200
    st = 20
    points = np.random.randint(0, high=n, size=(n, 2))
    f = np.zeros((m, n))
    f[points[:, 0], points[:, 1]] = 1
    gf = gaussian_scalespace(f, sigma=10.0)
    blobs = (gf > 0.7 * gf.max()).astype(int)

    c_list = get_segment_components(blobs, k=1, size_threshold=st)
    print(f'found {len(c_list)} connected components of size >= {st}')
    if len(c_list) > 0:
        z = np.zeros((m, n), dtype=int)
        for level, idx in c_list:
            z += indices_to_image(idx, (m, n)) * level

        plt.imshow(z)
        plt.show()
    else:
        plt.imshow(blobs)
        plt.show()
