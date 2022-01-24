# -*- coding: utf-8 -*-
"""
@Project: psychic-memory
@File: test_reconstruction.py test_reconstruction

@Description: test and debugging of some Damped ART modified.

"""
import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil
import astra
from graddiv import (
    grad2, grad3
)
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr
from joint_reconstruction_solvers import (
    Kaczmarz,
    Damped_ART_TV_Segmentation,
    Standard_step
)
from sklearn.cluster import KMeans
from random import shuffle

from tomo_utils import (
   reduce_and_normalise_system,
   Shepp_Logan
)

__author__ = "François Lauze, University of Copenhagen"
__date__ = "1/21/22"
__version__ = "0.0.1"

def print_lsqr_info(r):
    keys = ('lstop', 'itn', 'r1norm', 'r2norm', 'anorm', 'acond', 'arnorm', 'xnorm', 'var')
    A = {}
    for i, key in enumerate(keys):
        A[key] = r[i]

    for key, val in A.items():
        print(f"{key} -> {A[key]}")


def create_projection_data_(n, dirs=90):
    """
    Parallel projection matrix and projection data from Sheep-Logan.

    Uses Astra-Tomo to generate the matrix. n controls the phantom resolution
    and dirs the amount of directions. For debugging.
    """
    x = Shepp_Logan(n)
    detectors = int(ceil(n * sqrt(2.0)))
    vol_geometry = astra.create_vol_geom(n, n)
    proj_geometry = astra.create_proj_geom(
        'parallel',
        1.0,
        detectors,
        np.linspace(0, np.pi, num=dirs, endpoint=False))
    proj_id = astra.create_projector('line', proj_geometry, vol_geometry)
    matrix_id = astra.projector.matrix(proj_id)
    A = astra.matrix.get(matrix_id)
    b = A @ x.ravel()
    b.shape = dirs, detectors
    return x, A, b


def shuffle_labels(labels, level=0.1):
    l = len(labels)
    n = int(ceil(l * level))
    indices = list(range(l))
    random.shuffle(indices)
    indices = indices[:n]
    q = labels[indices]
    random.shuffle(q)
    labels[indices] = q
    return labels


def initial_segmentation(x, k, noise_level=0.1):
    """
    Runs a k-means segmenter on x and returns the cartoon image.

    In the case of the clean Shepp-Logan image, this should be the
    original image! Thus, some noise in the form of a certain percent
    of randomly permuted labels is added.
    """
    kmeans = KMeans(n_clusters=k).fit(x.reshape((-1, 1)))
    labels = kmeans.labels_
    centres = kmeans.cluster_centers_
    shuffle_labels(labels, noise_level)
    labels.shape = x.shape
    Idk = np.eye(k)
    return np.squeeze(Idk[labels] @ centres)


class DisplayImage:
    def __init__(self, x, dims, x_gt=None):
        if x_gt is not None:
            self.fig, (ax_gt, self.ax) = plt.subplots(1, 2)
            ax_gt.imshow(x_gt)
        else:
            self.fig, self.ax = plt.subplots(1, 2)
        self.xshape = dims
        self.imdata = self.ax.imshow(np.reshape(x, self.xshape))

    # makes the object callable like a function.
    # dummy second variable cause some sweep_callbacks take two parameters
    def __call__(self, y, dummy=None):
        self.imdata.set_data(np.reshape(y, self.xshape))
        plt.pause(0.01)


class Objective:
    def __init__(self, A, b, alpha, beta, dims, d, roi=None):
        self.A = A
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.dims = dims
        self.roi = roi
        if self.roi is None:
            self.roi = 1.0
        self.d = d
        self.values = []
        self.D = grad2 if self.dims[0] == 2 else grad3

    # makes the object callable like a function.
    def __call__(self, x, k=None):
        val = 0.5*((self.A @ x - self.b)**2).sum()
        if self.beta > 0:
            val += 0.5*self.beta*((self.roi * x - self.d)**2).sum()
        if self.alpha > 0:
            x.shape = self.dims[1]
            val += np.linalg.norm(self.D(x), axis=-1).sum()
            x.shape = x.size
        self.values.append(val)
        if k is not None:
            print(f"At sweep {k}.")


def test_solver():
    resolution = 100
    directions = 180

    x, A, b = create_projection_data_(resolution, directions)
    b_noisy = b + np.random.randn(*b.shape) * 0.1
    d = initial_segmentation(x, 5, noise_level=0.2)

    u, v = np.mgrid[-1:1:100*1j, -1:1:100*1j]
    roi = u**2 + v**2 < 0.5

    An, bn = reduce_and_normalise_system(A, b_noisy.ravel())
    dims = (2, x.shape)
    alpha = 0.0002
    beta = 0.001
    x0 = np.random.rand(*x.shape) * x.max()
    x0.shape = x0.size
    d.shape = d.size
    roi.shape = roi.size

    dp = DisplayImage(x0, dims[1], x_gt=x)
    obj_func = Objective(An, bn, alpha, beta, dims, d)

    damped_art = Damped_ART_TV_Segmentation(An, bn, alpha, beta, d, dims, x0, roi=None)# roi)
    damped_art.add_start_callback(dp)
    damped_art.add_start_callback(obj_func)
    damped_art.add_sweep_callback(obj_func)
    damped_art.add_sweep_callback(dp)
    damped_art.add_end_callback(obj_func)
    damped_art.step_callback(Standard_step(t=1000.0))
    damped_art.max_iterations = 120
    damped_art.tolerance = 1e-4
    damped_art.solve()

    _, ax = plt.subplots(1,1)
    ax.plot(obj_func.values, label="objective function values")
    ax.grid('on')
    ax.set_xlabel("iterations")
    ax.set_label(r"$f(x_k)$")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    #plt.ion()
    test_solver()
    input("Press a key to terminate...")

















































