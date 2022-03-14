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
    Standard_step,
    Segmentation_proximal,
    Special_Step
)
from fistaTV import Display_Field_Regularisation


from sklearn.cluster import KMeans
from random import shuffle

from tomo_utils import (
   reduce_and_normalise_system,
   Shepp_Logan
)

__author__ = "François Lauze, University of Copenhagen"
__date__ = "1/21/22"
__version__ = "0.0.1"


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
    v = Idk[labels]
    return np.squeeze(v @ centres), v, centres


class DisplayImage:
    def __init__(self, x, dims, x_gt=None, x_seg=None):
        self.xshape = dims
        if (x_gt is not None) and (x_seg is not None):
            self.fig, (ax_gt, ax_seg, self.ax) = plt.subplots(1, 3)
            ax_gt.imshow(np.reshape(x_gt, self.xshape))
            ax_gt.set_title("Ground truth")
            ax_seg.imshow(np.reshape(x_seg, self.xshape))
            ax_seg.set_title("Segmentation")
        elif x_gt is not None:
            self.fig, (ax_gt, self.ax) = plt.subplots(1, 2)
            ax_gt.imshow(np.reshape(x_gt, self.xshape))
            ax_gt.set_title("Ground truth")
        else:
            self.fig, self.ax = plt.subplots(1, 1)
        self.imdata = self.ax.imshow(np.reshape(x, self.xshape))
        self.ax.set_title("Reconstruction")

    # makes the object callable like a function.
    # dummy second variable cause some sweep_callbacks take two parameters
    def __call__(self, y, dummy=None):
        self.imdata.set_data(np.reshape(y, self.xshape))
        plt.pause(0.001)


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




def test_reconstruction_solver():
    resolution = 100
    directions = 180

    x, A, b = create_projection_data_(resolution, directions)
    b_noisy = b + np.random.randn(*b.shape) * 1.0
    d, _, _ = initial_segmentation(x, 5, noise_level=0.1)

    u, v = np.mgrid[-1:1:resolution*1j, -1:1:resolution*1j]
    roi = u**2 + v**2 < 0.125
    # start with a fake ROI, should be the same as full volume!
    # roi = np.ones_like(x, dtype=bool)

    An, bn = reduce_and_normalise_system(A, b_noisy.ravel())
    dims = (2, x.shape)
    alpha = 0.01
    beta = 0.02
    x0 = np.random.rand(*x.shape) * x.max()
    x0.shape = x0.size
    d.shape = d.size
    roi.shape = roi.size

    dp = DisplayImage(x0, dims[1], x_gt=x, x_seg=d)
    obj_func = Objective(An, bn, alpha, beta, dims, d)

    damped_art = Damped_ART_TV_Segmentation(An, bn, alpha, beta, d, dims, x0, roi=roi)
    damped_art.add_start_callback(dp)
    # damped_art.add_start_callback(obj_func)
    damped_art.add_sweep_callback(obj_func)
    damped_art.add_sweep_callback(dp)
    damped_art.add_end_callback(obj_func)
    damped_art.step_callback(Standard_step(t=10.0, a=0.7))
    damped_art.max_iterations = 1
    damped_art.iterations_fista = 10
    damped_art.tolerance = 1e-5
    damped_art.solve()

    _, ax = plt.subplots(1,1)
    ax.plot(obj_func.values, label="objective function values")
    ax.grid('on')
    ax.set_xlabel("iterations")
    ax.set_label(r"$f(x_k)$")
    ax.legend()
    plt.show()


def cluster_means(x, v):
    """
    compute cluster means from image data and membership vector v

    Parameters:
    -----------
    img: float ndarray
        array of size (m1, m2) or (m1, m2, m3) representing the data
    v: float ndarray
        array of size (m1, m2, K) or (m1, m2, m3, K) representing the
        cluster memberships

    Returns:
    -------
    c: float ndarray
        array of size (K,) of cluster values
    """

    # start by flattening temporarily
    img_shape = x.shape
    nb_pixels = x.size
    x.shape = nb_pixels
    K = v.shape[-1]
    v.shape = (nb_pixels, K)

    c = (x @ v) / v.sum(axis=0)

    v.shape = img_shape + (K,)
    x.shape = img_shape + ()
    return c


def kmeans_sqd_vector(x, c):
    """
    k-means squared-distance between sample and means vector

    Parameters
    ----------
    x : float(32) ndarray
        (m1,m2) or (m1,m2,m3) image/data array
    c : float ndarray
        (K,) array of cluster centers

    Returns
    -------
    g: float(32) ndarray
        array of shape(m1, m2, K) or (m1, m2, m3, K) containing the
        square distances (img[ijl] - c[k])^2
    """
    K = len(c)
    ndims = x.ndim
    img_shape = x.shape
    x.shape = img_shape + (1,)
    c.shape = (1,) * ndims + (K,)

    g = (x - c) ** 2
    x.shape = img_shape
    c.shape = K
    return g


def kmeans_sq_distance_vector(x, c):
    """
    k-means squared-distance between sample and means vector

    Parameters
    ----------
    x : float(32) ndarray
        (m1,m2) or (m1,m2,m3) image/data array
    c : float ndarray
        (K,) array of cluster centers

    Returns
    -------
    g: float(32) ndarray
        array of shape(N, K)  containing the
        square distances (x[n] - c[k])^2
    """
    K = len(c)
    x.shape = (-1, 1)
    c.shape = (1, K)

    g = (x - c) ** 2
    x.shape = x.size
    return g


###########################################
# getting a bit better...
############################################


class Display_NoROI_Field_Regularisation:
    def __init__(self, recons, gt, n_classes, delay=0.01):
        """Function object used as callback in test_tv_proximal_vectorial."""
        self.gt = gt
        self.delay = delay
        self.c = np.arange(n_classes)
        start_value = self.prepare_image(self.gt, label_data=True)
        self.fig, (self.ax_recons, self.ax_orig, self.ax_evol) = plt.subplots(1, 3)
        self.recons_obj = self.ax_recons.imshow(recons)
        self.gtobj = self.ax_orig.imshow(start_value)
        self.dobj = self.ax_evol.imshow(start_value)

    def prepare_image(self, v, label_data=False):
        bv = v if label_data else np.squeeze(v @ self.c)
        return bv + 0.5 * self.gt

    def change_gt(self, new_gt):
        new_gt = self.prepare_image(new_gt)
        self.gtobj.set_data(new_gt)
        plt.pause(self.delay)

    def change_recons(self, recons):
        self.recons_obj.set_data(recons)
        plt.pause(self.delay)

    def change_title(self, message):
        self.fig.suptitle(message)

    def __call__(self, x, k):
        self.dobj.set_data(self.prepare_image(x))
        self.ax_evol.set_title(f"Iteration {k}")
        plt.pause(self.delay)


def prepare_data(resolution, directions, sinogram_noise_level=1, n_classes=5, segmentation_noise_level=0.1):
    x, A, b = create_projection_data_(resolution, directions)
    b_noisy = b + np.random.randn(*b.shape) * sinogram_noise_level
    d, v0, centres = initial_segmentation(x, n_classes, noise_level=segmentation_noise_level)
    An, bn = reduce_and_normalise_system(A, b_noisy.ravel())
    return x, An, bn, d, v0, centres


def prepare_roi(resolution, radius):
    u, v = np.mgrid[-1:1:resolution*1j, -1:1:resolution*1j]
    return u**2 + v** 2 < radius


def test_full_solver_no_roi():
    resolution = 100
    directions = 180
    n_classes = 5

    # get phantom, projection matrices
    x, A, b, d, v, centres = prepare_data(resolution, directions, n_classes=n_classes)
    roi = None

    alpha = 0.01
    beta = 0.02
    gamma = 0.01

    dims = (2, x.shape)
    # init reconstruction with uniform noise
    x0 = np.random.rand(*x.shape) * x.max()
    x0.shape = x0.size
    d.shape = d.size
    x.shape = x.size

    # step parameters
    t = 10
    a = 0.7

    dp = DisplayImage(x0, dims[1], x_gt=x, x_seg=d)
    obj_func = Objective(A, b, alpha, beta, dims, d)

    damped_art = Damped_ART_TV_Segmentation(A, b, alpha, beta, d, dims, x0)
    damped_art.add_start_callback(dp)
    damped_art.add_sweep_callback(obj_func)
    damped_art.add_sweep_callback(dp)
    damped_art.add_end_callback(obj_func)
    damped_art.max_iterations = 1

    seg_proximal = Segmentation_proximal(beta, gamma, dims, M=roi)
    df = Display_NoROI_Field_Regularisation(np.reshape(x0, dims[1]), np.reshape(d, dims[1]), n_classes)
    seg_proximal.add_TV_callback(df)
    seg_proximal.iterations = 50

    for k in range(20):
        ss = Special_Step(t=t, a=a, k=k)
        damped_art.step_callback(ss)
        damped_art.solve()

        # need to compute g from damped_art.x and centres
        g = kmeans_sq_distance_vector(damped_art.x, centres)
        g.shape = dims[1] + (n_classes,)
        tau = ss()
        df.change_gt(v)
        df.change_recons(np.reshape(damped_art.x, dims[1]))
        df.change_title(f"At sweep {k}")
        seg_proximal.solve(v, g, tau)

        centres = cluster_means(damped_art.x, seg_proximal.v)
        d = np.squeeze(seg_proximal.v @ centres)
        d.shape = d.size
        damped_art.d = d


if __name__ == "__main__":
    # plt.ion()
    test_full_solver_no_roi()
    input("Press a key to terminate...")

















































