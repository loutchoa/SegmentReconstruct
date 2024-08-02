# -*- coding: utf-8 -*-
"""
@project: SegmentReconstruct
@file: slice_segmentation.py
@description: An example of segmentation of a slice using
an 'extended' FISTA model.

François Lauze, Chrysoula Stathaki
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from projectors import (
    project_on_simplex_field,
    project_on_simplex_vertex
)
from label_simplex_fields import (
    label_to_simplex_field,
    simplex_to_label_field
)
from graddiv import grad2v as D
from graddiv import div2v as Div
import fistaTV

__author__ = "François Lauze, University of Copenhagen"
__date__ = "3/11/22"
__version__ = "0.0.1"


def weighted_moments(v, x):
    """Compute mean and variance of v-weighted samples x."""
    v_sum = v.sum()
    v_shape = v.shape
    v.shape = v.size
    x.shape = x.size
    c = (v @ x) / v_sum
    sigma2 = (v @ (x - c) ** 2) / v_sum
    v.shape = v_shape
    x.shape = v_shape
    return c, sigma2


def weighted_moments_priors(v, x, delta, c0, max_iters=100, tol=1e-10, eps=1e-14, verbose=False):
    """
    Compute mean and variance of v-weighted samples x with a mean prior.

    It solves the minimisation problem in c, sigma:
    F(c,sigma) = Sum(v_n)*log(sigma) + (1/(2*sigma**2))Sum(v_n(c-x_n)**2) +
                 (delta/2)(c - c0)**2
    It is solved in sigma**2, not sigma. It is easy to see that sigma2 is
    the value of the variance function V(x) = Sum v_n(x-x_n)**2/Sum v_n
    at x = c. The equation in c, after sigma**2 substitution is a  cubic
    polynomial, and one solves for the root closest to the prior value c0.

    Parameters:
    ----------
    x : numpy array
        samples in x, size (m, n)
    v : numpy array
        weights for the samples in x, size (m, n)
    delta: float
        prior mean weight
    c0: float
        prior mean
    max_iters: int, optional
        number of iterations in Newton-Raphson method
    tol: float
        tolerance for stopping iterations.
    eps: float
        test for the derivative size in Newton-Raphson.
        If the derivative is in absolute value smaller than eps, stop
        and report absence of convergence
    verbose: bool, optional
        if True, write a few things about convergence.
    Return:
    ------
    float c: the shifted mean
    float sigma2: the variance around c
    bool converged: indicates that the solver converged (weird if not?)
    """
    phi = v.sum()
    v_shape = v.shape
    v.shape = v.size
    x.shape = x.size
    psi = v @ x
    chi = v @ x ** 2

    A = delta * phi
    B = -delta * (2 * psi + phi * c0)
    C = phi ** 2 + delta * chi + 2 * delta * psi * c0
    D = -delta * chi * c0 - phi * psi

    # the mean c should be the solution of Ac**3 + Bc**2 + C*c + D = 0
    # which is closest to c0. So the solution is obtained via a Newton-Raphson
    # method starting from c0.
    def f(y):
        return A * y ** 3 + B * y ** 2 + C * y + D

    def df(y):
        return 3 * A * y ** 2 + 2 * B * y + C

    c = c0
    converged = False
    bad_gradient = False
    dfc = 0
    for i in range(max_iters):
        dfc = df(c)
        if abs(dfc) < eps:
            # derivative is too close to zero!
            bad_gradient = True
            break
        c_new = c - f(c) / dfc
        if abs(c - c_new) < tol:
            converged = True
            break
        c = c_new

    sigma2 = c ** 2 - 2 * c * psi / phi + chi / phi
    v.shape = v_shape
    x.shape = v_shape
    if verbose:
        if converged:
            print(f'Newton-Raphson successful for shifted mean update, converged at iteration {i} to tolerance {tol}.')
        else:
            print(f'Newton-Raphson failed for shifted mean update.')
            if bad_gradient:
                print(f'\tbad gradient {dfc}, with norm less than allowed value {eps} at iteration {i}.')
            else:
                print(f'\tno convergence after {max_iters} iterations (bug?)')

    return c, sigma2, converged


def compute_gaussian_parameters(v, x, means_prior=None, delta=None, verbose=False):
    """
    Compyte parameters for the Gaussian mixture for the segmentation.

    Without priors, this is a classical mean/variance estimation. With
    priors, computing the mean is more complicated (degree 3 equation).
    Parameters:
    ----------
    v : float(32) numpy array
        current segmentation field, size (m,n,k)
    x : float(32) numpy array
        image to be segmented, size (m,n)
    means_prior: float(32) numpy array or None
        segment mean priors, size (k) if not None.
        The default is None, no mean prior.
    delta: float, float numpy array, or None
        weight of the mean prior. The default is None, no mean prior
    verbose: bool, optional
        Passed to weighted_moments_priors, if a prior is present. The
        default value is False.
    """
    m, n, k = v.shape
    means = np.zeros(k)
    variances = np.zeros(k)

    # without prior
    if means_prior is None or delta is None:
        for i in range(k):
            means[i], variances[i] = weighted_moments(v[:, :, i], x)

    else:
        if type(delta) is float:
            delta_array = np.array([delta]*k)
        else:
            delta_array = delta
        for i in range(k):
            means[i], variances[i], converged = weighted_moments_priors(v[:, :, i], x, delta_array[i], means_prior[i], verbose=verbose)
            if not converged:
                warnings.warn(f"Newton-Raphson did not converge for segment mean {i}.", RuntimeWarning)
    return means, variances


def data_vector(x, means, variances):
    """Create the vector g in the segmentation."""
    m, n = x.shape
    k = len(means)
    g = np.zeros((m, n, k))
    for i in range(k):
        g[:, :, i] = 0.5 * np.log(variances[i]) + (x - means[i]) ** 2 / (2 * variances[i])
    return g


def init_segmentation(x, k, means_prior):
    """Initialise a segmentation via a Gaussian Mixture Model fit."""
    x_shape = x.shape
    x.shape = -1, 1
    gmm = GaussianMixture(n_components=k, means_init=np.reshape(means_prior, (-1, 1)))
    labels = gmm.fit_predict(x)
    x.shape = x_shape
    labels.shape = x_shape
    means = np.squeeze(gmm.means_)
    variances = np.squeeze(gmm.covariances_)
    v = label_to_simplex_field(labels)
    # Now we sort in increasing means order
    indices = np.argsort(means)
    means = means[indices]
    variances = variances[indices]
    v = v[:, :, indices]
    return v, means, variances


class Step:
    """A step object, as used in other code parts."""

    def __init__(self, t=0.1, a=1.0):
        self.t = t
        self.a = a

    def __call__(self, i):
        return self.t / (i + 1) ** self.a


class MultiSegmentationDisplay:
    """Show a multiclass segmentation in progress."""

    def __init__(self, x, v, means, title="Segmentation"):
        """Init the display class."""
        self.x = x
        self.fig, self.axes = plt.subplots(1, 3, sharey=True, sharex=True)
        for ax in self.axes:
            ax.axis('off')
        self.fig.suptitle(title)
        self.soft_field = v
        self.K = v.shape[-1]
        self.means = means
        self.imdata_soft = None
        self.imdata_hard = None
        self.soft_segment = None
        self.hard_segment = None

        self.axes[0].imshow(x, cmap='Greys_r')
        self.axes[0].set_title("Image to be segmented.")
        self.represent_segmented_images()
        self.imdata_soft = self.axes[1].imshow(self.soft_segment)
        self.imdata_hard = self.axes[2].imshow(self.hard_segment)

    def update_field(self, v):
        """Update the segmentation field."""
        self.soft_field = v

    def update_mean(self, means):
        """Update the segment mean values."""
        self.means = means

    def represent_segmented_images(self):
        """Compute soft and hard segmented images using segment labels and means."""
        hard_field = self.soft_field.copy()
        project_on_simplex_vertex(hard_field)
        self.soft_segment = np.zeros_like(self.x)
        self.hard_segment = np.zeros_like(self.x)
        for k in range(self.K):
            self.soft_segment += np.squeeze(self.soft_field[:, :, k] * self.means[k])
            self.hard_segment += np.squeeze(hard_field[:, :, k] * self.means[k])

    def display_fista(self, v, i):
        """Update display as a TVProximal callback."""
        self.update_field(v)
        self.represent_segmented_images()
        self.imdata_soft.set_data(self.soft_segment)
        self.imdata_hard.set_data(self.hard_segment)
        self.axes[1].set_title(f'Soft segmentation, in FISTA, iteration {i}.')
        self.axes[2].set_title(f'Hard segmentation, in FISTA, iteration {i}.')
        plt.pause(0.01)

    def display_moment_change(self, means, new_title=None):
        """Update displays when means have been recomputed."""
        self.update_mean(means)
        self.represent_segmented_images()
        self.imdata_soft.set_data(self.soft_segment)
        self.imdata_hard.set_data(self.hard_segment)
        self.axes[1].set_title(f'In proximal descent.')
        self.axes[2].set_title(f'In proximal descent.')
        if new_title is not None:
            self.fig.suptitle(new_title)
        plt.pause(0.01)


def segment_slice_TV_GMM(x, alpha, K, step, means_prior=None, delta=None, max_iters=10, fista_iters=50, tol=1e-4):
    """
    Segment an image x in k segments with TV regularisation and potential mean prior.

    It iterates over FISTA steps with simplex projection and computation of means
    and variances. The segmentation is initialised with a Gaussian-Mixture-Model fit.

    Parameters:
    ----------
    x: float(32) numpy array
        image to be segmented
    alpha: float
        TV regularisation weight
    k: int
        number of segments
    step: function
        step(i) provides the proximal step length in iteration i.
    means_prior: float(32) numpy array or None, optional
        prior values on means, if not None. The default is None.
    delta: float, optional
        mean prior weight. The default is 1e10
    max_iters: int, optional
        maximum number of iterations. Each iteration makes a proximal
        for the label field / HMMFM and an update on the segment parameters.
        The default is 10.
    fista_iters: int, optional
        number of FISTA_TV iterations. The default is 50.
    tol : float, optional
        tolerance for convergence. The default is 1e-4

    Return:
    -------
        v : label field for the segmentation.
        means: segment means
        variances: segment variances
    """

    v, means, variances = init_segmentation(x, K, means_prior)
    ds = MultiSegmentationDisplay(x, v, means, title=f'Segmentation with {K} classes.')
    verbose = True

    g = data_vector(x, means, variances)
    tv_prox = fistaTV.TVProximal(
        D, Div, 2,
        projectC=project_on_simplex_field,
        vectorial=True,
        iterations=fista_iters
    )
    tv_prox.reset_dual_variable = True
    tv_prox.add_callback(ds.display_fista)
    for i in range(max_iters):
        tau = step(i)
        tv_prox.gamma = 2 * tau * alpha

        new_v = tv_prox.run(v - tau * g)
        new_means, new_variances = compute_gaussian_parameters(new_v, x, means_prior, delta, verbose=verbose)
        ds.display_moment_change(new_means, new_title=f'Done outer iteration {i}')
        if np.linalg.norm(new_means - means) < tol:
            break
        else:
            v = new_v
            means = new_means
            variances = new_variances
            g = data_vector(x, means, variances)

    ds.display_moment_change(new_means, new_title=f'Done segmentation.')
    return v, means, variances


def segmentation_test():
    x = plt.imread('SBAa_PBS_TT_ROI.tif').astype("float32")
    x_min = x.min()
    x_max = x.max()

    def to01(u):
        return (u - x_min)/(x_max - x_min)

    c0 = np.array([6000, 6800, 7500, 8500, 9900])

    x = to01(x)
    c0 = to01(c0)

    K = 5
    # alpha = 7.e6
    # delta = 10.e10
    alpha = 1.25
    delta = np.array([50., 1000., 500., 50., 80000])
    # delta = 10000*np.ones(K)
    step = Step(t=1.0)
    v, means, variances = segment_slice_TV_GMM(x, alpha, K, step, means_prior=c0, delta=delta, tol=1e-10, fista_iters=25, max_iters=25)
    input("Press a key to terminate...")


if __name__ == "__main__":
    segmentation_test()