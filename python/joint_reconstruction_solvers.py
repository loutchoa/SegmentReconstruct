#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: SegmentReconstruct
@filename: joint_reconstruction_solvers.py
    
@description: a ART with FISTA for a reconstruction part with
added reaction to segmentation. The segmentation proximal steps.

In the first implementation, of the reconstruction part, I assume,
like Chrysoula, that the segmentation ROI is the entire image, and this
simplifies somewhat the algebra.
In the second, remove this assumption, it becomes a slightly more complicated
with the introduction of a weighted TV minimisation, which is clearly different
than the usual weighted one!

I also provide a proximal step for the segmentation. The difficulty with it is
that in case of a ROI, discrete gradient and divergences need to be provided that
account for the behaviour at boundary.

François Lauze, Chrysoula Stathaki

"""
from random import shuffle

import matplotlib.pyplot as plt
import numpy
import numpy as np
# import  matplotlib.pyplot as plt
from fistaTV import (
    fistaTV, fistaTV_weighted, TVProximal
)
from tomo_utils import clean_projection_system, normalise_projection_system
from graddiv import (
    grad2, grad3, grad2v, grad3v,
    div2, div3, div2v, div3v
)
from irregular_domain import (
    Stencil2D, Stencil3D
)
from projectors import project_on_simplex_field


__author__ = "François Lauze, University of Copenhagen"
__date__ = "01-17-2022"
__version__ = "0.0.1"


# With ideas from "kaczmarz algorithms" GitHub repos
# Put max_iterations and tolerance out of init?
class Kaczmarz:
    def __init__(self, A, b, rho=1.0, x0=None, max_iterations=100, tolerance=1e-6,
                 callback_start=None, callback_sweep=None, callback_end=None):
        """Kaczmarz row-action method solvers (as opposed to block-row).

        Proposes two strategies: cyclic and random row sweep.

        Parameters:
        -----------
        A : csr matrix float or float32
            system / projection matrix of size (M, N)
        b : numpy float/float32 array
            system second member, size M
        rho : float, optional.
            relaxation parameter. It must be in (0,2). The default is 1.0
        x0 : numpy float/float32 array, optional
            solution guess, size N. The default is None, means null solution.
        max_iterations: int, optional.
            maximum number of iterations. The default is 100
        tolerance: float, optional
            convergence threshold. The default is 1e-6.
        callback_start: function(x)->None, optional.
            if not none, called before iterations start.
        callback_sweep: function(x)->None, optional.
            if not none, called at end of each sweep.
        callback_end: function(x)->None, optional.
            if not none, called before returning.
        """
        self.A = A
        self.M, self.N = A.shape
        self.b = b
        self.rho = rho
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.N, dtype=A.dtype)
        self.xk = self.x0.copy()

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cb_start = callback_start
        self.cb_sweep = callback_sweep
        self.cb_end = callback_end
        self._method = 'randomised'
        self.rows = None

    def start_callback(self):
        """Called when solve() start."""
        if self.cb_start is not None:
            self.cb_start(self.xk)

    def sweep_callback(self):
        """Called at end of each sweep."""
        if self.cb_sweep is not None:
            self.cb_sweep(self.xk)

    def end_callback(self):
        """Called when solve() is about to return."""
        if self.cb_end is not None:
            self.cb_end(self.xk)

    @property
    def strategy(self):
        return self._method

    @strategy.setter
    def strategy(self, method):
        if method in ('randomised', 'cyclic'):
            self._method = method
        else:
            raise ValueError(f'in strategy(), method must be "randomised" or "cyclic". Got f{method}')

    def sweep_order(self):
        self.rows = list(range(self.M))
        if self._method == 'randomised':
            shuffle(self.rows)

    def solve(self):
        """
        Runs the actual solver.
        """
        size = self.M * self.N
        self.start_callback()

        for k in range(self.max_iterations):
            self.sweep_order()
            for i in self.rows:
                ai = self.A[i]
                bi = self.b[i]
                self.xk += self.rho * (bi - ai @ self.xk) * ai
            self.sweep_callback()
        self.end_callback()


class Standard_step:
    def __init__(self, t=0.1, a=1.0):
        """
        Standard time step: t / (k + 1)**a.
        """
        self.t = t
        self.a = a

    def __call__(self, k):
        return self.t / (k + 1) ** self.a


class Special_Step:
    def __init__(self, t=0.1, a = 1.0, k=1):
        self.t = t
        self.a = a
        self.k = k

    def __call__(self, dummy=None):
        return self.t/(self.k + 1) ** self.a


def solver_smw(a, W, b, d):
    """
    Solve (a a.T + W^{-1})x = ba + d via Sherman-Morrison-Woodbury formula.

    Args:
        a ((n,1) ndarray): it should be the transpose of a given projection matrix line
        W ((n) ndarray): diagonal of a diagonal matrix, entries > 0
        b (float): measurement
        d (n) ndarray: term coming from segmentation and proximal
    """
    # Yet another attempt not to densify !
    ad = a.data
    idx = a.indices
    Wd = W*d
    Wa = W[idx] * ad
    L = ((b - ad @ Wd[idx]) / (1 + ad @ Wa)) * Wa
    Wd[idx] += L
    return Wd


class Inf:
    plus = float("inf")
    minus = -float("inf")


class Damped_ART_TV_Segmentation:
    def __init__(self, A, b, alpha, beta, d, dims, x0, rho=1.0, roi=None):
        """Damped ART with regularisation and reaction to segmentation.

        This class configures and runs the reconstruction in the joint
        algorithm.
        The system (A, b) is assumed to have been normalised: no null row
        in A, each row has norm 1, and b is normalised accordingly-

        - The maximum number of sweeps i by default 100 and can be changed
        from the corresponding method.
        - The number of FISTA-TV iterations is by default 50 and can be changed
        from the corresponding method.
        - the box constraints are by default set to [0, +inf]. They can be
        changed from the corresponding method.
        - the convergence threshold is set to 1e-3 and can be changed from the
        corresponding method.
        - step has the form t / (k + 1)**a where k is the iteration number.
        Another step strategy can be provided via a step callback method.

        TODO: vectorial values for x would be nice for parallel tomo!
            I just don't want to add it now, would make debugging more
            complicated.

        Parameters:
        ----------
        A : CSR float/float32
            projection matrix.
        b : float/float32 numpy array
            system second member / measurements. Size M with M the number of
            rows of A.
        alpha : float
            TV regularisation weight. If set to zero, regularisation is
            skipped.
        beta : float
            reaction to segmentation weight. If set to zero, segmentation
            reaction is skipped.
        d : float/float32 numpy array.
            segmentation cartoon image. Same size as x0.
        dims : tuple/list
            (ndim, shape) pair, where ndim should be 2 or 3 for a planar
            or volumic data, and shape is the non ravelled shape of x0
            (and d).
        x0 : float/float32 numpy array
            Solution estimate. size N with N the number of columns of A.
        rho : float, optional
            relaxation parameter. Should be in [0,1], default is 1.0
        roi : bool numpy array, optional
            Length N array representation the characteristic function of the
            segment ROI. If None, the roi is the entire volume.
            The default is None.
        """
        self.A = A
        self.b = b
        self.M, self.N = A.shape
        self.alpha = alpha
        self.beta = beta
        self.dims = dims
        self.d = d
        self.dims = dims
        self.x = x0.copy()
        self.previous_x = None
        self.rho = rho
        self.full_volume = roi is None
        self.roi = 1.0 if self.full_volume else roi
        self.regularise = alpha > 0.0
        self.react = beta > 0
        self.gamma = self.beta / (self.M + 1)

        # solver execution callbacks, can be used to
        # save or display information during execution.
        # start callbacks should have signature callback(x) -> None
        self.start_callbacks = []
        # sweep callbacks should have signature callback(x, k) -> None
        self.sweep_callbacks = []
        # end callbacks should have signature callback(x) -> None
        self.end_callbacks = []

        # step callback: must be callable step(k : int) -> float
        # by default it is t/(k+1)**a, t = 0.1, a = 1.0
        self.step = Standard_step()

        self._max_iterations = 100
        self._fista_iterations = 50
        self._tolerance = 1e-3
        self._method = 'randomised'
        self.rows = None
        self._box = (0.0, float("inf"))
        self.iteration = -1

    @property
    def strategy(self):
        return self._method

    @strategy.setter
    def strategy(self, method):
        if method in ('randomised', 'cyclic'):
            self._method = method
        else:
            raise ValueError(f'in strategy(), method must be "randomised" or "cyclic". Got f{method}')

    def sweep_order(self):
        self.rows = list(range(self.M))
        if self._method == 'randomised':
            shuffle(self.rows)

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, n):
        if n < 0:
            raise ValueError(f"maximum number of iterations must be >= 0.")
        self._max_iterations = n

    @property
    def iterations_fista(self):
        return self._fista_iterations

    @iterations_fista.setter
    def iterations_fista(self, n):
        if n < 0:
            raise ValueError(f"number of fista iterations must be >= 0.")
        self._fista_iterations = n

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, xmin=None, xmax=None):
        if xmin is None:
            xmin = self._box[0]
        if xmax is None:
            xmax = self._box[1]
        if xmin >= xmax:
            raise ValueError(f"the box is reduced to 1 value or empty.")
        self._box = (xmin, xmax)

    def box_project(self):
        if self._box[0] > Inf.minus:
            np.place(self.x, self.x < self._box[0], self._box[0])
        if self._box[1] < Inf.plus:
            np.place(self.x, self.x > self._box[1], self._box[1])

    def add_start_callback(self, callback):
        """Add a callback executed at start of the reconstruction part."""
        self.start_callbacks.append(callback)

    def add_sweep_callback(self, callback):
        """Add a callback executed at end of a sweep."""
        self.sweep_callbacks.append(callback)

    def add_end_callback(self, callback):
        """Add a callback executed at end of the reconstruction part."""
        self.end_callbacks.append(callback)

    def run_start_callbacks(self):
        for callback in self.start_callbacks:
            callback(self.x)

    def run_sweep_callbacks(self):
        for callback in self.sweep_callbacks:
            callback(self.x, self.iteration)

    def run_end_callbacks(self):
        for callback in self.end_callbacks:
            callback(self.x)

    def step_callback(self, callback):
        self.step = callback

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tol):
        if tol < 0:
            self._tolerance = 0.0
        else:
            self._tolerance = tol

    def converged(self):
        """
        Convergence test.

        This could also be parameterised via a callback,
        or at least easily changed...
        """
        return np.linalg.norm(self.x - self.previous_x) / self.x.size < self._tolerance

    def solve(self):
        """Runs a sweep of proximals."""
        self.run_start_callbacks()

        Vk = None  # dummy stuff to remove a PyCharm message
        Wk = None  # idem
        for self.iteration in range(self._max_iterations):
            self.previous_x = self.x.copy()

            tauk = self.step(self.iteration)
            ck = 1 / tauk if not self.react else self.gamma + 1 / tauk

            # The weight matrix for the region of interest, if not the full volume
            if not self.full_volume:
                Vk = (1 / tauk) * np.ones_like(self.x) + self.gamma * self.roi
                Wk = 1.0 / Vk

            # get straight or randomised row order
            self.sweep_order()
            for i in self.rows:
                ai = self.A[i]
                bi = self.b[i]

                # Todo? factor properly or everything in C?
                # maybe I should have specific methods?
                # no reaction toward segmentation:
                if not self.react:
                    z = self.x + ((bi - ai @ self.x) / (1 + ck)) * ai
                else:
                    # segmentation ROI is the full volume
                    if self.full_volume:
                        dki = (self.gamma * self.roi * self.d + (1 / tauk) * self.x) / ck
                        z = dki + ((bi - ai @ dki) / (1 + ck)) * ai
                    else:
                        # segmentation ROI is a smaller volume than x
                        dki = (self.gamma * self.roi * self.d + (1 / tauk) * self.x)
                        z = solver_smw(ai, Wk, bi, dki)

                self.x = (1 - self.rho) * self.x + self.rho * z
                self.x.shape = self.x.size

            # Done with matrix rows

            # if TV regularisation is needed:
            if self.regularise:
                if not self.react:  # no segmentation
                    self.x.shape = self.dims[1]
                    z = fistaTV(self.x, 2 * self.alpha * tauk, self._fista_iterations, xmin=Inf.minus)
                    self.x.shape = self.x.size
                else:
                    dki = (self.gamma * self.roi * self.d + (1 / tauk) * self.x) / ck
                    dki.shape = self.dims[1]
                    if self.full_volume:
                        z = fistaTV(dki, 2 * self.alpha / ck, self._fista_iterations, xmin=Inf.minus)
                    else:  # segmentation with region of interest
                        Wk.shape = self.dims[1]
                        z = fistaTV_weighted(dki, 2 * self.alpha, Wk, self._fista_iterations, omega=tauk, xmin=Inf.minus)
                        Wk.shape = Wk.size
                z.shape = z.size
                self.x = (1 - self.rho) * self.x + self.rho * z

            self.run_sweep_callbacks()
            if self.converged():
                break

        self.run_end_callbacks()


# Ala Andersen-Hansen, or all the v stuff in one step?
# the TV Proximal class can easily run it in one "step"
class Segmentation_proximal:
    def __init__(self, beta, gamma, dims, rho=1.0, method='fwsp', M=None):
        """Compute a proximal for the segmentation part.

        The class sets the weights, dimensions and domain.
        Different methods will control which kind of computation is actually
        performed. There are four choices:
            - 'fwsp' : Fista-TV with simplex projection: Fista is run, directly
                incorporating the simplex projection,
            - 'fwosp' : no simplex projection, just Fista-TV
            - 'gpwsp' : Beck and Teboulle's gradient projection method,
                with simplex projection,
            -  'gpwosp' : the same with simplex projection.

        Two types of callbacks:
            - callbacks with signature callback(v, k)->None can
            be added, they are actually passed to the TV Proximal calculation
            via the add_tv_callback(9 method
            - callbacks with signature callback(x, v, c)->None are added with
             add_seg_callback(). For instance to record the objective values.
             (maybe it's too much...)

        Parameters:
        ----------
        beta : float
            weight of the segmentation term
        alpha : float
            weight of the regularisation
        dim : list (int, shape)
            dimensionality of the problem and true image shapes,
            dimensionality should be 2 or 3, true image shape shoud
            be a int list
        rho : float, optional
            (over)relaxation parameter. The default is 1.0
        M : bool numpy array, optional
            the domain mask. If not specified, it means the whole domain.
            the default value is None
        """
        self.domain = M
        self.dims = dims
        self.gamma = gamma
        self.beta = beta
        self.rho = rho
        self.stencil = None
        self.D = None
        self.Div = None
        self._method = method
        self.seg_prox_callbacks = []
        self.v = None
        self._keep_dual_variable = False

        if self.domain is None:
            if self.dims[0] == 2:
                self.D = grad2v
                self.Div = div2v
            else:
                self.D = grad3v
                self.Div = div3v
        else:
            if self.dims[0] == 2:
                self.stencil = Stencil2D(self.domain)
                self.D = self.stencil.gradient_flattened
                self.Div = self.stencil.divergence_flattened
            else:
                self.stencil = Stencil3D(self.domain)
                self.D = self.stencil.gradient_flattened
                self.Div = self.stencil.divergence_flattened

        self.tv_prox = TVProximal(self.D, self.Div, self.dims[0], vectorial=True)

    @property
    def iterations(self):
        return self.tv_prox.iterations

    @iterations.setter
    def iterations(self, k):
        self.tv_prox.iterations = k

    def add_TV_callback(self, tv_callback):
        self.tv_prox.add_callback(tv_callback)

    def reset_TV_callbacks(self):
        self.tv_prox.callbacks = []

    def add_seg_callbacks(self, seg_callback):
        self.seg_prox_callbacks.append(seg_callback)

    @property
    def keep_dual_variable(self):
        return self._keep_dual_variable

    @keep_dual_variable.setter
    def keep_dual_variable(self, yesno):
        self._keep_dual_variable = yesno

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, name):
        if name in ('fwsp', 'fwosp', 'gpwsp', 'gpwosp'):
            self._method = name
            if name in ('fwsp', 'fwosp'):
                self.tv_prox.algorithm = 'fista'
            else:
                self.tv_prox.algorithm = 'gp'
        else:
            raise ValueError(f'method "{name}" illegal. Must be one of "fwsp", "fwosp", "gpwsp", "gpwosp".')

    def solve(self, v, g, tau):
        self.tv_prox.gamma = self.gamma * tau
        self.tv_prox.reset_dual_variable = not self.keep_dual_variable

        # in the first case, no simplex projection is performed during proximal calculations
        if self._method not in ('fwsp', 'gpwsp'):
            self.v = self.tv_prox.run(v - 0.5 * self.beta * tau * g)
        # in that case, it is.
        else:
            self.tv_prox.init_Pc(project_on_simplex_field)
            self.v = self.tv_prox.run(v - 0.5 * self.beta * tau * g)

        v[...] = (1 - self.rho) * v + self.rho * self.v
        if self.method in ('fwosp', 'gpwosp'):
            project_on_simplex_field(self.v)
        return self.v






















#