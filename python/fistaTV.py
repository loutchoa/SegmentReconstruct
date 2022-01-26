# coding=utf-8
"""
Project: PyCT
File: fistaTV

Description: Implementation of TV denoising via Fast
Iterative Shrinkage Thresholding Algorithm (FISTA)
following Beck and Teboulle's paper.'

Works in dimension 2 and 3

Author: François Lauze, University of Copenhagen
Date: 08-2016
Small stuffs: cleaning of the code and Python, August 2021
"""

import numpy as np
from graddiv import grad2, grad3, div2, div3
from operator import mul
from functools import reduce
from projectors import (
    project_on_field_of_balls,
    project_on_field_of_Frobenius_balls
)
from math import sqrt
import numbers
# from scalespace import gaussian_scalespace


def Pc(x, xmin, xmax):
    """
    Pc: simple box constraint projection
    """
    np.place(x, x < xmin, xmin)
    np.place(x, x > xmax, xmax)
#
#
# def project_on_field_of_balls(psi, l=1.0):
#     """
#     project a vector field on the field of radius l balls,
#     i.e. each vector v of the field is projected in a radius l ball. If
#     |v| <= l, nothing to do, if |v| > l, normalise v so that its norm is
#     actually l.
#
#     :param psi: numpy array.
#         the vector field to be projected
#     :param l: float. Then radius of each ball.
#     :return: psi projected
#     """
#
#     psishape = psi.shape
#     vdim = psishape[-1]
#     rdim = reduce(mul, psishape[:-1], 1)
#     psi.shape = (rdim, vdim)
#
#     psinorm = np.sqrt(np.sum(psi ** 2, axis=1))
#     psinorm = (psinorm / l) * (psinorm > l) + np.ones_like(psinorm) * (psinorm <= l)
#     # np.place(psinorm, psinorm <= 1.0, 1.0)
#     for i in range(vdim):
#         psi[:, i] /= psinorm
#     psi.shape = psishape
#     return psi


def fistaTV(b, l, n, xmin=0.0, xmax=float("inf")):
    """Compute the l-proximal of the TV-norm with range constraints.

    .. math::
    argmin_x \|x-b\|^2 + 2l\|Dx\|,\quad xmin <= x  <= xmax

    i.e. the proximal $prox_f(b)
    where f = l*\|D.\| + i_B(.) with

    i_B(x) = +infinity if one of the components of x does not
    satisfy xmin <= x_i <= xmax, and i_B(x) = 0 otherwise,
    i.e.,  identical bounds box constraint

    Parameters:
    ----------
    b : numpy float32 array.
        image to be denoised, 2 or 3D array.
    l : float.
        regularization weight
    n : integer.
        number of iterations
    xmin : float.
        min value of x-entry, default 0
    xmax : float.
        max value of x-entry, default +inf.

    Return:
    -------
    x : numpy array
        regularized image
    """
    # Have we active box constraints on solution values?
    bc = False if (xmin, xmax) == (-float("inf"), float("inf")) else True
    k = len(b.shape)
    grad = grad2 if k == 2 else grad3
    div = div2 if k == 2 else div3
    tau = 1. / (8 * l) if k == 2 else 1.0 / (12 * l)

    s = b.shape + (k,)
    r = np.zeros(s, dtype="float32")
    pc = np.zeros(s, dtype="float32")
    pn = np.zeros(s, dtype="float32")
    tc = 1.0

    for i in range(n):
        # print i
        a = b - l * div(r)
        if bc:
            Pc(a, xmin, xmax)

        pn = r - tau * grad(a)
        project_on_field_of_balls(pn)

        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4 * tc ** 2))
        r = pn + ((tc - 1) / tn) * (pn - pc)
        tc = tn
        pc = pn

    x = b - l * div(pc)
    if bc:
        Pc(x, xmin, xmax)

    return x


def fistaTV_weighted(b, l, W, n, omega=None, xmin=0.0, xmax=float("inf")):
    """
    computes
    .. math::
        argmin_x \|x-b\|^2 + 2l|\DWx\|,\quad xmin <= x  <= xmax


    i.e. the proximal $prox_f(b)
    where f = l*\|DW.\| + i_B(.) with

    i_B(x) = +infinity if one of the components of x does not
    satisfy xmin <= x_i <= xmax, and i_B(x) = 0 otherwise,
    i.e., a identical bounds box constraint, while W is a
    diagonal matrix.

    Parameters:
    ----------
    b : numpy float32 array.
        image to be denoised, 2 or 3D array.
    l : float.
        regularization weight
    W : numpy float32 array
        weight/space deformation array.
    n : integer.
        number of iterations.
    omega : float or None, optional.
        The sup norm of W. if set to None, this is evaluated.
    xmin : float. Optional
        min value of x-entry. The default value is 0.
    xmax : float, optional.
        max value of x-entry. The default value is +INF

    Return:
    ------
    x : numpy array
        regularized image
    """
    # Have we active box constraints on solution values?
    bc = False if (xmin, xmax) == (-float("inf"), float("inf")) else True
    # omega should be the sup norm of W.*W
    if omega is None:
        omega = (W ** 2).max()

    k = len(b.shape)
    grad = grad2 if k == 2 else grad3
    div = div2 if k == 2 else div3
    tau = 1. / (8 * omega * l) if k == 2 else 1.0 / (12 * omega * l)

    s = b.shape + (k,)
    r = np.zeros(s, dtype="float32")
    pc = np.zeros(s, dtype="float32")
    pn = np.zeros(s, dtype="float32")
    tc = 1.0

    for i in range(n):
        # print i
        a = b - l * W * div(r)
        if bc:
            Pc(a, xmin, xmax)

        pn = r - tau * grad(W * a)
        project_on_field_of_balls(pn)

        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4 * tc ** 2))
        r = pn + ((tc - 1) / tn) * (pn - pc)
        tc = tn
        pc = pn

    x = b - l * W * div(pc)
    if bc:
        Pc(x, xmin, xmax)

    return x



class TVProximal:
    def __init__(self, D, DT, dim, projectC=None, vectorial=False, iterations=50):
        """Compute a (generalised) proximal for a TV norm.

        Should handle
        - different gradients and divergences, as long as they
          satisfy proper duality relations.
        - mainly interested in the cases:
            TV_l(x) = ||l(s)D u(s)||
          as well as the "other" weighted:
            TV^l(x) = ||D[l(s)x(s)]||

        The problem to sole is min ||x-b|^2 + 2 gamma ||K x||
        with K being D, or D.l or l.D

        In both cases, l will be a weight function Omega -> R_+, l(s)
        being the weight at pixel/voxel s in Omega, the signal domain,
        its codomain being the positive reals.
        When l is constant, this is the classical proximal.

        Closed convex set constraint for signal u are also possible.
        Two algorithms are proposed: A FISTA-type one and a simpler
        Gradient projection method, slower, but much less memory
        requiring fixed-point algorithm.

        This implements the algorithms proposed in Beck and Teboulle
        "Fast Gradient-Based Algorithms for Constrained Total Variation
        Image Denoising and Deblurring Problems" with slight generalisation
        regarding the operators, so that some explicit operator norms should
        be provided.
        For instance, when D is the standard discrete forward 2D gradient
        its operator square-norm is (bounded by) 8, 12 for the 3D case.
        When using D.l, or l.D, their square operator norms are bounded by
          + 8 sup_s l(s)^2, for a 2D problem
          + 12 sup_s l(s)^2 for a 3D problem.
        Actually, in the case of "outer weight", the problem is handled
        slightly differently by letting l be the radius field for the
        balls where the dual variable has to be projected onto.

        Dual variable values might be initialised to 0 at each solver call,
        or kept, which could be important for some complex iterative
        problems performances.

        One or more callback functions can be added, they should
        have the signature callback(x: numpy array, k : int) -> None.

        ----------------------------------------------------------

        The class encapsulates proximal calculations, following two
        similar algorithms. This looks rather messy to me...

        I could parametrise the dual variable projection, maybe OK,
        maybe worse?

        Parameters:
        -----------
        D : function-like
            gradient operator
        DT : function-like
            adjoint gradient a.k.a. minus divergence operator.
        dim : int
            2 for a planar problem, 3 for a 3D one.
        projectC : function-like, optional
            projection onto convex set. If None, no projection
            Pc must be an in-place projector. The default in None.
        vectorial : bool, optional
            if True, signal data is interpreted as vector-valued,
            and scalar value otherwise. The default is False.
        iterations : int, optional
            number of iterations of the algorithm. The default is 50.
        """
        self._gamma = 1.0
        self.iterations = iterations
        self.D = D
        self.DT = DT
        self.dim = dim
        self.vectorial = vectorial
        self.omega = 8 if dim == 2 else 12
        self._algorithm = 'fista'
        self.callbacks = []
        self._reset_dual_variable = True
        self.phi = None
        self.Pc  = None
        self.init_Pc(projectC)
        self.l = 1
        self.inner = None
        # Other projections could be used, to change the norm used in TV...
        self.Pp = project_on_field_of_balls if not self.vectorial else project_on_field_of_Frobenius_balls
        self.i = None
        self.x = None

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if value > 0:
            self._gamma = value
        else:
            # should add a warning?
        self._gamma = 1.0

    def set_spatial_weight_field(self, l, inner, sup_norm_l2=-1):
        """Set the weight field l(s).

        If l is "inner", i.e., we use D.l and not l.D
        An estimate of ||D.l||^2 is needed. It is easily shown that
        ||D.l||^2 \leq ||D|| 2 ||l||^2_\infinite.
        """
        self.l = l
        if isinstance(l, np.ndarray):
            self.inner = inner
            if self.inner:
                if sup_norm_l2 > 0:
                    self.omega *= sup_norm_l2
                else:
                    self.omega *= l.max()**2
        else:
            self._gamma *= l

    def init_Pc(self, func):
        """Initialise the orthogonal projection on C.

        if func is None, no projection.
        """
        if func is not None:
            self.Pc = func
        else:
            def P_all(dummy_variable):
                pass
            self.Pc = P_all

    def Pp_function(self):
        """Initialise the dual variable projection.
        """
        if not self.inner:
            def f(x):
                self.Pp(x, radius=self.l)
            return f
        else:
            def f(x):
                self.Pp(x, radius=1.0)
            return f

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, method_name):
        if method_name is not in ('fista', 'gp')
            raise ValueError(f'algorith should either be "fista" or "gp" (gradient-projection)')
        self._algorithm = method_name

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def run_callbacks(self):
        """ Run the registered callback."""
        for callback in self.callbacks:
            callback(self.x, self.i)

    @property
    def reset_dual_variable(self):
        return self._reset_dual_variable

    @reset_dual_variable.setter
    def reset_dual_variable(self, yesno):
        self._reset_dual_variable = yesno

    def create_dual_variable(self, b):
        self.phi = np.zeros(b.shape + (self.dim,), dtype=b.dtype)

    def _run_fista(self, b):
        """Run a FISTA scheme. """
        if self.reset_dual_variable or self.phi is None:
            self.create_dual_variable(b)
        psi = self.phi.copy()
        previous_phi = self.phi.copy()
        t = 1.0
        L = self.omega * self._gamma

        self.x = b.copy()

        for self.i in range(self.iterations):
            self.x[...] = b - self._gamma * self.DT(psi)
            self.Pc(self.x)

            self.run_callbacks()

            self.phi[...] = self.DT(self.x)/L + psi
            self.Pp(self.phi)

            t_next = (1 + sqrt(1 + 4*t))/2
            theta = ((t - 1)/t_next)

            psi[...] = (1 + theta) * self.phi - theta * previous_phi
            previous_phi = self.phi
            t = t_next
        self.x = b - self._gamma * self.DT(self.phi)
        self.Pc(self.x)
        return self.x

    def _run_gp(self, b):
        """Run a gradient projection scheme."""
        if self.reset_dual_variable or self.phi is None:
            self.create_dual_variable(b)

        self.x = b.copy()

        for self.i in range(self.iterations):
            self.x[...] = b - self._gamma * self.DT(self.phi)
            self.Pc(self.x)
            L = self.omega * self._gamma

            self.run_callbacks()

            self.phi[...] += self.DT(self.x) / L
            self.Pp(self.phi)

        self.x = b - self._gamma * self.DT(self.phi)
        self.Pc(self.x)
        return self.x

    def run(self, b):
        if self._algorithm == 'fista':
            return self._run_fista(b)
        return self._run_gp(b)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from skimage import data
    import matplotlib.gridspec as gridspec

    image = data.camera().astype('float')
    # image = gaussian_scalespace(image, 0.1)

    M = image.max()
    m = image.min()
    # image = (image - m)/(M-m)

    l1 = 5.
    l2 = 10.
    l3 = 50.
    x = fistaTV(image, l1, 200, xmin=float(m), xmax=float(M))
    y = fistaTV(image, l2, 200, xmin=float(m), xmax=float(M))
    z = fistaTV(image, l3, 200, xmin=float(m), xmax=float(M))

    fig, ax = plt.subplots(2, 2)
    ((axl, axx), (axy, axz)) = ax
    axl.imshow(image, cmap=cm.Greys_r)
    axl.set_title('Original')
    axx.imshow(x, cmap=cm.Greys_r)
    axx.set_title(r"$\lambda=%d$" % l1)
    axy.imshow(y, cmap=cm.Greys_r)
    axy.set_title(r"$\lambda=%d$" % l2)
    axz.imshow(z, cmap=cm.Greys_r)
    axz.set_title(r"$\lambda=%d$" % l3)

    for s in ax.flatten():
        s.axis('off')

    # Now let's do the same thing with a W image
    # I choose a disk at the center of the image
    # of radius size/sqrt(2)
    dim1, dim2 = image.shape
    x, y = np.mgrid[-1:1:dim1 * 1j, -1:1:dim2 * 1j]
    alpha, beta = 1.6, 0.4
    W = alpha * (x ** 2 + y ** 2 > 0.5).astype(float) + beta
    omega = (W ** 2).max()

    l1 = 5.
    l2 = 10.
    l3 = 50.

    x = fistaTV_weighted(image, l1, W, 200, omega=omega, xmin=float(m), xmax=float(M))
    y = fistaTV_weighted(image, l2, W, 200, omega=omega, xmin=float(m), xmax=float(M))
    z = fistaTV_weighted(image, l3, W, 200, omega=omega, xmin=float(m), xmax=float(M))

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=4, ncols=3)
    axw = fig.add_subplot(gs[1:3, 0])
    axw.imshow(W)
    axw.set_title(f"W field, out = {alpha:.01f}, in = {beta:.01f}", fontsize=24)

    axl = fig.add_subplot(gs[0:2, 1])
    axx = fig.add_subplot(gs[0:2, 2])
    axy = fig.add_subplot(gs[2:4, 1])
    axz = fig.add_subplot(gs[2:4, 2])

    axl.imshow(image, cmap=cm.Greys_r)
    axl.set_title('Original', fontsize=18)
    axx.imshow(x, cmap=cm.Greys_r)
    axx.set_title(r"$\lambda=%d$" % l1, fontsize=24)
    axy.imshow(y, cmap=cm.Greys_r)
    axy.set_title(r"$\lambda=%d$" % l2, fontsize=24)
    axz.imshow(z, cmap=cm.Greys_r)
    axz.set_title(r"$\lambda=%d$" % l3, fontsize=24)

    for s in [axw, axl, axx, axy, axz]:
        s.axis('off')

    plt.show()
