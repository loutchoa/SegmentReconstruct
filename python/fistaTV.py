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
from scalespace import gaussian_scalespace

def Pc(x, xmin, xmax):
    """
    Pc: simple box constraint projection
    """
    np.place(x, x < xmin, xmin)
    np.place(x, x > xmax, xmax)


def Pp(psi, l=1.0):
    """
    project a vector field on the field of radius l balls,
    i.e. each vector v of the field is projected in a radius l ball. If 
    |v| <= l, nothing to do, if |v| > l, normalise v so that its norm is 
    actually l.
    
    :param psi: numpy array.
        the vector field to be projected
    :param lambda: float. Then radius of each ball.
    :return: psi projected
    """

    psishape = psi.shape
    vdim = psishape[-1]
    rdim = reduce(mul, psishape[:-1], 1)
    psi.shape = (rdim, vdim)

    psinorm = np.sqrt(np.sum(psi**2, axis=1))
    psinorm = (psinorm/l)*(psinorm > l) + np.ones_like(psinorm)*(psinorm <= l)
    #np.place(psinorm, psinorm <= 1.0, 1.0)
    for i in range(vdim):
        psi[:,i] /= psinorm
    psi.shape = psishape
    return psi





def fistaTV(b, l, n, xmin=0.0, xmax = float("inf")):
    """
    computes
    .. math::
    argmin_x \|x-b\|^2 + 2l\|Dx\|,\quad xmin <= x  <= xmax
    
    
    i.e. the proximal $prox_f(b)
    where f = l*\|D.\| + i_B(.) with
    
             
    i_B(x) = +infty if one of the components of $x$ does not
    satisfy xmin <= x_i <= xmax, and i_B(x) = 0 otherwise,
    i.e., a identical bounds box constraint
    
    :param b: numpy float32 array.
        image to be denoised, 2 or 3D array.
    :param l: float.
        regularization weight
    :param n: integer.
        number of iterations
    xmin: float.
        min value of x-entry, default 0
    xmax: float.
        max value of x-entry, default +infty.
        
    :return x: regularized image
    """
    # Have we active box constraints on solution values?
    bc = False if (xmin, xmax) == (-float("inf"), float("inf")) else True
    k = len(b.shape)
    grad = grad2 if k == 2 else grad3
    div = div2 if k == 2 else div3
    tau = 1./(8*l) if k == 2 else 1.0/(12*l)

    s = b.shape + (k,)
    r = np.zeros(s, dtype="float32")
    pc = np.zeros(s, dtype="float32")
    pn = np.zeros(s, dtype="float32")
    tc = 1.0

    for i in range(n):
        # print i
        a = b - l*div(r)
        if bc:
            Pc(a, xmin, xmax)

        pn = r - tau*grad(a)
        Pp(pn)

        tn = 0.5*(1.0 + np.sqrt(1.0 + 4*tc**2))
        r = pn + ((tc-1)/tn)*(pn - pc)
        tc = tn
        pc = pn

    x = b - l*div(pc)
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

    i_B(x) = +infty if one of the components of $x$ does not
    satisfy xmin <= x_i <= xmax, and i_B(x) = 0 otherwise,
    i.e., a identical bounds box constraint, while W is a
    diagonal matrix.

    :param b: numpy float32 array.
        image to be denoised, 2 or 3D array.
    :param l: float.
        regularization weight
    :param W: numpy float32 array
        weight/space deformation array.
    :param n: integer.
        number of iterations.
    :param omega: float or None, optional.
        The sup norm of W. if set to None, this is evaluated.
    xmin: float. Optional
        min value of x-entry. The default value is 0.
    xmax: float, optional.
        max value of x-entry. The default value is +INF

    :return x: regularized image
    """
    # Have we active box constraints on solution values?
    bc = False if (xmin, xmax) == (-float("inf"), float("inf")) else True
    # omega should be the sup norm of W.*W
    if omega is None:
        omega = (W**2).max()

    k = len(b.shape)
    grad = grad2 if k == 2 else grad3
    div = div2 if k == 2 else div3
    tau = 1./(8*omega*l) if k == 2 else 1.0/(12*omega*l)

    s = b.shape + (k,)
    r = np.zeros(s, dtype="float32")
    pc = np.zeros(s, dtype="float32")
    pn = np.zeros(s, dtype="float32")
    tc = 1.0

    for i in range(n):
        # print i
        a = b - l*W*div(r)
        if bc:
            Pc(a, xmin, xmax)

        pn = r - tau*grad(W*a)
        Pp(pn)

        tn = 0.5*(1.0 + np.sqrt(1.0 + 4*tc**2))
        r = pn + ((tc-1)/tn)*(pn - pc)
        tc = tn
        pc = pn

    x = b - l*W*div(pc)
    if bc:
        Pc(x, xmin, xmax)

    return x


if __name__ == "__main__":
    import  matplotlib.pyplot as plt
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

    fig, ax = plt.subplots(2,2)
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
    x, y = np.mgrid[-1:1:dim1*1j, -1:1:dim2*1j]
    alpha, beta = 1.6, 0.4
    W = alpha * (x ** 2 + y ** 2 > 0.5).astype(float) + beta
    omega = (W**2).max()

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

