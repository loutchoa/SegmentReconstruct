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
    argmin \|x-b\|^2 + 2*l*TV(x),\quad xmin <= x  <= xmax
    
    
    i.e. the proximal $prox_f(b)
    where f = l*TV(.) + i_B(.) with 
    
             
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



if __name__ == "__main__":
    import  matplotlib.pyplot as plt
    from matplotlib import cm
    from skimage import data
    image = data.camera().astype('float')
    image = gaussian_scalespace(image, 5.0)


    M = image.max()
    m = image.min()
    #image = (image - m)/(M-m)    

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
    plt.show()

