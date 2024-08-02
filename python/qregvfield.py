# -*- coding: utf-8 -*-
"""
@project: SegmentReconstruct, from PyCT (SSVM 2017)
@file: qregvfield.py

@description: A proximal solver for minimization of
          E(v) = \lambda g.v + ||Dv||_2^2, 
          v \in Sigma
where v is vector valued and Sigma is the convex product of
standard simplices: I want that for all n in domain D of v, 
                \sum_{k=1}^K vnk = 1.
The domain is a priori not rectangular and strictly included in 
my image domain. D is a discrete gradient operator, and D^TD is
a discrete negative Laplacian. Boundary conditions on the domain
are standard Neumann.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import data


__author__ = "François Lauze, University of Copenhagen"
__date__ = "09-18-2016"
__version__ = "0.0.1"


def build_basic_stencil(domain):
    """
    Build the stencil array part related to the Laplacian operator.
    
    Returns a array of stencils for each pixel in the domain, with
    reflected boundary conditions set.
    
    Parameters:
    ----------
    domain : numpy array
        a 2D binary mask.
    
    Returns:
    -------
    stencil: numpy array object
        the array describing pixel domain neighbor system
        and weights for a standard 2D discretised Laplace
        operator.
    """
    m, n = domain.shape
    ix, iy = np.where(domain)
    L = len(ix)
    stencil = np.empty(L, dtype=object)
    
    for i in range(L):
        # ordering of neighbors is west, east, south, north
        neighbors = np.array([True, True, True, True])
        coef = 4.0
        px = ix[i]
        py = iy[i]
        if (px-1 < 0) or (not domain[px-1, py]):
            neighbors[0] = False
            coef -= 1.0
            
        if (px+1>=m) or (not domain[px+1, py]):
            neighbors[1] = False
            coef -= 1.0
            
        if (py-1<0) or (not domain[px, py-1]):
            neighbors[2] = False
            coef -= 1.0
        
        if (py+1>=n) or (not domain[px, py+1]):
            neighbors[3] = False
            coef -= 1.0
            
        stencil[i] = (px, py, neighbors, coef)
    return stencil


    
def qregv(stencil, v, t, g, sor=1.0, max_iterations=100):
    """
    Compute the proximal prox_{tE}(v).
    
    This amount to solve the discretised PDE:
    .. math:: 
        -\Laplacian u + t^{-1}u = t^{-1}v - g
    
    and I use a Successive Over-Relaxation iterative solver
    (Gauss-Seidel by default with sor=1.0).
    
    Parameters:
    ----------
    stencil: numpy array object
        a list of domain pixels neighbors and weights.
    v : numpy array float
        current image / function to be regularised
    t : float 
        step/regularisation in proximal calculation
    g : numpy array
        data related quantity (be more specific!)
    sor : float
        over-relaxation parameter. Must be in (0,2]. The default
        is 1 (Gauss-Seidel relaxation).
    max_iterations : int
        maximum number of relaxation iterations. The default is
        100.
    Returns:
    -------
    u : numpy array float
        the proximal / reguarised image.
    """
    s = 1.0/t
    m,n = v.shape[:2]
    u = v.copy()
    for iterations in range(max_iterations):
        for px, py, neighbors, coef in stencil:
            numerator = s*v[px, py] - g[px, py]
            if neighbors[0]:
                numerator += u[px-1, py]
            if neighbors[1]:
                numerator += u[px+1, py]
            if neighbors[2]:
                numerator += u[px, py-1]
            if neighbors[3]:
                numerator += u[px, py+1]
            update = numerator/(coef + s)
            u[px, py] = (1-sor)*u[px, py] + sor*update
    return u


if __name__ == "__main__":
    v = data.camera().astype(float)
    
    m, n = v.shape
    vv = np.zeros((m,n,3))
    vv[:,:,0] = v
    vv[:,:,1] = v
    vv[:,:,2] = v
    x, y = np.mgrid[-m/2.0:m/2.0:m*1j, -n/2.0:n/2.0:n*1j]
    domain = x**2 + y**2 < 128**2
    
    
    t = 5000.0
    stencil = build_basic_stencil(domain)
    g = np.zeros(v.shape)
    u = qregv(stencil, vv, t, g, max_iterations=10)
    plt.imshow(u[:,:,0], cmap=cm.Greys)
    plt.show()
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
