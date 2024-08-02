# -*- coding: utf-8 -*-
"""
@file: graddiv.py
@brief discrete gradients and divergences for segmentation algorithms

Implements a series of discrete gradients and divergences, using forward differences for the gradient,
with Neumann-like boundary conditions at the "forward" sides, and implements corresponding divergences, i.e.
discrete divergences satisfying the adjunction property:
    div* = -adj(grad*) 

"""

__author__ = "François Lauze, University of Copenhagen"
__date__ = "09-10-2014"
__version__ = "0.1.0"

# replace by bohrium when ready
import numpy as np


def grad1(f):
    """
    Discrete gradient of the scalar valued 1D function f,
    i.e. finite difference ...
    
    Parameters:
        f : array of dimensions (m,).
    Returns:
        an array of dimensions (m,) containing x derivatives.
    """    
    m = len(f)
    gradf = np.zeros((m,), dtype=type(f[0]))
    gradf[0:m-1] = f[1:m] - f[0:m-1]
    
    return gradf
    
def grad1ip(f,gf):
    """
    Discrete gradient of the scalar valued 1D function f,
    i.e. finite difference, "in-place".
    
    Parameters:
        f : array of dimensions (m,).
        gf: array of same dimension, holding the result.
    """
    gf[:-1] = f[1:] - f[:-1]
    gf[-1] = 0
    
    
    

def grad2(f):
    """
    Discrete gradient of the scalar valued 2D function f,
    f discretization of a function R^2 to R. Uses forward 
    differences.
    
    Parameters:
        f : array of dimensions (m,n).
    Returns:
        an array of dimensions (m,n,2) containing x and y 
        derivatives.
    """
    x = 0
    y = 1
    m, n = f.shape
    gradf = np.zeros((m, n, 2),dtype=f.dtype)
    
    gradf[0:m-1,:,x] = f[1:m,:] - f[0:m-1,:]
    gradf[:,0:n-1,y] = f[:,1:n] - f[:,0:n-1]
    return gradf

def grad2ip(f, gf):
    """
    Discrete gradient of the scalar valued 2D function f,
    f discretization of a function R^2 to R. Uses forward 
    differences, "in-place"
    
    Parameters:
        f : array of dimensions (m,n).
        gf: array of dimensions (m,n,2) holding the result.
    """
    x = 0
    y = 1
    m, n = f.shape
    
    gf[0:m-1,:,x] = f[1:m,:] - f[0:m-1,:]
    gf[-1,:,x] = 0.0
    gf[:,0:n-1,y] = f[:,1:n] - f[:,0:n-1]
    gf[:,-1,y] = 0
    
    

def grad3(f):
    """
    Discrete gradient of the scalar valued 3D function f,
    f discretization of a function R^3 to R.
    
    Parameters:
        f : array of dimensions (m,n,p).
    Returns:
        an array of dimensions (m,n,p,3) containing x, y 
        and z derivatives.
    """
    x = 0
    y = 1
    z = 2
    m, n, p = f.shape
    gradf = np.zeros((m, n, p, 3), dtype=f.dtype)
    
    gradf[0:m-1,:,:,x] = f[1:m,:,:] - f[0:m-1,:,:]
    gradf[:,0:n-1,:,y] = f[:,1:n,:] - f[:,0:n-1,:]
    gradf[:,:,0:p-1,z] = f[:,:,1:p] - f[:,:,0:p-1]

    return gradf


def grad3ip(f, gf):
    """
    Discrete gradient of the scalar valued 3D function f,
    f discretization of a function R^3 to R. "in-place".
    
    Parameters:
        f : array of dimensions (m,n,p).
        gf: array of dimensions (m,n,p,3).
    """
    x = 0
    y = 1
    z = 2
    m, n, p = f.shape
    
    gf[0:m-1,:,:,x] = f[1:m,:,:] - f[0:m-1,:,:]
    gf[:,0:n-1,:,y] = f[:,1:n,:] - f[:,0:n-1,:]
    gf[:,:,0:p-1,z] = f[:,:,1:p] - f[:,:,0:p-1]
    gf[-1,:,:,x] = 0
    gf[:,-1,:,y] = 0
    gf[:,:,-1,z] = 0



def grad1v(f):
    """
    Discrete gradient of the vector valued 1D function f,
    i.e. finite difference ... 
    
    Parameters:
        f : array of dimensions (m,k).
    Returns:
        an array of dimensions (m,k)
    """    
    m, k = f.shape
    gradf = np.zeros((m,k), dtype=f.dtype)
    gradf[:-1,:] = f[1:,:] - f[:-1,:]
    gradf[-1,:] = 0

    return gradf
    
    



def grad1vip(f, gf):
    """
    Discrete gradient of the vector valued 1D function f,
    i.e. finite difference ... "in-place"
    
    Parameters:
        f : array of dimensions (m,k).
        gf: array of dimensions (m,k)
    """    
    m, k = f.shape
    gf[:-1,:] = f[1:,:] - f[:-1,:]
    gf[-1,:] = 0
    
    
    
def grad2v(f):
    """
    Discrete gradient of the vector valued 2D function f,
    f discretization of a function R^2 to R^k.
    
    Parameters:
        f : array of dimensions (m,n,k).
    Returns:
        an array of dimensions (m,n,k,2) containing x and y 
        derivatives.
    """
    x = 0
    y = 1
    m, n, k = f.shape
    gradf = np.zeros((m, n, k, 2), dtype=f.dtype)

    gradf[0:m-1,:,:,x] = f[1:m,:,:] - f[0:m-1,:,:]
    gradf[:,0:n-1,:,y] = f[:,1:n,:] - f[:,0:n-1,:]
    
    return gradf



def grad2vip(f, gf):
    """
    Discrete gradient of the vector valued 2D function f,
    f discretization of a function R^2 to R^k. "in-place"
    
    Parameters:
        f : array of dimensions (m,n,k).
        gf: array of dimensions (m,n,k,2)

    """
    x = 0
    y = 1
    m, n, k = f.shape

    gf[0:m-1,:,:,x] = f[1:m,:,:] - f[0:m-1,:,:]
    gf[:,0:n-1,:,y] = f[:,1:n,:] - f[:,0:n-1,:]
    gf[-1,:,:,x] = 0
    gf[:,-1,:,y] = 0
    
    
    
    
def grad3v(f):
    """
    Discrete gradient of the vector valued 3D function f,
    f discretization of a function R^3 to R^k.
    
    Parameters:
        f : array of dimensions (m,n,p,k).
    Returns:
        an array of dimensions (m,n,p,k,3) containing x, y
        and z derivatives.
    """
    x = 0
    y = 1
    z = 2
    m, n, p, k = f.shape
    gradf = np.zeros((m, n, p, k, 3), dtype=f.dtype)
    
    gradf[0:m-1,:,:,:,x] = f[1:m,:,:,:] - f[0:m-1,:,:,:]
    gradf[:,0:n-1,:,:,y] = f[:,1:n,:,:] - f[:,0:n-1,:,:]
    gradf[:,:,0:p-1,:,z] = f[:,:,1:p,:] - f[:,:,0:p-1,:]
    
    return gradf

   
def grad3vip(f, gf):
    """
    Discrete gradient of the vector valued 3D function f,
    f discretization of a function R^3 to R^k. "in-place"
    
    Parameters:
        f : array of dimensions (m,n,p,k).
        gf: array of dimensions (m,n,p,k,3)
    """
    x = 0
    y = 1
    z = 2
    m, n, p, k = f.shape
    
    gf[0:m-1,:,:,:,x] = f[1:m,:,:,:] - f[0:m-1,:,:,:]
    gf[:,0:n-1,:,:,y] = f[:,1:n,:,:] - f[:,0:n-1,:,:]
    gf[:,:,0:p-1,:,z] = f[:,:,1:p,:] - f[:,:,0:p-1,:]
    gf[-1,:,:,:,x] = 0
    gf[:,-1,:,:,x] = 0
    gf[:,:,-1,:,x] = 0
    

def div1(f):
    """
    Discrete divergence of the 1D vector field f,
    i.e. another forward difference, so that
    div1 is the adjoint of -grad1.       
    
    Parameters:
        f : a 1D array (preferably numpy)
    Returns:
        df: a 1D array containing a backward 
        difference of f.
    """
    m = len(f)
    divf = np.zeros((m,), dtype=f.dtype)
    
    divf[0] = f[0]
    divf[1:m-1] = f[1:m-1] - f[0:m-2]
    divf[m-1] = -f[m-2]
    
    return divf


def div1ip(f, df):
    """
    Discrete divergence of the 1D vector field f,
    i.e. another forward difference, so that
    div1 is the adjoint of -grad1. "In-place"
    
    Parameters:
        f : a 1D array (preferably numpy)
        df: a 1D array containing a backward 
        difference of f.
    """
    m = len(f)
    
    df[0] = f[0]
    df[1:m-1] = f[1:m-1] - f[0:m-2]
    df[m-1] = -f[m-2]
    
    
    

def div2(vf):
    """
    Discrete gradient of the 2D vector field vf,
    vf discretization of a function R^2 to R^2.
    div2 is the adjoint of -grad2.
    .
    
    Parameters:
        vf : array of dimensions (m,n,2).
    Returns:
        an array of dimensions (m,n) containing 
        the sum of the x, and y derivatives.
    """
    x = 0
    y = 1
    m, n = vf.shape[:-1]
    divf = np.zeros((m,n), dtype=vf.dtype)
  
    divf[0,:] =  vf[0,:,x]
    divf[1:m-1,:] =  vf[1:m-1,:,x] - vf[0:m-2,:,x]
    divf[m-1,:] = -vf[m-2,:,x]
    
    divf[:,0] = divf[:,0] + vf[:,0,y]
    divf[:,1:n-1] = divf[:,1:n-1] + vf[:,1:n-1,y] - vf[:,0:n-2,y]
    divf[:,n-1] = divf[:,n-1] - vf[:,n-2,y]
    
    return divf
    
    
def div2ip(vf, df):
    """
    Discrete gradient of the 2D vector field vf,
    vf discretization of a function R^2 to R^2.
    div2 is the adjoint of -grad2. "in-place"
    .
    
    Parameters:
        vf : array of dimensions (m,n,2).
        df : array of dimensions (m,n).
    """
    x = 0
    y = 1
    m, n = vf.shape[:-1]
   
    df[0,:] =  vf[0,:,x]
    df[1:m-1,:] =  vf[1:m-1,:,x] - vf[0:m-2,:,x]
    df[m-1,:] = -vf[m-2,:,x]
    
    df[:,0] = df[:,0] + vf[:,0,y]
    df[:,1:n-1] = df[:,1:n-1] + vf[:,1:n-1,y] - vf[:,0:n-2,y]
    df[:,n-1] = df[:,n-1] - vf[:,n-2,y] 





def div3(vf):
    """
    Discrete gradient of the 3D vector field vf,
    vf discretization of a function R^3 to R^3.
    div3 is the adjoint of -grad3.
    .
    
    Parameters:
        vf : array of dimensions (m,n,p,3).
    Returns:
        an array of dimensions (m,n,p) containing 
        the sum of the x, y and z derivatives.
    """
    x = 0
    y = 1
    z = 2
    m, n, p = vf.shape[:-1]
    divf = np.zeros((m,n,p), dtype=vf.dtype)

    divf[0,:,:] = vf[0,:,:,x]
    divf[1:m-1,:,:] = vf[1:m-1,:,:,x] - vf[0:m-2,:,:,x]
    divf[m-1,:,:] = -vf[m-2,:,:,x]
    
    divf[:,0,:] = divf[:,0,:] + vf[:,0,:,y]
    divf[:,1:n-1,:] = divf[:,1:n-1,:] + vf[:,1:n-1,:,y] - vf[:,0:n-2,:,y]
    divf[:,n-1,:] = divf[:,n-1,:] - vf[:,n-2,:,y]
    
    divf[:,:,0] = divf[:,:,0] + vf[:,:,0,z]
    divf[:,:,1:p-1] = divf[:,:,1:p-1] + vf[:,:,1:p-1,z] - vf[:,:,0:p-2,z]
    divf[:,:,p-1] = divf[:,:,p-1] - vf[:,:,p-2,z]
    
    return divf


def div3ip(vf, df):
    """
    Discrete gradient of the 3D vector field vf,
    vf discretization of a function R^3 to R^3.
    div3 is the adjoint of -grad3. "in-place".
    .
    
    Parameters:
        vf: array of dimensions (m,n,p,3).
        df: array of dimensions (m,n,p)
    """
    x = 0
    y = 1
    z = 2
    m, n, p = vf.shape[:-1]
   
    df[0,:,:] = vf[0,:,:,x]
    df[1:m-1,:,:] = vf[1:m-1,:,:,x] - vf[0:m-2,:,:,x]
    df[m-1,:,:] = -vf[m-2,:,:,x]
    
    df[:,0,:] = df[:,0,:] + vf[:,0,:,y]
    df[:,1:n-1,:] = df[:,1:n-1,:] + vf[:,1:n-1,:,y] - vf[:,0:n-2,:,y]
    df[:,n-1,:] = df[:,n-1,:] - vf[:,n-2,:,y]
    
    df[:,:,0] = df[:,:,0] + vf[:,:,0,z]
    df[:,:,1:p-1] = df[:,:,1:p-1] + vf[:,:,1:p-1,z] - vf[:,:,0:p-2,z]
    df[:,:,p-1] = df[:,:,p-1] - vf[:,:,p-2,z]
    
    
    
    
def div1v(f):
    """
    Discrete divergence of the 1D "multi" vector field f,
    i.e. another forward difference, so that
    div1 is the adjoint of -grad1.       
    
    Parameters:
        f : a 1D array (preferably numpy)
    Returns:
        df: a 1D array containing a backward 
        difference of f.
    """
    m, k = f.shape
    divf = np.zeros((m,k), dtype=f.dtype)
    
    divf[0,:] = f[0,:]
    divf[1:m-1,:] = f[1:m-1,:] - f[0:m-2,:]
    divf[m-1,:] = -f[m-2,:]
    
    return divf    

    
    
        
def div1vip(f, df):
    """
    Discrete divergence of the 1D "multi" vector field f,
    i.e. another forward difference, so that
    div1 is the adjoint of -grad1. "in-place"  
    
    Parameters:
        f : a 1D array (preferably numpy)
        df: a 1D array
    """
    m, k = f.shape
  
    df[0,:] = f[0,:]
    df[1:m-1,:] = f[1:m-1,:] - f[0:m-2,:]
    df[m-1,:] = -f[m-2,:]
    
    
    
    
def div2v(vf):
    """
    Discrete gradient of the 2D "multi" vector field vf,
    vf discretization of a function R^2 to R^(2k).
    div2v is the adjoint of -grad2v.
    .
    
    Parameters:
        vf : array of dimensions (m,n,k,2).
    Returns:
        an array of dimensions (m,n,k) containing 
        the sum of the x, and y derivatives.
    """
    x = 0
    y = 1
    m, n, k = vf.shape[:-1]
    divf = np.zeros((m,n,k), dtype=vf.dtype)
 
    divf[0,:,:] =  vf[0,:,:,x]
    divf[1:m-1,:,:] =  vf[1:m-1,:,:,x] - vf[0:m-2,:,:,x]
    divf[m-1,:,:] = -vf[m-2,:,:,x]
    
    divf[:,0,:] = divf[:,0,:] + vf[:,0,:,y]
    divf[:,1:n-1,:] = divf[:,1:n-1,:] + vf[:,1:n-1,:,y] - vf[:,0:n-2,:,y]
    divf[:,n-1,:] = divf[:,n-1,:] - vf[:,n-2,:,y]
    
    return divf
    
    
    
    
def div2vip(vf, df):
    """
    Discrete gradient of the 2D "multi" vector field vf,
    vf discretization of a function R^2 to R^(2k).
    div2v is the adjoint of -grad2v. "in-place".
    
    Parameters:
        vf: array of dimensions (m,n,k,2).
        df: array of dimensions (m,n,k) containing 
        the sum of the x, and y derivatives.
    """
    x = 0
    y = 1
    m, n, k = vf.shape[:-1]
 
    df[0,:,:] =  vf[0,:,:,x]
    df[1:m-1,:,:] =  vf[1:m-1,:,:,x] - vf[0:m-2,:,:,x]
    df[m-1,:,:] = -vf[m-2,:,:,x]
    
    df[:,0,:] = df[:,0,:] + vf[:,0,:,y]
    df[:,1:n-1,:] = df[:,1:n-1,:] + vf[:,1:n-1,:,y] - vf[:,0:n-2,:,y]
    df[:,n-1,:] = df[:,n-1,:] - vf[:,n-2,:,y]    
    


def div3v(vf):
    """
    Discrete gradient of the 3D "multi" vector field vf,
    vf discretization of a function R^3 to R^(3k).
    div3v is the adjoint of -grad3v.
    .
    
    Parameters:
        vf : array of dimensions (m,n,p,k,3).
    Returns:
        an array of dimensions (m,n,p,k) containing 
        the sum of the x, y and z derivatives.
    """
    x = 0
    y = 1
    z = 2
    m, n, p, k = vf.shape[:-1]
    divf = np.zeros((m,n,p,k), dtype=vf.dtype)

    divf[0,:,:,:] = vf[0,:,:,:,x]
    divf[1:m-1,:,:,:] = vf[1:m-1,:,:,:,x] - vf[0:m-2,:,:,:,x]
    divf[m-1,:,:,:] = -vf[m-2,:,:,:,x]
    
    divf[:,0,:,:] = divf[:,0,:,:] + vf[:,0,:,:,y]
    divf[:,1:n-1,:,:] = divf[:,1:n-1,:,:] + vf[:,1:n-1,:,:,y] - vf[:,0:n-2,:,:,y]
    divf[:,n-1,:,:] = divf[:,n-1,:,:] - vf[:,n-2,:,:,y]
    
    divf[:,:,0,:] = divf[:,:,0,:] + vf[:,:,0,:,z]
    divf[:,:,1:p-1,:] = divf[:,:,1:p-1,:] + vf[:,:,1:p-1,:,z] - vf[:,:,0:p-2,:,z]
    divf[:,:,p-1,:] = divf[:,:,p-1,:] - vf[:,:,p-2,:,z]
    
    return divf
    
    
    
def div3vip(vf, df):
    """
    Discrete gradient of the 3D "multi" vector field vf,
    vf discretization of a function R^3 to R^(3k).
    div3v is the adjoint of -grad3v. "in-place"
    .
    
    Parameters:
        vf : array of dimensions (m,n,p,k,3).
        df : array of dimensions (m,n,p,k)
    """
    x = 0
    y = 1
    z = 2
    m, n, p, k = vf.shape[:-1]

    df[0,:,:,:] = vf[0,:,:,:,x]
    df[1:m-1,:,:,:] = vf[1:m-1,:,:,:,x] - vf[0:m-2,:,:,:,x]
    df[m-1,:,:,:] = -vf[m-2,:,:,:,x]
    
    df[:,0,:,:] = df[:,0,:,:] + vf[:,0,:,:,y]
    df[:,1:n-1,:,:] = df[:,1:n-1,:,:] + vf[:,1:n-1,:,:,y] - vf[:,0:n-2,:,:,y]
    df[:,n-1,:,:] = df[:,n-1,:,:] - vf[:,n-2,:,:,y]
    
    df[:,:,0,:] = df[:,:,0,:] + vf[:,:,0,:,z]
    df[:,:,1:p-1,:] = df[:,:,1:p-1,:] + vf[:,:,1:p-1,:,z] - vf[:,:,0:p-2,:,z]
    df[:,:,p-1,:] = df[:,:,p-1,:] - vf[:,:,p-2,:,z]
       
    
    

def add_ascent_grad2s(xi, v, td):
    """
    Gradient ascent for the dual variable in CCP splitting algorithm.
    2D scalar case (i.e., a 2D Chan-Vese type algorithm).

    Arguments:
    ----------
        xi: numpy array
            dual variable, dimensions should be (m,n,2). It is modified
            in place.
        v : numpy array
            label array, dimension should be (m,n)
        td: float
            gradient ascent time step
    """
    # I guess I should check dimensions!
    # But I will do it afterwards
    m, n = xi.shape[:-1]
    xi[0:m-1,:,0] += td*(v[1:m,:] - v[0:m-1,:])
    xi[:,0:n-1,1] += td*(v[:,1:n] - v[:,0:n-1])
# add_ascent_grad2s()


def add_ascent_grad2v(xi, v, td):
    """
    Gradient ascent for the dual variable in CCP splitting algorithm.
    2D vectorial case (i.e., a  2D CCP type algorithm).

    Arguments:
    ----------
        xi: numpy array
            dual variable, dimensions should be (m,n,k,2). It is modified
            in place.
        v : numpy array
            label array, dimension should be (m,n,k)
        td: float
            gradient ascent time step
    """
    # I guess I should check dimensions!
    # But I will do it afterwards
    m, n = xi.shape[:-2]
    xi[0:m-1,:,:,0] += td*(v[1:m,:,:] - v[0:m-1,:,:])
    xi[:,0:n-1,:,1] += td*(v[:,1:n,:] - v[:,0:n-1,:])
# add_ascent_grad2v()


def add_ascent_grad3s(xi, v, td):
    """
    Gradient ascent for the dual variable in CCP splitting algorithm.
    3D scalar case (i.e., a 3D Chan-Vese type algorithm).

    Arguments:
    ----------
        xi: numpy array
            dual variable, dimensions should be (m,n,p,3). It is modified
            in place.
        v : numpy array
            label array, dimension should be (m,n,p)
        td: float
            gradient ascent time step
    """
    # I guess I should check dimensions!
    # But I will do it afterwards
    m, n, p = xi.shape[:-1]
    xi[0:m-1,:,:,0] += td*(v[1:m,:,:] - v[0:m-1,:,:])
    xi[:,0:n-1,:,1] += td*(v[:,1:n,:] - v[:,0:n-1,:])
    xi[:,:,0:p-1,2] += td*(v[:,:,1:p] - v[:,:,0:p-1])
# add_ascent_grad3s()


def add_ascent_grad3v(xi, v, td):
    """
    Gradient ascent for the dual variable in CCP splitting algorithm.
    3D vectorial case (i.e., a  3D CCP type algorithm).

    Arguments:
    ----------
        xi: numpy array
            dual variable, dimensions should be (m,n,p,k,3). It is modified
            in place.
        v : numpy array
            primal variable, label array, dimension should be (m,n,p,k)
        td: float
            gradient ascent time step
    """
    # I guess I should check dimensions!
    # But I will do it afterwards
    m, n, p = xi.shape[:-2]
    
    xi[0:m-1,:,:,:,0] += td*(v[1:m,:,:,:] - v[0:m-1,:,:,:])
    xi[:,0:n-1,:,:,1] += td*(v[:,1:n,:,:] - v[:,0:n-1,:,:])
    xi[:,:,0:p-1,:,2] += td*(v[:,:,1:p,:] - v[:,:,0:p-1,:])
# add_ascent_grad3v()


def add_descent_div2s(v, xi, g, tp):
    """
    Gradient descent for the primal variable in CCP splitting algorithm.
    2D scalar case (i.e., a 2D Chan-Vese type algorithm).

    Arguments:
    ----------
      v : numpy array
          primal variable, i.e., the label array, dimension (m,n).
          Modified in-place.
      xi: numpy array
          dual variable, dimensions should be (m,n,2)
      g: numpy array:
          data term gradient array, size (m,n).
      tp: float
          gradient descent time step
    """
    # the 4 corners
    m,n = v.shape
    v[0,0]   += tp*( xi[0  ,0  ,0] + xi[0  ,0  ,1] - g[0  ,0  ])
    v[m-1,0] += tp*(-xi[m-2,0  ,0] + xi[m-1,0  ,1] - g[m-1,0  ])
    v[0,n-1] += tp*( xi[0,n-1  ,0] - xi[0  ,n-2,1] - g[0  ,n-1])
    v[-1,-1] += tp*(-xi[m-2,n-1,0] - xi[m-1,n-2,1] - g[m-1,n-1])

    # the 4 edges
    v[1:m-1,0]   += tp*(xi[1:m-1, 0, 0] - xi[0:m-2, 0, 0] + xi[1:m-1, 0, 1] - g[1:m-1,  0])
    v[1:m-1,n-1] += tp*(xi[1:m-1,n-1,0] - xi[0:m-2,n-1,0] - xi[1:m-1,n-2,1] - g[1:m-1,n-1])
    v[0,1:n-1]   += tp*( xi[0, 1:n-1, 0] + xi[0 ,1:n-1, 1] - xi[0,  0:n-2, 1] - g[0,  1:n-1])
    v[m-1,1:n-1] += tp*(-xi[m-2,1:n-1,0] + xi[m-1,1:n-1,1] - xi[m-1,0:n-2, 1] - g[m-1,1:n-1])

    # interior
    v[1:m-1,1:n-1] += tp*(xi[1:m-1,1:n-1,0] - xi[0:m-2,1:n-1,0] + xi[1:m-1,1:n-1,1] - xi[1:m-1,0:n-2,1] - g[1:m-1,1:n-1])
# add_descent_div2s()


def add_descent_div2v(v, xi, g, tp):
    """
    Gradient descent for the primal variable in CCP splitting algorithm.
    2D vector case (i.e., a 2D CCP type type algorithm).

    Arguments:
    ----------
      v : numpy array
          primal variable, i.e., the label array, dimension (m,n,k).
          Modified in-place.
      xi: numpy array
          dual variable, dimensions should be (m,n,k,2)
      g: numpy array:
          data term gradient array, size (m,n,k).
      tp: float
          gradient descent time step
    """
    # the 4 corners
    m,n,k = v.shape
    v[0,0,:]   += tp*( xi[0  ,0  ,:, 0] + xi[0  ,0  ,:, 1] - g[0  ,0  ,:])
    v[m-1,0,:] += tp*(-xi[m-2,0  ,:, 0] + xi[m-1,0  ,:, 1] - g[m-1,0  ,:])
    v[0,n-1,:] += tp*( xi[0,n-1  ,:, 0] - xi[0  ,n-2,:, 1] - g[0  ,n-1,:])
    v[-1,-1,:] += tp*(-xi[m-2,n-1,:, 0] - xi[m-1,n-2,:, 1] - g[m-1,n-1,:])

    # the 4 edges
    v[1:m-1,0,:]   += tp*(xi[1:m-1, 0, :, 0] - xi[0:m-2, 0, :, 0] + xi[1:m-1, 0, :, 1] - g[1:m-1,  0 ,:])
    v[1:m-1,n-1,:] += tp*(xi[1:m-1,n-1,:, 0] - xi[0:m-2,n-1,:, 0] - xi[1:m-1,n-2,:, 1] - g[1:m-1,n-1 ,:])
    v[0,1:n-1,:]   += tp*( xi[0, 1:n-1,:,  0] + xi[0 ,1:n-1, :, 1] - xi[0,  0:n-2, :, 1] - g[0,  1:n-1 ,:])
    v[m-1,1:n-1,:] += tp*(-xi[m-2,1:n-1,:, 0] + xi[m-1,1:n-1,:, 1] - xi[m-1,0:n-2, :, 1] - g[m-1,1:n-1 ,:])

    # interior
    v[1:m-1,1:n-1, :] += tp*(xi[1:m-1,1:n-1,:,0] - xi[0:m-2,1:n-1,:,0] + xi[1:m-1,1:n-1,:,1] - xi[1:m-1,0:n-2,:,1] - g[1:m-1,1:n-1,:])
# add_descent_div2v()


def add_descent_div3s(v, xi, g, tp):
    """
    Gradient descent for the primal variable in CCP splitting algorithm.
    3D scalar case (i.e., a 3D Chan-Vese type type algorithm).

    Arguments:
    ----------
      v : numpy array
          primal variable, i.e., the label array, dimension (m,n,p).
          Modified in-place.
      xi: numpy array
          dual variable, dimensions should be (m,n,p,3)
      g: numpy array:
          data term gradient array, size (m,n,p).
      tp: float
          gradient descent time step
    """
    m,n,p = v.shape
    # the 8 corners
    v[  0,  0,  0] += tp*( xi[  0,  0,  0,0] + xi[  0,  0,  0,1] + xi[  0,  0,  0,2] - g[  0,  0,  0])
    v[  0,  0,p-1] += tp*( xi[  0,  0,p-1,0] + xi[  0,  0,p-1,1] - xi[  0,  0,p-2,2] - g[  0,  0,p-1])
    v[  0,n-1,  0] += tp*( xi[  0,n-1,  0,0] - xi[  0,n-2,  0,1] + xi[  0,n-1,  0,2] - g[  0,n-1,  0])
    v[m-1,  0,  0] += tp*(-xi[m-2,  0,  0,0] + xi[m-1,  0,  0,1] + xi[m-1,  0,  0,2] - g[m-1,  0,  0])
    v[  0,n-1,p-1] += tp*( xi[  0,n-1,p-1,0] - xi[  0,n-2,p-1,1] - xi[  0,n-1,p-2,2] - g[  0,n-1,p-1])
    v[m-1,  0,p-1] += tp*(-xi[m-2,  0,p-1,0] + xi[m-1,  0,p-1,1] - xi[m-1,  0,p-2,2] - g[m-1,  0,p-1])
    v[m-1,n-1,  0] += tp*(-xi[m-2,n-1,  0,0] - xi[m-1,n-2,  0,1] + xi[m-1,n-1,  0,2] - g[m-1,n-1,  0])
    v[m-1,n-1,p-1] += tp*(-xi[m-2,n-1,p-1,0] - xi[m-1,n-2,p-1,1] - xi[m-1,n-1,p-2,2] - g[m-1,n-1,p-1])

    # the 12 edges
    v[1:m-1,  0,  0] += tp*(xi[1:m-1,  0,  0,0] - xi[0:m-2,  0,  0,0] + xi[1:m-1,  0,  0,1] + xi[1:m-1,  0,  0,2] - g[1:m-1,  0,  0]);
    v[1:m-1,  0,p-1] += tp*(xi[1:m-1,  0,p-1,0] - xi[0:m-2,  0,p-1,0] + xi[1:m-1,  0,p-1,1] - xi[1:m-1,  0,p-2,2] - g[1:m-1,  0,p-1]);
    v[1:m-1,n-1,  0] += tp*(xi[1:m-1,n-1,  0,0] - xi[0:m-2,n-1,  0,0] - xi[1:m-1,n-2,  0,1] + xi[1:m-1,n-1,  0,2] - g[1:m-1,n-1,  0]);
    v[1:m-1,n-1,p-1] += tp*(xi[1:m-1,n-1,p-1,0] - xi[0:m-2,n-1,p-1,0] - xi[1:m-1,n-2,p-1,1] - xi[1:m-1,n-1,p-2,2] - g[1:m-1,n-1,p-1]);

    v[  0,1:n-1,  0] += tp*( xi[  0,1:n-1,  0,0] + xi[  0,1:n-1,  0,1] - xi[  0,0:n-2,  0,1] + xi[  0,1:n-1,  0,2] - g[  0,1:n-1,  0]);
    v[  0,1:n-1,p-1] += tp*( xi[  0,1:n-1,p-1,0] + xi[  0,1:n-1,p-1,1] - xi[  0,0:n-2,p-1,1] - xi[  0,1:n-1,p-2,2] - g[  0,1:n-1,p-1]);
    v[m-1,1:n-1,  0] += tp*(-xi[m-2,1:n-1,  0,0] + xi[m-1,1:n-1,  0,1] - xi[m-1,0:n-2,  0,1] + xi[m-1,1:n-1,  0,2] - g[m-1,1:n-1,  0]);
    v[m-1,1:n-1,p-1] += tp*(-xi[m-2,1:n-1,p-1,0] + xi[m-1,1:n-1,p-1,1] - xi[m-1,0:n-2,p-1,1] - xi[m-1,1:n-1,p-2,2] - g[m-1,1:n-1,p-1]);

    v[  0,  0,1:p-1] += tp*( xi[  0,  0,1:p-1,0] + xi[  0,  0,1:p-1,1] + xi[  0,  0,1:p-1,2] - xi[  0,  0,0:p-2,2] - g[  0,  0,1:p-1]);
    v[  0,n-1,1:p-1] += tp*( xi[  0,n-1,1:p-1,0] - xi[  0,n-2,1:p-1,1] + xi[  0,n-1,1:p-1,2] - xi[  0,n-1,0:p-2,2] - g[  0,n-1,1:p-1]);
    v[m-1,  0,1:p-1] += tp*(-xi[m-2,  0,1:p-1,0] + xi[m-1,  0,1:p-1,1] + xi[m-1,  0,1:p-1,2] - xi[m-1,  0,0:p-2,2] - g[m-1,  0,1:p-1]);
    v[m-1,n-1,1:p-1] += tp*(-xi[m-2,n-1,1:p-1,0] - xi[m-1,n-2,1:p-1,1] + xi[m-1,n-1,1:p-1,2] - xi[m-1,n-1,0:p-2,2] - g[m-1,n-1,1:p-1]);

    # the 6 faces
    v[1:m-1,1:n-1,0]   += tp*(xi[1:m-1,1:n-1,  0,0] - xi[0:m-2,1:n-1,  0,0] + xi[1:m-1,1:n-1,  0,1] - xi[1:m-1,0:n-2,  0,1] + xi[1:m-1,1:n-1,  0,2] - g[1:m-1,1:n-1,  0]);
    v[1:m-1,1:n-1,p-1] += tp*(xi[1:m-1,1:n-1,p-1,0] - xi[0:m-2,1:n-1,p-1,0] + xi[1:m-1,1:n-1,p-1,1] - xi[1:m-1,0:n-2,p-1,1] - xi[1:m-1,1:n-1,p-2,2] - g[1:m-1,1:n-1,p-1]);

    v[1:m-1,  0,1:p-1] += tp*(xi[1:m-1,  0,1:p-1,0] - xi[0:m-2,  0,1:p-1,0] + xi[1:m-1,  0,1:p-1,1] + xi[1:m-1,  0,1:p-1,2] - xi[1:m-1,  0,0:p-2,2] - g[1:m-1, 0,1:p-1]);
    v[1:m-1,n-1,1:p-1] += tp*(xi[1:m-1,n-1,1:p-1,0] - xi[0:m-2,n-1,1:p-1,0] - xi[1:m-1,n-2,1:p-1,1] + xi[1:m-1,n-1,1:p-1,2] - xi[1:m-1,n-1,0:p-2,2] - g[1:m-1,n-1,1:p-1]);

    v[  0,1:n-1,1:p-1] += tp*( xi[  0,1:n-1,1:p-1,0] + xi[  0,1:n-1,1:p-1,1] - xi[  0,0:n-2,1:p-1,1] + xi[  0,1:n-1,1:p-1,2] - xi[  0,1:n-1,0:p-2,2] - g[  0,1:n-1,1:p-1]);
    v[m-1,1:n-1,1:p-1] += tp*(-xi[m-2,1:n-1,1:p-1,0] + xi[m-1,1:n-1,1:p-1,1] - xi[m-1,0:n-2,1:p-1,1] + xi[m-1,1:n-1,1:p-1,2] - xi[m-1,1:n-1,0:p-2,2] - g[m-1,1:n-1,1:p-1]);


    # the interior
    v[1:m-1, 1:n-1, 1:p-1] += tp*(xi[1:m-1,1:n-1,1:p-1,0] - xi[0:m-2,1:n-1,1:p-1,0] +
                                  xi[1:m-1,1:n-1,1:p-1,1] - xi[1:m-1,0:n-2,1:p-1,1] +
                                  xi[1:m-1,1:n-1,1:p-1,2] - xi[1:m-1,1:n-1,0:p-2,2] - g[1:m-1,1:n-1,1:p-1])
# add_descent_div3s


def add_descent_div3v(v, xi, g, tp):
    """
    Gradient descent for the primal variable in CCP splitting algorithm.
    3D vector case (i.e., a 3D CCP type type algorithm).

    Arguments:
    ----------
      v : numpy array
          primal variable, i.e., the label array, dimension (m,n,p,k).
          Modified in-place.
      xi: numpy array
          dual variable, dimensions should be (m,n,p,k,3)
      g: numpy array:
          data term gradient array, size (m,n,p,k).
      tp: float
          gradient descent time step
    """
    m,n,p = v.shape[:-1]
    # the 8 corners
    v[  0,  0,  0, :] += tp*( xi[  0,  0,  0, :, 0] + xi[  0,  0,  0, :, 1] + xi[  0,  0,  0, :, 2] - g[  0,  0,  0, :])
    v[  0,  0,p-1, :] += tp*( xi[  0,  0,p-1, :, 0] + xi[  0,  0,p-1, :, 1] - xi[  0,  0,p-2, :, 2] - g[  0,  0,p-1, :])
    v[  0,n-1,  0, :] += tp*( xi[  0,n-1,  0, :, 0] - xi[  0,n-2,  0, :, 1] + xi[  0,n-1,  0, :, 2] - g[  0,n-1,  0, :])
    v[m-1,  0,  0, :] += tp*(-xi[m-2,  0,  0, :, 0] + xi[m-1,  0,  0, :, 1] + xi[m-1,  0,  0, :, 2] - g[m-1,  0,  0, :])
    v[  0,n-1,p-1, :] += tp*( xi[  0,n-1,p-1, :, 0] - xi[  0,n-2,p-1, :, 1] - xi[  0,n-1,p-2, :, 2] - g[  0,n-1,p-1, :])
    v[m-1,  0,p-1, :] += tp*(-xi[m-2,  0,p-1, :, 0] + xi[m-1,  0,p-1, :, 1] - xi[m-1,  0,p-2, :, 2] - g[m-1,  0,p-1, :])
    v[m-1,n-1,  0, :] += tp*(-xi[m-2,n-1,  0, :, 0] - xi[m-1,n-2,  0, :, 1] + xi[m-1,n-1,  0, :, 2] - g[m-1,n-1,  0, :])
    v[m-1,n-1,p-1, :] += tp*(-xi[m-2,n-1,p-1, :, 0] - xi[m-1,n-2,p-1, :, 1] - xi[m-1,n-1,p-2, :, 2] - g[m-1,n-1,p-1, :])

    # the 12 edges
    v[1:m-1,  0,  0, :] += tp*( xi[1:m-1,  0,  0, :, 0] - xi[0:m-2,  0,  0, :, 0] + xi[1:m-1,  0,  0, :, 1] + xi[1:m-1,  0,  0, :, 2] - g[1:m-1,  0,  0, :]);
    v[1:m-1,  0,p-1, :] += tp*( xi[1:m-1,  0,p-1, :, 0] - xi[0:m-2,  0,p-1, :, 0] + xi[1:m-1,  0,p-1, :, 1] - xi[1:m-1,  0,p-2, :, 2] - g[1:m-1,  0,p-1, :]);
    v[1:m-1,n-1,  0, :] += tp*( xi[1:m-1,n-1,  0, :, 0] - xi[0:m-2,n-1,  0, :, 0] - xi[1:m-1,n-2,  0, :, 1] + xi[1:m-1,n-1,  0, :, 2] - g[1:m-1,n-1,  0, :]);
    v[1:m-1,n-1,p-1, :] += tp*( xi[1:m-1,n-1,p-1, :, 0] - xi[0:m-2,n-1,p-1, :, 0] - xi[1:m-1,n-2,p-1, :, 1] - xi[1:m-1,n-1,p-2, :, 2] - g[1:m-1,n-1,p-1, :]);

    v[  0,1:n-1,  0, :] += tp*( xi[  0,1:n-1,  0, :, 0] + xi[  0,1:n-1,  0, :, 1] - xi[  0,0:n-2,  0, :, 1] + xi[  0,1:n-1,  0, :, 2] - g[  0,1:n-1,  0, :]);
    v[  0,1:n-1,p-1, :] += tp*( xi[  0,1:n-1,p-1, :, 0] + xi[  0,1:n-1,p-1, :, 1] - xi[  0,0:n-2,p-1, :, 1] - xi[  0,1:n-1,p-2, :, 2] - g[  0,1:n-1,p-1, :]);
    v[m-1,1:n-1,  0, :] += tp*(-xi[m-2,1:n-1,  0, :, 0] + xi[m-1,1:n-1,  0, :, 1] - xi[m-1,0:n-2,  0, :, 1] + xi[m-1,1:n-1,  0, :, 2] - g[m-1,1:n-1,  0, :]);
    v[m-1,1:n-1,p-1, :] += tp*(-xi[m-2,1:n-1,p-1, :, 0] + xi[m-1,1:n-1,p-1, :, 1] - xi[m-1,0:n-2,p-1, :, 1] - xi[m-1,1:n-1,p-2, :, 2] - g[m-1,1:n-1,p-1, :]);

    v[  0,  0,1:p-1, :] += tp*( xi[  0,  0,1:p-1, :, 0] + xi[  0,  0,1:p-1, :, 1] + xi[  0,  0,1:p-1, :, 2] - xi[  0,  0,0:p-2, :, 2] - g[  0,  0,1:p-1, :]);
    v[  0,n-1,1:p-1, :] += tp*( xi[  0,n-1,1:p-1, :, 0] - xi[  0,n-2,1:p-1, :, 1] + xi[  0,n-1,1:p-1, :, 2] - xi[  0,n-1,0:p-2, :, 2] - g[  0,n-1,1:p-1, :]);
    v[m-1,  0,1:p-1, :] += tp*(-xi[m-2,  0,1:p-1, :, 0] + xi[m-1,  0,1:p-1, :, 1] + xi[m-1,  0,1:p-1, :, 2] - xi[m-1,  0,0:p-2, :, 2] - g[m-1,  0,1:p-1, :]);
    v[m-1,n-1,1:p-1, :] += tp*(-xi[m-2,n-1,1:p-1, :, 0] - xi[m-1,n-2,1:p-1, :, 1] + xi[m-1,n-1,1:p-1, :, 2] - xi[m-1,n-1,0:p-2, :, 2] - g[m-1,n-1,1:p-1, :]);

    # the 6 faces
    v[1:m-1,1:n-1,  0, :] += tp*( xi[1:m-1,1:n-1,  0, :, 0] - xi[0:m-2,1:n-1,  0, :, 0] + xi[1:m-1,1:n-1,  0, :, 1] - xi[1:m-1,0:n-2,  0, :, 1] + xi[1:m-1,1:n-1,  0, :, 2] - g[1:m-1,1:n-1,  0, :]);
    v[1:m-1,1:n-1,p-1, :] += tp*( xi[1:m-1,1:n-1,p-1, :, 0] - xi[0:m-2,1:n-1,p-1, :, 0] + xi[1:m-1,1:n-1,p-1, :, 1] - xi[1:m-1,0:n-2,p-1, :, 1] - xi[1:m-1,1:n-1,p-2, :, 2] - g[1:m-1,1:n-1,p-1, :]);

    v[1:m-1,  0,1:p-1, :] += tp*( xi[1:m-1,  0,1:p-1, :, 0] - xi[0:m-2,  0,1:p-1, :, 0] + xi[1:m-1,  0,1:p-1, :, 1] + xi[1:m-1,  0,1:p-1, :, 2] - xi[1:m-1,  0,0:p-2, :, 2] - g[1:m-1,  0,1:p-1, :]);
    v[1:m-1,n-1,1:p-1, :] += tp*( xi[1:m-1,n-1,1:p-1, :, 0] - xi[0:m-2,n-1,1:p-1, :, 0] - xi[1:m-1,n-2,1:p-1, :, 1] + xi[1:m-1,n-1,1:p-1, :, 2] - xi[1:m-1,n-1,0:p-2, :, 2] - g[1:m-1,n-1,1:p-1, :]);

    v[  0,1:n-1,1:p-1, :] += tp*( xi[  0,1:n-1,1:p-1, :, 0] + xi[  0,1:n-1,1:p-1, :, 1] - xi[  0,0:n-2,1:p-1, :, 1] + xi[  0,1:n-1,1:p-1, :, 2] - xi[  0,1:n-1,0:p-2, :, 2] - g[  0,1:n-1,1:p-1, :]);
    v[m-1,1:n-1,1:p-1, :] += tp*(-xi[m-2,1:n-1,1:p-1, :, 0] + xi[m-1,1:n-1,1:p-1, :, 1] - xi[m-1,0:n-2,1:p-1, :, 1] + xi[m-1,1:n-1,1:p-1, :, 2] - xi[m-1,1:n-1,0:p-2, :, 2] - g[m-1,1:n-1,1:p-1, :]);


    # the interior
    v[1:m-1, 1:n-1, 1:p-1, :] += tp*(xi[1:m-1,1:n-1,1:p-1, :, 0] - xi[0:m-2,1:n-1,1:p-1, :, 0] +
                                     xi[1:m-1,1:n-1,1:p-1, :, 1] - xi[1:m-1,0:n-2,1:p-1, :, 1] +
                                     xi[1:m-1,1:n-1,1:p-1, :, 2] - xi[1:m-1,1:n-1,0:p-2, :, 2] - g[1:m-1,1:n-1,1:p-1, :])
# add_descent_div3v
    
    
    
    
    
    
     
    
def adj_diff(f, g, gf, dg):
    ff = f.flatten()
    fg = g.flatten()
    fgf = gf.flatten()
    fdg = dg.flatten()
    
    return (ff*fdg).sum() + (fgf*fg).sum()
  
    
if __name__ == "__main__":
    from numpy.random import randint
    
    # 1D case, scalar valued
    print("Check adjunction, 1D scalar case")
    f = randint(-5, high=5, size=(3,))
    g = randint(-5, high=5, size=(3,))
    gf = grad1(f)
    dg = div1(g)
    print(adj_diff(f, g, gf, dg))
    print("")

    # 1D case, vector valued
    print("Check adjunction, 1D vector case")
    f = randint(-5, high=5, size=(3,3))
    g = randint(-5, high=5, size=(3,3))
    gf = grad1v(f)
    dg = div1v(g)
    print(adj_diff(f, g, gf, dg))
    print("")        
    
    # 2D case, scalar valued
    print("Check adjunction, 2D scalar case")
    f = randint(-5, high=5, size=(2,3))
    g = randint(-5, high=5, size=(2,3,2))
    gf = grad2(f)
    dg = div2(g)
    print(adj_diff(f, g, gf, dg))
    print("")

    # 2D case, vector valued
    print("Check adjunction, 2D vector case")
    f = randint(-5, high=5, size=(2,3,3))
    g = randint(-5, high=5, size=(2,3,3,2))
    gf = grad2v(f)
    dg = div2v(g)
    print(adj_diff(f, g, gf, dg))
    print("")
    
    
    # 3D case, scalar valued
    print("Check adjunction, 3D scalar case")
    f = randint(-5, high=5, size=(2,3,4))
    g = randint(-5, high=5, size=(2,3,4,3))
    gf = grad3(f)
    dg = div3(g)
    print(adj_diff(f, g, gf, dg))
    print("")

    # 3D case, vector valued
    print("Check adjunction, 3D vector case")
    f = randint(-5, high=5, size=(2,3,3,4))
    g = randint(-5, high=5, size=(2,3,3,4,3))
    gf = grad3v(f)
    dg = div3v(g)
    print(adj_diff(f, g, gf, dg))
    print("")
    