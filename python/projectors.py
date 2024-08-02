#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: CodeFromFrancois - SegmentReconstruct
@filename: projectors.py
    
@description: A module containing some projections and proximals
This is a pure Python reimplementation of my old C++ code.
"""
from functools import reduce
from operator import mul
import numpy as np

__author__ = "François Lauze, University of Copenhagen"
__date__ = "2021-11-26"
__version__ = "0.0.1"


def product_dimensions(a):
    return reduce(mul, a, 1)


def partial_dimensions(a, k):
    """
    compute some partial flatten dimensionality for the k first dims of a 
    """
    return reduce(mul, a.shape[:k], 1)


def project_on_field_of_balls(phi, radius=1.0):
    """
   Project a vector field on balls field of given radius.
   if radius is an array, it should have the same non vectorial
   dimensions as phi, otherwise it will crash. phi is modified in-place
   
   Parameters:
   -----------
   phi: float ndarray
       A dual variable to be projected. dim should be (basedim + K)
   radius: float or float ndarray, optional
       Radius of th balls in the field
       
   Returns:
   --------
       phi (also modified in-place)
   """
    base_shape = phi.shape[:-1]
    base_size = partial_dimensions(phi, -1)
    K = phi.shape[-1]

    if type(radius) is np.ndarray:
        radius_shape = radius.shape
        radius.shape = radius.size

    phi.shape = (base_size, K)
    norm_phi = np.linalg.norm(phi, axis=1)
    np.place(norm_phi, norm_phi <= radius, 1.0)
    np.copyto(norm_phi, radius / norm_phi, where=norm_phi > radius)
    norm_phi.shape = (-1, 1)
    phi *= norm_phi
    phi.shape = base_shape + (K,)
    if type(radius) is np.ndarray:
        radius.shape = radius_shape


def project_on_field_of_Frobenius_balls(phi, radius=1.0):
    """
   Project a vector field on matrix balls field of given radius.
   if radius is an array, it should have the same non matricial
   dimensions as phi, otherwise it will crash. phi is modified in-place
   
   Parameters:
   -----------
   phi: float ndarray
       A dual variable to be projected. dim should be (basedim + (K, n))
       n is expected to be 2 or 3, but it does not really matters....
   radius: float or float ndarray, optional
       Radius of th balls in the field
       
   Returns:
   --------
       phi (also modified in-place)
   """
    base_shape = phi.shape[:-2]
    matrix_shape = phi.shape[-2:]
    base_size = product_dimensions(base_shape)
    matrix_size = product_dimensions(matrix_shape)

    if type(radius) is np.ndarray:
        radius_shape = radius.shape
        radius.shape = radius.size

    phi.shape = (base_size, matrix_size)
    nphi = np.linalg.norm(phi, axis=1)
    np.place(nphi, nphi <= radius, 1.0)
    np.copyto(nphi, radius / nphi, where=nphi > radius)
    nphi.shape = (-1, 1)
    phi *= nphi
    phi.shape = base_shape + matrix_shape
    if type(radius) is np.ndarray:
        radius.shape = radius_shape


def project_on_simplex_field(y):
    """
    classical projection into standard simplex algorithm.
    y is a base_dim + (K,) array and each K-vector represents a sample
    to be projected. y is modified in-place

    Parameters:
    ---------
    y: numpy array
        field to be projected, size (m1, m2, K) or (m1, m2, m3, K) where K
        is the simplex embedding dimension.
    Returns:
    --------
    z : numpy array:
        projected field, same size as y.
    """
    base_shape = y.shape[:-1]
    base_size = partial_dimensions(y, -1)
    K = y.shape[-1]
    y.shape = (base_size, K)
    z = np.sort(y, axis=1)
    z = (np.cumsum(z[:, ::-1], axis=1) - 1) / np.arange(1, K + 1)
    z = np.max(z, axis=1).reshape(base_size, 1)
    # ellipsis notation ensures that y is overwritten!
    y[...] = np.maximum(y - z, 0)
    y.shape = base_shape + (K,)


def project_on_simplex_vertex(v):
    """
    Project in-place each simplex entry to the closest vertex.

    Parameters:
    ----------
    v: numpy array
        field of standard simplex valued vectors
    Returns:
    -------
    None
    """
    v_shape = v.shape
    base_dim, k = v.shape[:-1], v.shape[-1]
    b_dim = product_dimensions(base_dim)
    v.shape = (b_dim, k)
    Ik = np.eye(k)
    # ellipsis notation ensures that v is overwritten!
    v[...] = Ik[np.argmax(v, axis=1)]
    v.shape = v_shape

