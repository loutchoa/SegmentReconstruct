#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: Segmentation
@filename: label_simplex_fields.py
    
@description:
    transform a label field into a binary simplex
    field. 
    Transform a simplex valued field into a label field
    via thresholding/projection onto vertices

This is a pure Python reimplementation of my old C++ code.
This should be moved into some simplex/segmentation package
"""

from functools import reduce
from operator import mul
import numpy as np

__author__ = "François Lauze, University of Copenhagen"
__date__ = "2021-11-16"
__version__ = "0.0.1"


def label_to_simplex_field(u, dtype='float32'):
    """
    Convert a integer valued label field into a simplex valued field.
    :param u: ndarray int
        input field, dimension dim. Assume that labels are integers 
        0 <= l <= K_1
    :return: ndarray float
        output field, dimension (dim, K)
    """
    
    K = u.max() + 1
    idK = np.eye(K, dtype=dtype)
    return idK[u]


def simplex_to_label_field(v):
    """
    Project/convert a simplex valued field to an integer one.
    :param v: ndarray of floats
        a field of dimension base_dim + (K,), supposed to be simplex valued
    :return: ndarray of ints
        output field, of dimensions base_dim, with values in {0..K-1}
    """
    
    return np.argmax(v, axis=-1)


if __name__ == '__main__':
    u = np.random.randint(0,5,size=(2,8))
    v = label_to_simplex_field(u)
    w = simplex_to_label_field(v)
    print(f"u = \n{u}")
    print(f"v = \n{v}")
    print(f"w = \n{w}")

    