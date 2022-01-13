#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: Segmentation
    
@filename: label_simplex_fields.py
    
@description:
    transform a label field into a binary simplex
    field. 
    Transform a simplex valued field into a label field
    via thresholdding/projection onto vertices

This is a pure Python reimplementation of my old C++ code.
This should be moved into some simplex/segmenation package

@author: François Lauze, University of Copenhagen    
Created on Tue Nov 16 14:16:12 2021

"""

from functools import reduce
from operator import mul
import numpy as np


__version__ = "0.0.1"
__author__ = "François Lauze"



def label_to_simplex_field(u):
    """
    Convert a integer valued label field into a simplex valued field.
    :param u: ndarray int
        input field, dimension dim. Assume that labels are integers 
        0 <= l <= K_1
    :return: ndarray float
        output field, dimemnsion (dim, K) 
    """
    
    ushape = u.shape
    u.shape = u.size
    K = u.max() + 1
    v = np.zeros(u.shape + (K,), dtype='float32')
    
    id_K = np.eye(K)
    for k in range(K):
        idx = np.where(u == k)
        v[idx] = id_K[k]
        
    u.shape = ushape
    v.shape = ushape + (K,)
    return v



def simplex_to_label_field(v):
    """
    Project/convert a simplex valued field to an integer one.
    :param v: ndarray of floats
        a field of dimension basedim + (K,), supposed to be simplex valued
    :return: ndarray of ints
        output field, of dimensions basedim, with values in {0..K-1}
    """
    
    basedim = v.shape[:-1]
    K = v.shape[-1]
    
    lbasedim = reduce(mul, basedim, 1)
    v.shape = (lbasedim, K)
    u = np.argmax(v, axis=1)
    v.shape = basedim + (K,)
    u.shape = basedim
    return u


    

if __name__ == '__main__':
    u = np.random.randint(0,5,size=(2,8))
    v = label_to_simplex_field(u)
    w = simplex_to_label_field(v)
    print(f"u = \n{u}")
    print(f"v = \n{v}")
    print(f"w = \n{w}")

    