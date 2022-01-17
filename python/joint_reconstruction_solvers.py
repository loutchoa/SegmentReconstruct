#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: psychic-memory
    
@filename: joint_reconstruction_solvers.py
    
@description: a ART with FISTA for a reconstruction part with
added reaction to segmentation.
In the first implementation, I assume, as Chryoula, as the segmentation ROI is
the entire image, and this simplifies somewhat the algebra.


@author: François Lauze, University of Copenhagen    
Created on Mon Jan 17 19:07:41 2022

"""
from random import shuffle
import numpy as np
#import  matplotlib.pyplot as plt
from fistaTV import fistaTV
from tomo_utils import clean_projection_system, normalise_projection_system
from graddiv import grad2, grad3

__version__ = "0.0.1"
__author__ = "François Lauze"


# Francois: a rewriting, to see whether with my ops I could compare and
# find the bug... 
# I will assume that k-means or whatever part of segmentation has been run,
# and I only need d := E_v(c) and that, as above, the ROI is the whole image so 
# Pi is the identity. This simplifies the algebra! In particular, one can go
# with the "magic lemma" directly, Lemma 1.3.1 of my notes
# Maybe the long list of parameters should be provided in a class?
def Row_Action_Reconstruction_F(A, b, d,
                                alpha, beta,
                                x0, rho, t, a,
                                maxiter = 10000,
                                fista_iters = 50,
                                xmin=0.0, xmax = float("inf"),
                                conv_thresh=1e-3,
                                normalise=True):
    """
    Run an iterative proximal for the reconstruction part.
    
    Modified Damped-ART with term-wise reaction to segmentation part
    and regularisation and box-projection. 

    Parameters
    ----------
    A : csr matrix of float (or float32)
        projection matrix, assumed to have no zero lines.
    b : numpy array float(32)
        measurements / sinogram stuff. 2D or 3D array.
    d : numpy array float(32):
        the segmentation image (i.e. E_v(c)).
    alpha : float
        TV regularisation weight.
    beta : float
        double weight of segmentation term
    x0 :  numpy array of float (or float32)
            Initial reconstruction value.
    rho : float
        relaxation parameter, should be in [0, 2]
    t : float
        used to build the sequence of proximal step lengths with a below.
    a : float
        used to build sequence of proximal step lengths with t above:
        tau_k = t/(k+1)**a with k the sweep number
    maxiter : int, optional
        maximum number of iterations (sweeps). The default is 10000.
    fista_iters : int, optional
        number of iterations of FISTA_TV. The default is 50.
    xmin : float, optional
        min value of a pixel. The default is 0.0.
    xmax : float, optional
        max value of a pixel. The default is float("inf").
    conv_thresh : float, optional
        convergence threshold for the reconstruction. The default is 1e-3.
    normalise : bool, optional.
        if True, the system is normalised to have norm one matrix rows.
    Returns
    -------
    x : reconstruction
    f : list of objective values
    iter: number of iterations (full sweeps) performed
    """
    # shortcut
    pinf = float("Inf")
    minf = -pinf
    
    bshape = b.shape
    D = grad2 if b.ndim == 2 else grad3
    b.shape = b.size

    # Normalise the system? I guess always, but not sure at that point?
    # I could assume that null rows and corresponding measurements have 
    # been removed?
    if normalise:
        An, bn = clean_projection_system(A, b)
        normalise_projection_system(An, bn)
    else:
        An = A
        bn = b
        
    def objective(x):
        return 0.5*((An@x - bn)**2).sum()\
            + (beta/2)*((x-d)**2).sum() \
            + alpha*np.abs(D(x)).sum()
        
    def Pbox(x) : 
        if xmin > minf:
            np.place(x < xmin, xmin)
        if xmax < pinf:
            np.place(x > xmax, xmax)
    
    x = x0.copy()
    f = [objective(x)]
    
    # number of rows of A, i.e., observations
    M = A.shape[0]
    # the M + 1 because the reconstruction functional is composed of a sum
    # of M + 1 functions: one per matrix row and the spatial regularisation.
    # the box constraint is applied to each proximal computation if necessary.
    gamma = beta/(M+1)
    
    for k in range(maxiter):
        # for convergence
        x_prec = x.copy()
        
        # some preparation:
        # the time/gradient step length at this sweep/scan of the rows
        tau_k = t/(k+1)**a
        # c_k is the sum of weights in mixing segmentation image and reconstruction, 
        # segmentation image d: weight gamma, current reconstruction x: weight 1/c_k
        c_k = gamma + (1/tau_k)
    
        # start with the pure reconstruction part, randomise
        # the order of lines of A first
        # If the rows have been normalised, randomisation should improve
        # the convergence, it does for Kaczmarz. And this is probably the case 
        # even without normalisation
        row_indices = list(range(M))
        shuffle(row_indices)
        
        for i in row_indices:
            # d_k is the weighted term from segmentation reaction d with weigh
            # gamma and the previous value of x with weight 1/tau_k ,
            # divided by c_k -- see the computation for the proximals. Still,
            # it means that d_k is a convex combination of d and x
            d_k = (gamma*d + (1/tau_k)*x)/c_k
            a_i = A[i]
            b_i = b[i]
            
            # some work could be done to only look at the modified 
            # pixel values, i.e. those in support(a_i). Actually, the list of
            # non zero indices should be stored in A.indices...
            z = d_k + ((b_i - a_i@d_k)/(a_i@a_i + c_k))*a_i
            x = (1-rho)*x + rho*z
            Pbox(x)
            
        # Then the modified FISTA-TV step with regularisation weight
        # update d_k with the current value of x
        d_k = (gamma*d + (1/tau_k)*x)/c_k
        # FISTA-TV weight
        l = 2*alpha/c_k
        # should I apply or not apply box constraint conditions in FISTA-TV
        # I ignore them so as to follow Andersen-Hansen 2014
        z = fistaTV(d_k, l, fista_iters, minf, pinf)
        x = (1-rho)*x + rho*z
        Pbox(x)
        
        # I could have instead computed
        # x = fistaTV(d_k_norm, l, fista_iters, xmin, xmax)
        # and ignored the Pbox(x) which would be unnecessary..
        f.append(objective(x))
        
        if np.linalg.norm(x - x_prec) < conv_thresh:
            break

    b.shape = bshape
    return x, f, k    
