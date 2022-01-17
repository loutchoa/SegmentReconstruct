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

import numpy as np
import  matplotlib.pyplot as plt


__version__ = "0.0.1"
__author__ = "François Lauze"


