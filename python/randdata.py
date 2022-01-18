# -*- coding: utf-8 -*-
"""
Project: PyCT

File: randdata.py
Description k-phases random data generated via random disks

Author: François Lauze, University of Copenhagen
Date: October 2016
"""


import os
import os.path
import numpy as np


def random_image(image_size, number_of_disks, grey_levels_distribution, bg=0.0, grey_levels=None,
                 min_radius=1, max_radius=50):
    """
    Random 2D image of size (m,m) with random disks overlapping with discrete intensity distributions.

    Arguments:
    ----------
    m : int
        image size (square (m,m) image)
    number_of_disks: int
        number of disks to be generated.
    grey_levels_distribution: 1D float numpy array
        probability distribution of grey levels where len(grey_levels_distribution) gives the
        number og grey levels, grey_levels_distribution.sum() must be 1 and grey_levels_distribution entries must
        be positive.
    bg: float
        background value. default is 0.0
    grey_levels: None or 1D float numpy array
        grey_levels corresponding to each class of the grey level distribution
        grey_levels_distribution. Thus if not None, len(grey_levels) must be the same as len(grey_levels_distribution).
        If None: from bg + 1 up to bg + k.
    min_radius: float
        minimal radius for generated disks
    max_radius; float
        maximal radius for generated disks
    """

    def sample_grey_levels_distribution():
        l = np.random.randn(1)
        for i in range(k):
            if l < grey_levels_distribution[i]:
                return i
        return k - 1
    grey_levels_distribution = np.array(grey_levels_distribution)
    k = grey_levels_distribution.size
    if grey_levels is None:
        grey_levels = np.array([bg + i for i in range(1, k + 1)])
    else:
        grey_levels = np.array(grey_levels)

    u = np.zeros((image_size, image_size))

    c = np.random.randint(5, image_size - 6, size=(number_of_disks, 2))
    r = np.random.randn(number_of_disks) * (max_radius - min_radius) + min_radius

    a,b = np.mgrid[0:image_size, 0:image_size]

    for i in range(number_of_disks):
        val = grey_levels[sample_grey_levels_distribution()]
        s = np.where((a - c[i,0])**2 + (b-c[i,1])**2 <= r[i]**2)
        u[s] = val

    return u


def generate_test_base():
    basedir = '.'
    b200 = basedir + '/200'
    b300 = basedir + '/300'
    b400 = basedir + '/400'

    if not os.path.exists(b200):
        os.mkdir(b200)
    if not os.path.exists(b300):
        os.mkdir(b300)
    if not os.path.exists(b400):
        os.mkdir(b400)

    grey_levels_distribution = [0.1, 0.2, 0.7]
    grey_levels = [0.25, 0.78, 1.0]

    m = 200
    for i in range(10):
        u = random_image(m, 5 * m, grey_levels_distribution=grey_levels_distribution, grey_levels=grey_levels)
        name = 'random_disk_%d' % (i+1)
        np.savez(b200 + '/' + name, x=u)

    m = 300
    for i in range(10):
        u = random_image(m, 5 * m, grey_levels_distribution=grey_levels_distribution,
                         grey_levels=grey_levels, min_radius=4, max_radius=8)
        name = 'random_disks_%d' % (i+1)
        np.savez(b300 + '/' + name, x=u)

    m = 400
    for i in range(10):
        u = random_image(m, 5 * m, grey_levels_distribution=grey_levels_distribution, grey_levels=grey_levels,
                         min_radius=4, max_radius=9)
        name='random_disks_%d' % (i+1)
        np.savez(b400 + '/' + name, x=u)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib_utils import impixelinfo

    m = 300
    v = random_image(m, m * 5, [0.1, 0.2, 0.7], grey_levels=[0.25, 0.78, 1.0], min_radius=3, max_radius=7)
    plt.imshow(v, cmap=cm.Greys_r)
    impixelinfo()
    plt.show()

