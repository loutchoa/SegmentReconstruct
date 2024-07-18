# -*- coding: utf-8 -*-
"""
@project: imgtools
@file: scalespace.py
@escription: scalespace implements Fourier-Based Gaussian scale-space / convolution and
scale derivatives for 1D, 2D, 3D signals,signals that can be vector valued.

In spatial domain:

    - 1D signals: should have shape/dimension (x)
    - 1D vectorial signals. Should have shape/dimensions (x,v)
    - 2D signals: should have dimensions (x,y)
    - 2D vectorial signals: should have dimensions (x,y,v)
    - 3D signals: should have dimensions (x,y,z)
    - 3D vectorial signals: should have dimensions (x,y,z,v)

In the vectorial case, a copy of the array with vectorial dimension moved
to first position is made, as it is necessary in order to perform Fourier
transform:

    - 1D vectorial: (v,x)
    - 2D vectorial: (v,x,y)
    - 3D vectorial: (v,x,y,z)

In the Fourier domain approach, the vectorial dimension must be the first,

Neumann / reflected  boundary conditions can  be incorporated by properly
mirroring the signals with the drawback of memory size (x2 in 1D, x4 in 2D,
x8 in 3D).


* Added 08-2016: possibility that the vectorial dimension is first. This in
  accordance with some DWI nifti data.

* Added 09-2017: anisotropic smoothing in x, y, z. This is in order to be able to 
  deal with non square pixels / cube voxels, especially with some nifty DWI files.

* Added 09-2017: structure tensor, Hessian of Gaussian, Laplacian of Gaussiam, 
  mean curvature at Gaussian scale.

* Added 10-2017: determinant and trace of symmetric matrix fields.

* Added 10-2017: support for float32 numpy arrays.

* Modified 10-2017 and November 2019: fft supports now pyfftw for best performances, with 
  code running for float32 and float dtypes. If it cannot find the pyfftw module, it falls 
  back to numpy fft.

* Modified 11-2018: used // instead of / for series of integer division, make it Python3
  compatible

* 01-2021. Scale normalisation is implemented for gausssian_scale_space() function,
  but I should modify the code to push it automatically to some of the functions
  that use it. **kwargs like mechanism?

* Need some tests for mean curvature, hessian...

copyright 2014-2021 François Lauze, University of Copenhagen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

from operator import add
import numpy as np
from numpy import inf as Inf
from multiprocessing import cpu_count
from functools import reduce
from numbers import Number
from typing import Union, Tuple, List, Iterable

# Set to FFT to 'pyfftw' to use pyfftw if possible, else numpy
# set FFT to '' does the same as 'pyfftw'
# set FFT to 'numpy' or anything not '' or 'pyfftw' to use numpy
FFT = "numpy"   # Right now, pyfftw is not working properly, so I set it to numpy
if FFT in ['pyfftw', '']:
    try:
        import pyfftw
        pyfftw.interfaces.cache.enable()
        FFT_THREADS = cpu_count()  # - there is a bug in pyfftw, so I set it to 1?

        def fftn (data: np.ndarray):
            return pyfftw.interfaces.numpy_fft.fftn(data, threads=FFT_THREADS)

        def ifftn(data: np.ndarray):
            return pyfftw.interfaces.numpy_fft.ifftn(data, threads=FFT_THREADS)

    except ImportError:
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn
else:
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn


__author = "François Lauze, university of Copenhagen"
__date__ = "Date: 2014-2024."
__version = '1.57.0796' # pi/2

# Pi and sqrt(-1)
_pi = np.pi
_i = 1j


def gss_fft(f, vectorial):
    """
    Compute Forward Fourier Transform of f using fftw3 or numpy

    :param f: Spatial-domain function.
    :param vectorial: True means the last dimension is vectorial,
        False means f is a scalar valued array.

    :returns: Forward Fourier transform of f.
    """
    ff_dtype = complex if f.dtype in [float, complex] else 'complex64'
    if not vectorial:
        ff = fftn(f)
    else:
        ff = np.zeros_like(f, dtype=ff_dtype)
        vdim = f.shape[0]
        for v in range(vdim):
            ff[v] = fftn(f[v])
    return ff



def gss_ifft(f, vectorial):
    """
    Compute Inverse Fourier Transform of f using fftw3 or numpy

    :param f: Frequency-domain function.
    :param vectorial: True means the last dimension is vectorial,
            False means f is a scalar valued array.

    :returns: Inverse Fourier transform of f.
    """
    if not vectorial:
        ff = ifftn(f)
        # ff /= normfact
    else:
        vdim = f.shape[0]
        ff = np.zeros_like(f, dtype=f.dtype)
        for v in range(vdim):
            ff[v] = ifftn(f[v])
        # ff /= normfact
    return ff.real



def _to_float_array(s, n, dtype=float, argname1='sigma', argname2='dim'):
    """
    Attempt at converting s to a (numpy) array of length n
    If s turns to be a scalar, just makes an array with repeating values of s n times.
    If s is not scalar, it should already be array like with len n.
    
    :param s : Iterable to be converted to a float array
    :param n: positive integer, coding the length of the array
    :returns: np.ndarray, float array of length n
    :raise: TypeError if s is not convertible to a numpy array or if s is not iterable, ValueError if len(s) != n
    """
    try:
        s = np.array([float(s)]*n, dtype=dtype)
    except:
        try:
            s = np.array(s, dtype=dtype)
        except:
            return TypeError('Argument %s is not convertible to a numpy array.' % argname1)
        if len(s) != n:
            return ValueError('Length of argument %s does not match length of argument %s.' % (argname1, argname2))
    return s


def nbc_fourier(f, vectorial=False):
    """
    Reflective boundary conditions to be used with Fourier
    transform, mimicking Neumann BCs

    :param f: input image to be symmetrised, dimension up to 3 plus a potential
        vectorial one.
    :param vectorial: boolean
        - False assumes scalar valued image
        - True assumes vector valued image, read from the FIRST dimension.
          (the reason is efficiency of Fourier transform).

    :returns: rf: image extended by reflections.
    :raise:
        ValueError if f is more than 3-dimensional or if f is 1D with vectorial=True
    """
    dim_f = len(f.shape)
    if ((dim_f > 3 and not vectorial) or
            (dim_f > 4 and vectorial) or
            (dim_f == 1 and vectorial)):
        raise TypeError('Bad dimensions.')

    # I treat differently vectorial and non-vectorial cases.
    if not vectorial:
        if dim_f == 1:
            # there is no real notion of vertical or horizontal 1D image,
            # or more precisely it seems to be "horizontal" by default.
            # that's why I think I need hstack() there.
            rf = np.hstack((f, f[::-1]))
        elif dim_f == 2:
            vf = np.vstack((f, f[::-1, :]))
            rf = np.hstack((vf, vf[:, ::-1]))
        else:
            vf = np.vstack((f, f[::-1, :, :]))
            hf = np.hstack((vf, vf[:, ::-1, :]))
            rf = np.concatenate((hf, hf[:, :, ::-1]), axis=2)

    else:
        if dim_f == 2:
            # remark similar to the non-vectorial case?
            rf = np.hstack((f, f[:, ::-1]))
        elif dim_f == 3:
            vf = np.concatenate((f, f[:, ::-1, :]), axis=1)
            rf = np.concatenate((vf, vf[:, :, ::-1]), axis=2)
        else:
            vf = np.concatenate((f, f[:, ::-1, :, :]), axis=1)
            hf = np.concatenate((vf, vf[:, :, ::-1, :]), axis=2)
            rf = np.concatenate((hf, hf[:, :, :, ::-1]), axis=3)

    return rf


def derivative_filter_fourier(dim, order, dtype=float):
    """
    Derivative filter in Fourier domain.

    :param dim: array of dimensions, at most len 3.
    :param order: list-like, derivative orders for each dimension.
    :param dtype: numpy.dtype for created array.
    :returns: D : numpy array. Fourier filter for this derivative
    :raises: ValueError if dimensions > 3 or if len(dim) != len(order)
    """
    l_dim = len(dim)
    lorder = len(order)
    if (l_dim > 3 or lorder > 3) or (l_dim != lorder):
        raise ValueError('Incorrect dimensions or orders.')

    if l_dim == 1:
        m = dim[0] // 2
        k = order[0]
        omega = (np.hstack((range(m), range(-m, 0))) / (2.0 * m)) ** k
        D = omega * ((2 * _i * _pi) ** k)
    elif l_dim == 2:
        m = dim[0] // 2
        n = dim[1] // 2
        k1, k2 = order
        fact = (2 * _i * _pi) ** (k1 + k2)
        omega1 = (np.hstack((range(m), range(-m, 0))) / (2.0 * m)) ** k1
        omega2 = (np.hstack((range(n), range(-n, 0))) / (2.0 * n)) ** k2
        D = (np.kron(omega1, omega2) * fact).reshape((2 * m, 2 * n))
    else:
        m = dim[0] // 2
        n = dim[1] // 2
        p = dim[2] // 2
        k1, k2, k3 = order
        fact = (2 * _i * _pi) ** (k1 + k2 + k3)
        omega1 = (np.hstack((range(m), range(-m, 0))) / (2.0 * m)) ** k1
        omega2 = (np.hstack((range(n), range(-n, 0))) / (2.0 * n)) ** k2
        omega3 = (np.hstack((range(p), range(-p, 0))) / (2.0 * p)) ** k3
        D = (np.kron(np.kron(omega1, omega2), omega3) * fact).reshape((2 * m, 2 * n, 2 * p))

    if dtype is not float:
        D = D.astype(dtype)
    return D



def gaussian_filter_fourier_dim1(m, sigma, dtype=float):
    """
    Create a 1D Gaussian filter in the Fourier domain, size 2m.

    :param int m: Half-array dimension.
    :param float sigma: Gaussian standard deviation.
    :param numpy.dtype dtype: Base type of the created array. Default is `float`.
    :returns: 1D filter with dtype set to `complex`.
    """
    G = np.zeros(2 * m, dtype=complex)
    if sigma == Inf:
        G[0] = 1.0
        return G
    M = (np.arange(m + 1, dtype=float) / (2 * m)) ** 2
    G[:m + 1] = np.exp(-M * 2 * (_pi * sigma) ** 2)
    G[m:] = G[m:0:-1]
    if dtype is not float:
        G = G.astype(dtype)
    return G


def gaussian_filter_fourier(dim, sigma, dtype=float):
    """
    Gaussian filter in Fourier space.

    Computes the Fourier transform of a Gaussian with given dimensions and
    standard deviation.

    :param dim: Array of dimensions of the filter array.
    :param sigma: Standard deviation of the filter. If sigma is float, teh filter is isotropic.
        If sigma is array-like, it must have the same length as `dim`, specifying per-axis smoothing.
    :param dtype: Base type of the created array. Default is `float`.

    :returns: The Fourier transform of the Gaussian kernel.

    :raises ValueError: If `len(dim) > 3` or if `sigma` is array-like and `len(sigma)` differs from `len(dim)`.
    :raises TypeError: If `sigma` is neither a float-compatible scalar nor an array.
    """
    if len(dim) > 3:
        raise ValueError('Incorrect number of dimensions.')

    sigma = _to_float_array(sigma, len(dim))
    if isinstance(sigma, Exception): raise sigma
    # hopefully sigma is now a float numpy array with correct length

    if len(dim) == 1:
        return gaussian_filter_fourier_dim1(dim[0] // 2, sigma[0], dtype=dtype)
    elif len(dim) == 2:
        G0 = gaussian_filter_fourier_dim1(dim[0] // 2, sigma[0], dtype=dtype)
        G1 = gaussian_filter_fourier_dim1(dim[1] // 2, sigma[1], dtype=dtype)
        G0.shape = (G0.size, 1)
        G1.shape = (1, G1.size)
        return np.kron(G0, G1)
    else:
        G0 = gaussian_filter_fourier_dim1(dim[0] // 2, sigma[0], dtype=dtype)
        G1 = gaussian_filter_fourier_dim1(dim[1] // 2, sigma[1], dtype=dtype)
        G2 = gaussian_filter_fourier_dim1(dim[2] // 2, sigma[2], dtype=dtype)

        G0.shape = (G0.size, 1, 1)
        G1.shape = (1, G1.size, 1)
        G2.shape = (1, 1, G2.size)
        return np.kron(np.kron(G0, G1), G2)
        

def gss_remove_symmetry(f, vectorial):
    """
    Take the first half / quadrant / octant of f to get back to
    original function when the function f has been extended by
    mirror symmetries to simulate Neumann BCs.

    Internal function to the scalespace module.

    :param f: Symmetrised function.
    :param vectorial: True means the last dimension is vectorial, False means f is a scalar valued array.

    :returns: de-symmetrised function.
    """

    if not vectorial:
        if len(f.shape) == 1:
            m = f.shape[0]
            return f[:m // 2]
        elif len(f.shape) == 2:
            m, n = f.shape
            return f[:m // 2, :n // 2]
        else:
            m, n, p = f.shape
            return f[:m // 2, :n // 2, :p // 2]
    else:
        if len(f.shape) == 2:
            m = f.shape[1]
            return f[:, m // 2]
        elif len(f.shape) == 3:
            m, n = f.shape[1:]
            return f[:, :m // 2, :n // 2]
        else:
            m, n, p = f.shape[1:]
            return f[:, m // 2, n // 2, p // 2]


def gss_apply_filter_fourier(f, H, vectorial):
    """
    Applies the filter H to the image f in the frequency domain, i.e.,
    point-wise multiplies f by H, taking care of the potentially
    vector-valued structure of f.

    :param f: Function to be filtered, in the frequency domain.
    :param H: Filter function, in the frequency domain. Should have the
            same dimensions as f (except for f's last one if f is vectorial).
    :param vectorial: True means the last dimension is vectorial,
            False means f is a scalar valued array.

    :returns: Frequency domain filtering of f by H.
    """
    if not vectorial:
        ff = f * H
    else:
        vdim = f.shape[0]
        ff = np.zeros(f.shape, dtype=complex)
        for v in range(vdim):
            ff[v, :] = f[v, :] * H
    return ff


def gaussian_scalespace(f, sigma, order=None, vectorial=False, vectorial_first=False, reflected=True):
    """
    Compute Gaussian scale space and Gaussian derivatives.

    This function computes Gaussian scale-space / Gaussian derivatives of an
    array, that can be seen as a scalar or vector valued function. Maximum
    dimensions are 3 (not counting the vectorial one).

    :param f: Input function, its dimensions must be 1, 2, or 3, possibly including a
              vectorial one i.e. adding an extra dimension (thus up to 4).
              f must have dtype 'float' or 'float32'.
    :param sigma: Standard deviation of the Gaussian kernel.
                  If array-like, the length of sigma must match the image dimension (1, 2 or 3)
                  and it corresponds to standard deviations applied per axis.
    :param order: Array-like, optional. Each line corresponds to a particular derivative to be computed.
                  If array of order of derivatives of size (m, dim(f)). If not given, zero
                  order derivative, i.e., plain scale-space.
    :param vectorial: Boolean, optional. Specifies whether the first/last dimension should be considered as
                      vectorial (i.e., f should be a vector valued function).
    :param vectorial_first: Boolean. True means that the vectorial dimension is the first one,
                            False (default) means the last one.
    :param reflected: Boolean. True: mirror symmetries the function to mimic Neumann BCs.
                      False: do nothing, i.e., assume standard toric BCs.

    :returns: Gaussian scale-space or list of Gaussian derivatives, depending on order.
    :rtype: float(32) numpy or list of float numpy arrays.

    :raises ValueError: If some dimensions are incorrect or mismatch.
    :raises TypeError: If non None order is not convertible to numpy array.

    Example:
    --------
    We compute the Laplacian of Gaussian of a 2D or 3D image f, non vectorial, with
    reflected boundary conditions
        >>> from operator import add
        >>> LoGf = reduce(add, gaussian_scalespace(f, sigma, order=2*np.eye(f.ndim)))
        >>>
    """
    # check if order is array-like when not None
    if order is not None:
        try:
            order = np.array(order)
        except:
            raise TypeError('order parameter is not array-like.')

    # check dimensions of f and order.
    f_dim = f.ndim if not vectorial else f.ndim - 1
    if f_dim > 3:
        raise ValueError('Bad dimension: dimension of f > 3.')
    if order is not None:
        if ((len(order.shape) == 1) or
                (len(order.shape) == 2 and order.shape[1] != f_dim) or
                (len(order.shape) > 2)):
            raise ValueError('Bad dimensions: dimensions mismatch.')

    dim = f.shape if not vectorial else f.shape[1:] if vectorial_first else f.shape[:-1]

    # if derivative order was not specified, set to 0
    # otherwise be sure that order.shape has the form
    # (m, f_dim)
    if order is None:
        order = np.zeros((1, f_dim))
    else:
        if len(order.shape) == 1:
            order.shape = (1, order.shape[0])

    nb_filters = order.shape[0]

    # if reflected is False, I need to dimensions of f to be even,
    # because it is easier when constructing the filters; I don't
    # want to touch that part, so if at least one of the dimensions
    # fails to be even, raise an exception.
    if not reflected:
        # original code: any seems to return a generator, converted to True,
        # as non-zero. Mysterious bug in this expression?
        # works properly inside a standard ipython:
        # if any(dim[i] % 2  for i in range(len(dim))):
        # 
        # Will it fix it?
        odd_dim = [dim[i] % 2 for i in range(len(dim))]
        if any(odd_dim):
            raise ValueError("(non-vectorial) dimensions of f must be even when reflected=False.")

    # prepare data. If data is vectorial, the last dimension is rolled
    # to the first, so that Fourier transform will deal with contiguous
    # slices of data.
    if vectorial and not vectorial_first:
        f = np.rollaxis(f, f_dim).copy()

    # Complexify f and applies mirroring symmetry if reflected == True
    ff_dtype = complex if f.dtype is float else 'complex64'

    cf = f.astype(ff_dtype)
    if reflected:
        cf = nbc_fourier(cf, vectorial)
        dim = cf.shape if not vectorial else cf.shape[1:]

    # Fourier transform it
    fcf = gss_fft(cf, vectorial)

    # Create the Fourier Gaussian filter
    G = gaussian_filter_fourier(dim, sigma, dtype=ff_dtype)

    # Compute the different Fourier derivatives filters and apply them
    res = []
    for i in range(nb_filters):
        D = derivative_filter_fourier(dim, order[i, :], dtype=ff_dtype)
        res.append(gss_apply_filter_fourier(fcf, G * D, vectorial))

    # Fourier transform back (and get real part)
    for i in range(nb_filters):
        res[i] = gss_ifft(res[i], vectorial)

    # Remove the reflected parts if reflected == True.
    # Unroll the vectorial part if necessary, and
    # create contiguous arrays from it.

    for i in range(nb_filters):
        if reflected:
            res[i] = gss_remove_symmetry(res[i], vectorial)
        if vectorial and not vectorial_first:
            res[i] = np.rollaxis(res[i], 0, len(res[i].shape)).copy()

    if nb_filters == 1:
        return res[0]
    else:
        return res


def Fourier_gaussian_scalespace(f, sigma, order=None, vectorial=False):
    """
    Compute Gaussian scale-space and Gaussian derivatives in Fourier
    domain, assuming that f is already in Fourier domain, a la Jon.

    :param f: Fourier transform of the input function, its dimensions must
              be 1, 2, or 3, possibly including a vectorial one i.e., adding
              an extra dimension (thus up to 4). In the case where
              vectorial == True, the vectorial dimension is assumed to be the
              first one. This has to do with coding of the other operations,
              Fourier-Transform and filtering ones.
    :param sigma: Standard deviation of the Gaussian kernel.
                  If array-like, the length of sigma must match the image dimension (1, 2, or 3)
                  and it corresponds to standard deviations applied per axis.
    :param order: Array-like, optional. Each line corresponds to a particular derivative to be computed.
                  If array of order of derivatives of size (m, dim(f)). If not given, zero
                  order derivative, i.e., plain scale-space.
    :param vectorial: Boolean, optional. Specifies whether the first dimension should be considered as
                      vectorial (i.e., f should be a vector valued function).

    :returns: Gaussian scale-space or list of Gaussian derivatives, depending on order, in Fourier domain.
    :rtype: Complex numpy array or list of float numpy arrays.

    :raises ValueError: If some dimensions are incorrect or mismatch.
    :raises TypeError: If non None order is not convertible to numpy array.
    """
    # check if order is array-like when not None
    if order is not None:
        try:
            order = np.array(order)
        except:
            raise TypeError('order parameter is not array-like.')

    # check dimensions of f and order.
    fdim = len(f.shape) if not vectorial else len(f.shape) - 1
    if fdim > 3:
        raise ValueError('Bad dimension: dimension of f > 3.')
    if order is not None:
        if ((len(order.shape) == 1) or
                (len(order.shape) == 2 and order.shape[1] != fdim) or
                (len(order.shape) > 2)):
            raise ValueError('Bad dimensions: dimensions mismatch.')

    dim = f.shape if not vectorial else f.shape[1:]

    # if derivative order was not specified, set to 0
    # otherwise be sure that order.shape has the form
    # (m,dimf)
    if order is None:
        order = np.zeros((1, fdim))
    else:
        if len(order.shape) == 1:
            order.shape = (1, order.shape[0])

    nb_filters = order.shape[0]

    # if reflected is False, I need to dimensions of f to be even,
    # because it is easier when constructing the filters; I don't
    # want to touch that part, so if at least one of the dimensions
    # fails to be even, raise an exception.
    if any((dim[i] % 2) == True for i in range(len(dim))):
        raise ValueError("(non-vectorial) dimensions of f must be even.")

    # Create the Fourier Gaussian filter
    G = gaussian_filter_fourier(dim, sigma, dtype=f.dtype)

    # Compute the different Fourier derivatives filters and apply them
    res = []
    for i in range(nb_filters):
        D = derivative_filter_fourier(dim, order[i, :], dtype=f.dtype)
        res.append(gss_apply_filter_fourier(f, G * D, vectorial))

    if nb_filters == 1:
        return res[0]
    else:
        return res


def structure_tensor(f, inner_scale, outer_scale, vectorial=False, vectorial_first=False, reflected=True):
    """
    This is an illustration of the use of Gaussian Scale-space to 
    compute classical structure tensor of an image/volume.
    As is of usage, the structure tensor of a vectorial image is
    obtained by summing the tensors of each scalar field.

    :param f: Array of dim 1, 2, 3, or 4. 1 is scalar, while 4 is always vectorial.
    :param inner_scale: Inner scale for computation of derivatives. If float array-like, its length
                        must match f (non-vectorial) dimensions. This would be typically used to accommodate
                        non-square (resp. cubic) pixels (resp. voxels).
    :param outer_scale: Outer scale for smoothing of structure field. If float array-like, its length
                        must match f (non-vectorial) dimensions. This would be typically used to accommodate
                        non-square (resp. cubic) pixels (resp. voxels).
    :param vectorial: True means that the data is vectorial. The last dimension if vectorial_first is False
                      or the first one if vectorial_first is True.
                      When f.ndim is in {1, 4}, this is ignored.
    :param vectorial_first: True means first dim encodes the vectorial values,
                            False means last dim encodes the vectorial values, whenever it makes sense.
    :param reflected: If True, reflected boundary conditions will be applied.
                      If False, standard periodic are used (needs non-vectorial dimensions to be even).

    :returns: Structure tensor field. Each entry is a compressed symmetric matrix, only upper diagonal
              saved, in standard order (line concatenation).
              Dimensions, ignoring the potential vectorial dimension of f (are marginalized in st calculations):
              - f.ndim=1, f.shape = (m), st.shape = (m)
              - f.ndim=2, f.shape = (m, n), st.shape = (m, n, 3)
              - f.ndim=3, f.shape = (m, n, p), st.shape = (m, n, p, 6)
    :rtype: numpy float array
    """

    if (f.ndim <= 0) or (f.ndim > 4):
        raise TypeError('Only dim1 to 3, with maybe vectorial content images can be used.')
    n = f.ndim if not vectorial else f.ndim - 1
    if n == 0:
        raise ValueError('A dimension 1 array cannot be vectorial.')
    elif n == 4:
        raise ValueError('A dimension 4 array must be vectorial.')

    fders = gaussian_scalespace(f, inner_scale, order=np.eye(n), vectorial=vectorial,
                                vectorial_first=vectorial_first, reflected=reflected)
    if type(fders) == np.ndarray:
        fders = [fders]
    if vectorial:
        axis = 0 if vectorial_first else -1
        for i in range(len(fders)):
            np.sum(fders[i], axis=axis, out=fders[i])

    if n == 1:
        fx, = fders
        st = gaussian_scalespace(fx, outer_scale, vectorial=False, reflected=reflected)
    elif n == 2:
        fx, fy = fders
        st = np.zeros(fx.shape + (3,))
        st[:, :, 0] = fx ** 2
        st[:, :, 1] = fx * fy
        st[:, :, 2] = fy ** 2
        st = gaussian_scalespace(st, outer_scale, vectorial=True, vectorial_first=False, reflected=reflected)
    else:
        fx, fy, fz = fders
        st = np.zeros(fx.shape + (6,))
        st[:, :, :, 0] = fx ** 2
        st[:, :, :, 1] = fx * fy
        st[:, :, :, 2] = fx * fz
        st[:, :, :, 3] = fy ** 2
        st[:, :, :, 4] = fy * fz
        st[:, :, :, 5] = fz ** 2
        st = gaussian_scalespace(st, outer_scale, vectorial=True, vectorial_first=False, reflected=reflected)
    return st



def hessian_of_gaussian(f, sigma, vectorial=False, vectorial_first=False, reflected=True, marginalize=False):
    """
    Hessian of Gaussian for a signal, image, or volume. 
    
    :param f: Array of dim 1, 2, 3, or 4. Dim 1 is always scalar while dim 4 is always vectorial.
    :param sigma: Gaussian scale for evaluation of derivatives. If float array-like, its length
                  must match f (non-vectorial) dimensions. This would be typically used to
                  accommodate non-square (resp. cubic) pixels (resp. voxels).
    :param vectorial: True means that the data is vectorial. The last dimension if
                  vectorial_first is False or the first one if vectorial_first is True.
                  When f.ndim is in {1, 4}, this is ignored.
    :param vectorial_first: True means first dim encodes the vectorial values,
                  False means last dim encodes the vectorial values, whenever it makes sense.
    :param reflected: If True, reflected boundary conditions will be applied.
                  If False, standard periodic are used (needs non-vectorial dimensions to be even).
    :param marginalize: If True and vectorial is True, returns Hessian per vectorial dimensions.
                  If False, sum result over vectorial dimensions.

    :returns: Hessian matrix field. Each entry is a compressed symmetric matrix, only upper diagonal
              saved, in standard order (line concatenation).
              Dimensions:
              - Vectorial non-marginalized or scalar f:
                - f.shape if f is dim 1
                - f.shape + (3,) if f has dim 2
                - f.shape + (6,) if f has dim 3
              - Vectorial, marginalized: same as above but without vectorial dimensions.
    :rtype: numpy float array

    :raises ValueError: If dimensions are incorrect or mismatch.
    :raises TypeError: If input types are incorrect.
    """
    if (f.ndim <= 0) or (f.ndim > 4):
        raise TypeError('Only dim1 to 3, with maybe vectorial content images can be used.')
    n = f.ndim if vectorial else f.ndim - 1
    if n == 0:
        raise ValueError('A dimension 1 array cannot be vectorial.')
    elif n == 4:
        raise ValueError('A dimension 4 array must be vectorial.')

    if n == 1:
        order = (2,)
    elif n == 2:
        order = ((2, 0), (1, 1), (0, 2))
    else:
        order = ((2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2))
    ders = gaussian_scalespace(f, sigma, order=order, vectorial=vectorial,
                               vectorial_first=vectorial_first, reflected=reflected)

    if n == 1:
        return ders
    elif n == 2:
        fxx, fxy, fyy = ders
        if (not vectorial):
            shape = f.shape + (3,)
            hf = np.zeros(shape)
            hf[:, :, 0] = fxx
            hf[:, :, 1] = fxy
            hf[:, :, 2] = fyy
        elif vectorial:
            if not marginalize:
                shape = f.shape + (3,)
                hf = np.zeros(shape)
                hf[:, :, :, 0] = fxx
                hf[:, :, :, 1] = fxy
                hf[:, :, :, 2] = fyy
            else:
                if vectorial_first:
                    shape = f.shape[1:] + (3,)
                    axis = 0
                else:
                    shape = f.shape[:-1] + (3,)
                    axis = -1
                hf = np.zeros(shape)
                hf[:, :, :, 0] = fxx.sum(axis=axis)
                hf[:, :, :, 1] = fxy.sum(axis=axis)
                hf[:, :, :, 2] = fyy.sum(axis=axis)
    else:
        fxx, fxy, fxz, fyy, fyz, fzz = ders
        if not vectorial:
            shape = f.shape + (6,)
            hf = np.zeros(shape)
            hf[:, :, :, 0] = fxx
            hf[:, :, :, 1] = fxy
            hf[:, :, :, 2] = fxz
            hf[:, :, :, 3] = fyy
            hf[:, :, :, 4] = fyz
            hf[:, :, :, 5] = fzz
        else:
            if not marginalize:
                shape = f.shape + (6,)
                hf = np.zeros(shape)
                hf[:, :, :, :, 0] = fxx
                hf[:, :, :, :, 1] = fxy
                hf[:, :, :, :, 2] = fxz
                hf[:, :, :, :, 3] = fyy
                hf[:, :, :, :, 4] = fyz
                hf[:, :, :, :, 5] = fzz
            else:
                if vectorial_first:
                    shape = f.shape[1:] + (6,)
                    axis = 0
                else:
                    shape = f.shape[:-1] + (6,)
                    axis = -1
                hf = np.zeros(shape)
                hf[:, :, :, 0] = fxx.sum(axis=axis)
                hf[:, :, :, 1] = fxy.sum(axis=axis)
                hf[:, :, :, 2] = fxz.sum(axis=axis)
                hf[:, :, :, 3] = fyy.sum(axis=axis)
                hf[:, :, :, 4] = fyz.sum(axis=axis)
                hf[:, :, :, 5] = fzz.sum(axis=axis)

    return hf

def determinant_symmetric_field(f, vectorial=False, vectorial_first=False):
    """
    `determinant_symmetric_field` is not using Gaussian scale space but is supposed to be
    applied to fields obtained by `structure_tensor` and/or `hessian_of_gaussian`. This
    is why it may be convenient to have this function here, and why this imposes
    restrictions on dimensions.

    :param f: Array of dim 1 to 5, though 5 is a bad idea.
              For the non-vectorial case, i.e., vectorial = False (always true when ndim == 1):
              - f.ndim == 1: The function is the identity, it returns a reference to its
                argument. Note that vectorial_first == True triggers a ValueError exception.
              - f.ndim == 2: Cannot happen. Triggers a ValueError exception.
              - f.ndim == 3: This must be (m, n, 3), else triggers a ValueError exception.
                Returns a (m, n) field.
              - f.ndim == 4: This must be (m, n, p, 6), else triggers a ValueError exception.
                Returns a (m, n, p) field.
              - f.ndim == 5: Cannot happen, it will trigger a ValueError exception.
              For the vectorial case, i.e., vectorial == True:
              - f.ndim == 1: Cannot happen. Triggers ValueError exception.
              - f.ndim == 2: Vectorial_first ignored, the function is the identity, it
                returns a reference to its argument.
              - f.ndim == 3: Cannot happen. Triggers ValueError exception.
              - f.ndim == 4: Expects shape (m, n, v, 3) if vectorial_first or (v, m, n, 3) otherwise.
                Returns a (m, n, v) or (v, m, n) field.
              - f.ndim == 5: Expects shape (m, n, p, v, 6) if vectorial_first or (v, m, n, p, 6) otherwise.
                Returns a (m, n, p, v) or (v, m, n) field.
    :param vectorial: True indicates that f has a vectorial component, the last (default, vectorial_first = False)
                      or first (when vectorial_first == True).
    :param vectorial_first: See above.

    :returns: The determinant field.
    :rtype: numpy array
    """
    if (f.ndim <= 0) or (f.ndim > 5):
        raise TypeError('dimension must be in range {1..5}.')
    n = f.ndim if not vectorial else f.ndim - 1
    if not vectorial:
        if n == 1:
            return f
        if n == 2:
            raise ValueError('Argument cannot be a symmetric matrix field.')
        if n == 3:
            if f.shape[-1] != 3:
                raise ValueError('Argument cannot be a symmetric matrix field.')
            return f[:, :, 0] * f[:, :, 2] - f[:, :, 1] ** 2
        if n == 4:
            if f.shape[-1] != 6:
                raise ValueError('Argument cannot be a symmetric matrix field.')
            return f[:, :, :, 0] * f[:, :, :, 3] * f[:, :, :, 5] + \
                2 * f[:, :, :, 1] * f[:, :, :, 2] * f[:, :, :, 4] - \
                f[:, :, :, 1] ** 2 * f[:, :, :, 5] - \
                f[:, :, :, 2] ** 2 * f[:, :, :, 3] - \
                f[:, :, :, 4] ** 2 * f[:, :, :, 0]
        if n == 5:
            raise ValueError('Argument cannot be a 3D "scalar" symmetric matrix field.')
    else:
        if n == 1:
            raise ValueError('Argument cannot be a symmetric matrix field.')
        if n == 2:
            return f
        if n == 3:
            raise ValueError('Argument cannot be a symmetric matrix field.')
        if n == 4:
            if f.shape[-1] != 3:
                raise ValueError('Argument cannot be a symmetric matrix field.')
            return f[:, :, :, 0] * f[:, :, :, 2] - f[:, :, :, 1] ** 2
        if n == 5:
            if f.shape[-1] != 6:
                raise ValueError('Argument cannot be a symmetric matrix field.')
            return f[:, :, :, :, 0] * f[:, :, :, :, 3] * f[:, :, :, :, 5] + \
                2 * f[:, :, :, :, 1] * f[:, :, :, :, 2] * f[:, :, :, :,4] - \
                f[:, :, :, :, 1] ** 2 * f[:, :, :, :, 5] - \
                f[:, :, :, :, 2] ** 2 * f[:, :, :, :, 3] - \
                f[:, :, :, :, 4] ** 2 * f[:, :, :, :, 0]

def trace_symmetric_field(f, vectorial=False, vectorial_first=False):
    """
    `trace_symmetric_field` is not using Gaussian scale space but is supposed to be
    applied to fields obtained by `structure_tensor` and/or `hessian_of_gaussian`. This
    is why it may be convenient to have this function here, and why this imposes
    restrictions on dimensions.

    :param f: Array of dim 1 to 5, though 5 is a bad idea.
              For the non-vectorial case, i.e., vectorial = False (always true when ndim == 1):
              - f.ndim == 1: The function is the identity, it returns a reference to its
                argument. Note that vectorial_first == True triggers a ValueError exception.
              - f.ndim == 2: Cannot happen. Triggers a ValueError exception.
              - f.ndim == 3: This must be (m, n, 3), else triggers a ValueError exception.
                Returns a (m, n) field.
              - f.ndim == 4: This must be (m, n, p, 6), else triggers a ValueError exception.
                Returns a (m, n, p) field.
              - f.ndim == 5: Cannot happen, it will trigger a ValueError exception.
              For the vectorial case, i.e., vectorial == True:
              - f.ndim == 1: Cannot happen. Triggers ValueError exception.
              - f.ndim == 2: Vectorial_first ignored, the function is the identity, it
                returns a reference to its argument.
              - f.ndim == 3: Cannot happen. Triggers ValueError exception.
              - f.ndim == 4: Expects shape (m, n, v, 3) if vectorial_first or (v, m, n, 3) otherwise.
                Returns a (m, n, v) or (v, m, n) field.
              - f.ndim == 5: Expects shape (m, n, p, v, 6) if vectorial_first or (v, m, n, p, 6) otherwise.
                Returns a (m, n, p, v) or (v, m, n) field.
    :param vectorial: True indicates that f has a vectorial component, the last (default, vectorial_first = False)
                      or first (when vectorial_first == True).
    :param vectorial_first: See above.

    :returns: The trace field.
    :rtype: numpy array
    """
    if (f.ndim <= 0) or (f.ndim > 5):
        raise TypeError('dimension must be in range {1..5}.')
    n = f.ndim if not vectorial else f.ndim - 1
    if not vectorial:
        if n == 1:
            return f
        if n == 2:
            raise ValueError('Argument cannot be a symmetric matrix field.')
        if n == 3:
            if f.shape[-1] != 3:
                raise ValueError('Argument cannot be a symmetric matrix field.')
            return f[:, :, 0] + f[:, :, 2]
        if n == 4:
            if f.shape[-1] != 6:
                raise ValueError('Argument cannot be a symmetric matrix field.')
            return f[:, :, :, 0] + f[:, :, :, 3] + f[:, :, :, 5]
        if n == 5:
            raise ValueError('Argument cannot be a 3D "scalar" symmetric matrix field.')
    else:
        if n == 1:
            raise ValueError('Argument cannot be a symmetric matrix field.')
        if n == 2:
            return f
        if n == 3:
            raise ValueError('Argument cannot be a symmetric matrix field.')
        if n == 4:
            if f.shape[-1] != 3:
                raise ValueError('Argument cannot be a symmetric matrix field.')
            return f[:, :, :, 0] + f[:, :, :, 2]
        if n == 5:
            if f.shape[-1] != 6:
                raise ValueError('Argument cannot be a symmetric matrix field.')
            return f[:, :, :, :, 0] + f[:, :, :, :, 3] + f[:, :, :, :, 5]

def laplacian_of_gaussian(f, sigma, vectorial=False, vectorial_first=False, reflected=True, marginalize=False):
    """
    Laplacian of Gaussian.
    
    :param f: Array of dim 1, 2, 3, or 4. Dim 1 is always scalar while dim 4 is always vectorial.
    :param sigma: Gaussian scale for evaluation of derivatives. If float array-like, its length
                  must match f (non-vectorial) dimensions. This would be typically used to accommodate
                  non-square (resp. cubic) pixels (resp. voxels).
    :param vectorial: True means that the data is vectorial. The last dimension if vectorial_first is False
                      or the first one if vectorial_first is True.
                      When f.ndim is in {1, 4}, this is ignored.
    :param vectorial_first: True means first dim encodes the vectorial values,
                            False means last dim encodes the vectorial values, whenever it makes sense.
    :param reflected: If True, reflected boundary conditions will be applied.
                      If False, standard periodic are used (needs non-vectorial dimensions to be even).
    :param marginalize: If True and vectorial is True, returns Hessian per vectorial dimensions.
                        If False, sum result over vectorial dimensions.

    :returns: Laplacian of Gaussian array. If not marginalize and data array f is vectorial,
              or if data array f is scalar, same dimensions as f.
              If data array is vectorial and marginalize, the vectorial dimension disappears.
    :rtype: numpy array

    :raises ValueError: If dimensions are incorrect or mismatch.
    :raises TypeError: If input types are incorrect.
    """
    if (f.ndim <= 0) or (f.ndim > 4):
        raise TypeError('Only dim 1 to 3, with maybe vectorial content images can be used.')
    n = f.ndim if vectorial else f.ndim - 1
    if n == 0:
        raise ValueError('A dimension 1 array cannot be vectorial.')
    elif n == 4:
        raise ValueError('A dimension 4 array must be vectorial.')

    ders = gaussian_scalespace(f, sigma, order=2 * np.eye(n), vectorial=vectorial,
                               vectorial_first=vectorial_first, reflected=reflected)
    logf = reduce(add, ders)
    if marginalize and vectorial:
        axis = 0 if vectorial_first else -1
        np.sum(logf, axis=axis, out=logf)
    return logf
    

def mean_curvature_gaussian(f, sigma, reflected=True, eps=1e-8):
    """
    Compute mean curvature at scale sigma. f must be scalar
    otherwise it is just parallel computations of curvature.

    :param f: Array of dim 2, 3, or 4. Dim 1 is excluded while dim 4 is always vectorial.
    :param sigma: Gaussian scale for evaluation of derivatives. If float array-like, its length
                  must match f (non-vectorial) dimensions. This would be typically used to accommodate
                  non-square (resp. cubic) pixels (resp. voxels).
    :param reflected: If True, reflected boundary conditions will be applied.
                      If False, standard periodic are used (needs non-vectorial dimensions to be even).
    :param eps: Regularization of denominator if necessary.

    :returns: Mean curvature at Gaussian scale of array f. Scalar valued, i.e., same dimensions as f.
    :rtype: numpy array

    :raises ValueError: If dimensions are incorrect or mismatch.
    :raises TypeError: If input types are incorrect.
    """
    if (f.ndim <= 1) or (f.ndim > 3):
        raise TypeError('Only dim2 and 3, scalar arrays are accepted.')

    # Need all derivatives up to order 2
    if f.ndim == 2:
        order = ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    else:
        order = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2))

    ders = gaussian_scalespace(f, sigma, order=order, vectorial=False, reflected=reflected)
    if f.ndim == 2:
        fx, fy, fxx, fxy, fyy = ders
        mcf = (fx ** 2 * fyy - 2 * fx * fy * fxy + fy ** 2 * fxx) / (fx ** 2 + fy ** 2 + eps) ** 1.5
    else:
        fx, fy, fz, fxx, fxy, fxz, fyy, fyz, fzz = ders
        mcf = (fx ** 2 * (fyy + fzz) + fy ** 2 * (fxx + fzz) + fz ** 2 * (fxx + fyy) - 2 * (
                    fx * fy * fxy + fx * fz * fxz + fy * fz * fyz)) / \
              (fx ** 2 + fy ** 2 + fz ** 2 + eps) ** 1.5
    return mcf


# Test code for gaussian_scalespace
# TODO test the pure Fourier version!
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from math import (cos, sin)

    f = np.zeros((201, 201))
    f[100, 100] = 1.0
    sigma = 30.0

    orders = [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1],
              [0, 2], [3, 0], [2, 1], [1, 2], [0, 3]]
    f00, f10, f01, f20, f11, f02, f30, f21, f12, f03 = gaussian_scalespace(
        f, sigma=sigma, order=orders)

    fig = plt.figure(constrained_layout=True)
    fig.canvas.setWindowTitle('Gaussian pyramid')
    gs = fig.add_gridspec(nrows=8, ncols=8)
    ax00 = fig.add_subplot(gs[:2, 3:5])
    ax00.imshow(f00)
    # ax00.set_title('00')

    ax10 = fig.add_subplot(gs[2:4, 2:4])
    ax10.imshow(f10)
    # ax10.set_title('10')
    ax01 = fig.add_subplot(gs[2:4, 4:6])
    ax01.imshow(f01)
    # ax01.set_title('01')

    ax20 = fig.add_subplot(gs[4:6, 1:3])
    ax20.imshow(f20)
    # ax20.set_title('20')
    ax11 = fig.add_subplot(gs[4:6, 3:5])
    ax11.imshow(f11)
    # ax11.set_title('11')
    ax02 = fig.add_subplot(gs[4:6, 5:7])
    ax02.imshow(f02)
    # ax02.set_title('02')

    ax30 = fig.add_subplot(gs[6:, 0:2])
    ax30.imshow(f30)
    # ax30.set_title('30')
    ax21 = fig.add_subplot(gs[6:, 2:4])
    ax21.imshow(f21)
    # ax21.set_title('21')
    ax12 = fig.add_subplot(gs[6:, 4:6])
    ax12.imshow(f12)
    # ax12.set_title('12')
    ax03 = fig.add_subplot(gs[6:, 6:])
    ax03.imshow(f03)
    # ax03.set_title('03')

    for ax in [ax00, ax10, ax01, ax20, ax11, ax02, ax30, ax21, ax12, ax03]:
        ax.axis('off')



    orders = [[1,0],[0,1]]
    N = 6
    fx, fy = gaussian_scalespace(f, sigma=sigma/3.0, order=orders)
    I = np.zeros_like(f)
    fig, ax = plt.subplots(2,3)
    ax.shape = ax.size
    for i in range(N):
        ax[i].imshow(fx * cos(2 * i * _pi / N) + fy * sin(2 * i * _pi / N))
        ax[i].set_title(r'1st order, angle $\frac{%d\pi}{%d}$' % (2*i, N))
        ax[i].axis('off')
   


    plt.show()
