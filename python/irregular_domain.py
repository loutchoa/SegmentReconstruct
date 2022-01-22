"""
@project: psychic-memory
    
@file: irregular_domain.py
    
@description: routines for discrete gradient and divergences 
of functions on an irregular domain embedded in a regular grid.
    
@author: François Lauze, University of Copenhagen    
Created on Thurs Jan 20 16:48 2022

@TODO:
  + some FISTA-TV algorithms, or in a depending module
  + some in-place routines used in Chambolle-Pock (add-ascent_grad,
    add_descent_div...) and friends?
"""

import numpy as np

from matplotlib import pyplot as plt
from icecream import ic

__version__ = "0.0.1"
__author__ = "François Lauze"


class Stencil2D:
    def __init__(self, M):
        """
        Data for discrete gradient and divergence on a 2D irregular domain.

        Args:
            M 2D numpy array of bool: 
                domain mask / characteristic function
        """
        # index arrays used to efficient computations of
        # finite differences
        self.idx_by_end = None
        self.idx_by_start = None
        self.idx_by_central_part = None
        self.idx_bx_end = None
        self.idx_bx_start = None
        self.idx_bx_central_part = None
        self.idx_fy = None
        self.idx_fx = None

        self.domain = M
        self.dx, self.dy = M.shape
        self.x, self.y = np.where(M)
        self.coordinates = list(zip(self.x, self.y))
        self.l = len(self.coordinates)
        # lookup table to match 1D indices to coordinates
        # and the opposite
        self.F = -np.ones_like(M, dtype=int)
        self.F[self.x, self.y] = range(self.l)

        # compute the finite difference index arrays
        self.fdiff_indices()
        self.bdiff_indices()

    def flatten(self, f):
        """Flatten the representation of f."""
        return f[self.x, self.y]

    def unflatten(self, f):
        """Restore the 2D shape of a vectorised signal."""
        v_dim = f.shape[1:]
        uf_shape = (self.dx, self.dy) + v_dim if len(v_dim) > 0 else (self.dx, self.dy)
        uf = np.zeros(uf_shape, dtype=f.dtype)
        uf[self.x, self.y] = f
        return uf

    def fdiff_indices(self):
        """Get indices for forward differentiation."""
        self.idx_fx = []
        self.idx_fy = []
        for centre, (x, y) in enumerate(self.coordinates):
            east = -1 if x == self.dx - 1 else self.F[x + 1, y]
            north = -1 if y == self.dy - 1 else self.F[x, y + 1]
            if east != -1:
                self.idx_fx.append((centre, east))
            if north != -1:
                self.idx_fy.append((centre, north))
        self.idx_fx = np.array(self.idx_fx).T
        self.idx_fy = np.array(self.idx_fy).T

    def bdiff_indices(self):
        """
        Get indices for backward differentiation.

        Boundary conditions are relatively complicated compared to
        the ones from forward differences.
        """
        self.idx_bx_central_part = []
        self.idx_bx_start = []
        self.idx_bx_end = []
        self.idx_by_central_part = []
        self.idx_by_start = []
        self.idx_by_end = []

        for centre, (x, y) in enumerate(self.coordinates):
            east = -1 if x == self.dx - 1 else self.F[x + 1, y]
            west = -1 if x == 0 else self.F[x - 1, y]
            north = -1 if y == self.dy - 1 else self.F[x, y + 1]
            south = -1 if y == 0 else self.F[x, y - 1]

            if west == -1 and east == -1:
                pass
            elif west == -1:
                self.idx_bx_start.append(centre)
            elif east == -1:
                self.idx_bx_end.append((centre, west))
            else:
                self.idx_bx_central_part.append((centre, west))

            if south == -1 and north == -1:
                pass
            elif south == -1:
                self.idx_by_start.append(centre)
            elif north == -1:
                self.idx_by_end.append((centre, south))
            else:
                self.idx_by_central_part.append((centre, south))

        self.idx_bx_start = np.array(self.idx_bx_start)
        self.idx_bx_central_part = np.array(self.idx_bx_central_part).T
        self.idx_bx_end = np.array(self.idx_bx_end).T
        self.idx_by_start = np.array(self.idx_by_start)
        self.idx_by_central_part = np.array(self.idx_by_central_part).T
        self.idx_by_end = np.array(self.idx_by_end).T

    def fdiff_x_flattened(self, f):
        """Forward difference of vectorised f in x-direction."""
        fx = np.zeros_like(f)
        centre, east = self.idx_fx
        fx[centre] = f[east] - f[centre]
        return fx

    def fdiff_y_flattened(self, f):
        """Forward difference of vectorised f in y-direction."""
        fy = np.zeros_like(f)
        centre, north = self.idx_fy
        fy[centre] = f[north] - f[centre]
        return fy

    def gradient_flattened(self, f):
        """Forward differences gradient of vectorised f."""
        grad_f = np.zeros(f.shape + (2,), dtype=f.dtype)
        grad_f[..., 0] = self.fdiff_x_flattened(f)
        grad_f[..., 1] = self.fdiff_y_flattened(f)
        return grad_f

    def gradient_from_2D(self, f):
        """Vectorised forward differences gradient from 2D f."""
        return self.gradient_flattened(f[self.x, self.y])

    def gradient_2D(self, f):
        """2D forward differences gradient from 2D, for convenience."""
        return self.unflatten(self.gradient_from_2D(f))

    def bdiff_x_flattened(self, f, fx=None):
        """Backward difference of vectorised f in x-direction."""
        return_value = fx is None
        if return_value:
            fx = np.zeros_like(f)
        start = self.idx_bx_start
        fx[start] += f[start]
        centre, west = self.idx_bx_central_part
        fx[centre] += f[centre] - f[west]
        centre, west = self.idx_bx_end
        fx[centre] -= f[west]
        if return_value:
            return fx

    def bdiff_y_flattened(self, f, fy=None):
        """Backward difference of vectorised f in y-direction."""
        return_value = fy is None
        if return_value:
            fy = np.zeros_like(f)
        start = self.idx_by_start
        fy[start] += f[start]
        centre, south = self.idx_by_central_part
        fy[centre] += f[centre] - f[south]
        centre, south = self.idx_by_end
        fy[centre] -= f[south]
        if return_value:
            return fy

    def divergence_flattened(self, f):
        """Backward difference computation of divergence of field f."""
        div_f = np.zeros(f.shape[:-1], dtype=f.dtype)
        self.bdiff_x_flattened(f[..., 0], div_f)
        self.bdiff_y_flattened(f[..., 1], div_f)
        return div_f

    def divergence_from_2D(self, f):
        """Vectorised backward differences divergence from 2D field f."""
        return self.divergence_flattened(self.flatten(f))

    def divergence_2D(self, f):
        """2D backward differences divergence from 2D field, for convenience."""
        return self.unflatten((self.divergence_from_2D(f)))


class Stencil3D:
    def __init__(self, M):
        """
        Data for discrete gradient and divergence on a 3D irregular domain.

        Args:
            M 3D numpy array of bool:
                domain mask / characteristic function
        """
        # index arrays used to efficient computations of
        # finite differences
        self.idx_bz_end = None
        self.idx_bz_start = None
        self.idx_bz_central_part = None
        self.idx_by_end = None
        self.idx_by_start = None
        self.idx_by_central_part = None
        self.idx_bx_end = None
        self.idx_bx_start = None
        self.idx_bx_central_part = None
        self.idx_fz = None
        self.idx_fy = None
        self.idx_fx = None

        self.domain = M
        self.dx, self.dy, self.dz = M.shape
        self.x, self.y, self.z = np.where(M)
        self.coordinates = list(zip(self.x, self.y, self.z))
        self.l = len(self.coordinates)
        # lookup table to match 1D indices to coordinates
        # and the opposite
        self.F = -np.ones_like(M, dtype=int)
        self.F[self.x, self.y, self.z] = range(self.l)

        # compute the finite difference index arrays
        self.fdiff_indices()
        self.bdiff_indices()

    def flatten(self, f):
        """Flatten the representation of f."""
        return f[self.x, self.y, self.z]

    def unflatten(self, f):
        """Restore the 2D shape of a vectorised signal."""
        v_dim = f.shape[2:]
        uf_shape = (self.dx, self.dy, self.dz) + v_dim if len(v_dim) > 0 else (self.dx, self.dy, self.dz)
        uf = np.zeros(uf_shape, dtype=f.dtype)
        uf[self.x, self.y, self.dz] = f
        return uf

    def fdiff_indices(self):
        """Get indices for forward differentiation."""
        self.idx_fx = []
        self.idx_fy = []
        self.idx_fz = []

        for centre, (x, y, z) in enumerate(self.coordinates):
            east = -1 if x == self.dx - 1 else self.F[x + 1, y, z]
            north = -1 if y == self.dy - 1 else self.F[x, y + 1, z]
            up = -1 if z == self.dz - 1 else self.F[x, y, z + 1]
            if east != -1:
                self.idx_fx.append((centre, east))
            if north != -1:
                self.idx_fy.append((centre, north))
            if up != -1:
                self.idx_fz.append((centre, up))

        self.idx_fx = np.array(self.idx_fx).T
        self.idx_fy = np.array(self.idx_fy).T
        self.idx_fz = np.array(self.idx_fz).T

    def bdiff_indices(self):
        """
        Get indices for backward differentiation.

        Boundary conditions are relatively complicated compared to
        the ones from forward differences.
        """
        self.idx_bx_central_part = []
        self.idx_bx_start = []
        self.idx_bx_end = []
        self.idx_by_central_part = []
        self.idx_by_start = []
        self.idx_by_end = []
        self.idx_bz_central_part = []
        self.idx_bz_start = []
        self.idx_bz_end = []

        for centre, (x, y, z) in enumerate(self.coordinates):
            east = -1 if x == self.dx - 1 else self.F[x + 1, y, z]
            west = -1 if x == 0 else self.F[x - 1, y, z]
            north = -1 if y == self.dy - 1 else self.F[x, y + 1, z]
            south = -1 if y == 0 else self.F[x, y - 1, z]
            up = -1 if z == self.dz - 1 else self.F[x, y, z + 1]
            down = -1 if z == 0 else self.F[x, y, z -1]
            if west == -1 and east == -1:
                pass
            elif west == -1:
                self.idx_bx_start.append(centre)
            elif east == -1:
                self.idx_bx_end.append((centre, west))
            else:
                self.idx_bx_central_part.append((centre, west))

            if south == -1 and north == -1:
                pass
            elif south == -1:
                self.idx_by_start.append(centre)
            elif north == -1:
                self.idx_by_end.append((centre, south))
            else:
                self.idx_by_central_part.append((centre, south))

            if down == -1 and up == -1:
                pass
            elif down == -1:
                self.idx_bz_start.append(centre)
            elif up == -1:
                self.idx_bz_end.append((centre, down))
            else:
                self.idx_bz_central_part.append((centre, down))

        self.idx_bx_start = np.array(self.idx_bx_start)
        self.idx_bx_central_part = np.array(self.idx_bx_central_part).T
        self.idx_bx_end = np.array(self.idx_bx_end).T
        self.idx_by_start = np.array(self.idx_by_start)
        self.idx_by_central_part = np.array(self.idx_by_central_part).T
        self.idx_by_end = np.array(self.idx_by_end).T
        self.idx_bz_start = np.array(self.idx_bz_start)
        self.idx_bz_central_part = np.array(self.idx_bz_central_part).T
        self.idx_bz_end = np.array(self.idx_bz_end).T

    def fdiff_x_flattened(self, f):
        """Forward difference of vectorised f in x-direction."""
        fx = np.zeros_like(f)
        centre, east = self.idx_fx
        fx[centre] = f[east] - f[centre]
        return fx

    def fdiff_y_flattened(self, f):
        """Forward difference of vectorised f in y-direction."""
        fy = np.zeros_like(f)
        centre, north = self.idx_fy
        fy[centre] = f[north] - f[centre]
        return fy

    def fdiff_z_flattened(self, f):
        """Forward difference of vectorised f in z-direction."""
        fz = np.zeros_like(f)
        centre, up = self.idx_fz
        fz[centre] = f[up] - f[centre]
        return fz

    def gradient_flattened(self, f):
        """Forward differences gradient of vectorised f."""
        grad_f = np.zeros(f.shape + (3,), dtype=f.dtype)
        grad_f[..., 0] = self.fdiff_x_flattened(f)
        grad_f[..., 1] = self.fdiff_y_flattened(f)
        grad_f[...,2] = self.fdiff_z_flattened(f)
        return grad_f

    def gradient_from_3D(self, f):
        """Vectorised forward differences gradient from 3D f."""
        return self.gradient_flattened(f[self.x, self.y, self.z])

    def gradient_3D(self, f):
        """3D forward differences gradient from 3D, for convenience."""
        return self.unflatten(self.gradient_from_3D(f))

    def bdiff_x_flattened(self, f, fx=None):
        """Backward difference of vectorised f in x-direction."""
        return_value = fx is None
        if return_value:
            fx = np.zeros_like(f)
        start = self.idx_bx_start
        fx[start] += f[start]
        centre, west = self.idx_bx_central_part
        fx[centre] += f[centre] - f[west]
        centre, west = self.idx_bx_end
        fx[centre] -= f[west]
        if return_value:
            return fx

    def bdiff_y_flattened(self, f, fy=None):
        """Backward difference of vectorised f in y-direction."""
        return_value = fy is None
        if return_value:
            fy = np.zeros_like(f)
        start = self.idx_by_start
        fy[start] += f[start]
        centre, south = self.idx_by_central_part
        fy[centre] += f[centre] - f[south]
        centre, south = self.idx_by_end
        fy[centre] -= f[south]
        if return_value:
            return fy

    def bdiff_z_flattened(self, f, fz=None):
        """Backward difference of vectorised f in z-direction."""
        return_value = fz is None
        if return_value:
            fz = np.zeros_like(f)
        start = self.idx_bz_start
        fz[start] += f[start]
        centre, down = self.idx_bz_central_part
        fz[centre] += f[centre] - f[down]
        centre, down = self.idx_bz_end
        fz[centre] -= f[down]
        if return_value:
            return fz

    def divergence_flattened(self, f):
        """Backward difference computation of divergence of field f."""
        div_f = np.zeros(f.shape[:-1], dtype=f.dtype)
        self.bdiff_x_flattened(f[..., 0], div_f)
        self.bdiff_y_flattened(f[..., 1], div_f)
        self.bdiff_z_flattened(f[..., 2], div_f)
        return div_f

    def divergence_from_3D(self, f):
        """Vectorised backward differences divergence from 3D field f."""
        return self.divergence_flattened(self.flatten(f))

    def divergence_3D(self, f):
        """2D backward differences divergence from 3D field, for convenience."""
        return self.unflatten((self.divergence_from_3D(f)))










if __name__ == "__main__":
    from skimage import data
    # seed = 243521
    # np.random.seed(seed)

    m, n = 5, 5
    M = np.array(
        [[0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 0, 0 ,0]], dtype=bool
    )
    stencil = Stencil2D(M)
    print("x : ", stencil.x)
    print("y : ", stencil.y)

    f1 = np.random.randint(-5, 6, size=(m, n))
    f2 = np.random.randint(-5, 6, size=(m, n, 2))


    print(f"M = \n{M.astype(int)}")
    print(f"f1 = \n{f1*M}")
    print(f"f2[:, :, 0] = \n{f2[...,0]*M},\nf2[:, :, 1] = {f2[..., 1]*M}")
    Grad_f1 = stencil.gradient_2D(f1)
    Div_f2 = stencil.divergence_2D(f2)
    print(f"Grad_f1[:, :, 0] = \n{Grad_f1[..., 0]},\nGrad_f1[:, :, 1] = \n{Grad_f1[..., 1] * M}")
    print(f"Div_f2 = \n{Div_f2}")

    f_f1 = stencil.flatten(f1)
    f_f2 = stencil.flatten(f2)
    grad_f1 = stencil.flatten(Grad_f1)
    div_f2 = stencil.flatten(Div_f2)

    print("<df1, f2>  = ", (grad_f1 * f_f2).sum())
    print("<f1, divf2>  = ", (f_f1 * div_f2).sum())

    f = data.camera().astype(float)
    m, n = f.shape
    x, y = np.mgrid[-1:1:m * 1j, -1:1:n * 1j]
    M = x ** 2 + y ** 2 < 0.5
    stencil = Stencil2D(M)
    df = stencil.gradient_from_2D(f)
    Df = stencil.unflatten(df)
    lapf = stencil.divergence_flattened(df)
    Lapf = stencil.unflatten(lapf)

    gradient_figure, (axf, axfx, axfy, axlapf) = plt.subplots(1, 4, sharex=True, sharey=True)
    axf.imshow(f)
    axfx.imshow(Df[:, :, 0])
    axfy.imshow(Df[:, :, 1])
    axlapf.imshow(Lapf)

    plt.show()
