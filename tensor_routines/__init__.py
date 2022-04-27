"""
Tensor routines
===============

Author: Dr.-Ing. Mauricio Fernández

This package offers my personal collection of tensor routines for continuum
mechanics and other disciplines.

Examples
========

Create a numeric stiffness tensor of an isotropic or cubic material based on
its 2 and 3 respective eigenvalues

>>> import tensor_routines.numpy_routines as tn
... stiffness_iso = tn.stiffness_iso_l(1, 2)
... stiffness_cub = tn.stiffness_cub_l(1, 2, 3)

Create a 4th-order symbolic tensor, symmetrize it left and right, and print its
normalized Voigt notation matrix

>>> import tensor_routines.sympy_routines as ts
... a = ts.t(4, "a")
... a = ts.symmetrize_lr(a)
... a = ts.nvn(a)
... print(a)

"""

__version__ = "0.1.0"

# %% Import

import numpy as np

# %% Globals - initialize

# Voigt notation conventions
VN_CONVENTIONS = {}
VN_CONVENTIONS["original"] = np.array([
    [1, 1],  # diagonals
    [2, 2],
    [3, 3],
    [2, 3],  # off-diagonals
    [1, 3],
    [1, 2]
]) - 1
VN_CONVENTIONS["abaqus"] = np.array([
    [1, 1],  # diagonals
    [2, 2],
    [3, 3],
    [1, 2],  # off-diagonals
    [1, 3],
    [2, 3]
]) - 1
VN_CONVENTION = []

# %% Auxiliary routines


def set_vn_convention(convention="original", info=False):
    global VN_CONVENTION
    VN_CONVENTION = [list(pair) for pair in VN_CONVENTIONS[convention]]
    if info:
        print(f"Global index convention for Voigt notation: {convention}")
        print("[printing indices starting from 1]")
        print(np.array(VN_CONVENTION) + 1)


def get_vn_convention(convention: str = None):
    if convention is None:
        indices = VN_CONVENTION
    else:
        indices = VN_CONVENTIONS[convention]
    return indices


def get_vn_current_convention(info=True):
    if info:
        for k, v in VN_CONVENTIONS.items():
            if np.all(v == VN_CONVENTION):
                print(
                    f"Currently used index convention for Voigt notation: {k}")
                print("[printing indices starting from 1]")
                print(np.array(VN_CONVENTION) + 1)
    return VN_CONVENTION


# %% Set globals

set_vn_convention("abaqus")
