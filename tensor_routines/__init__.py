"""Tensor routines

Author: Dr.-Ing. Mauricio Fernández

This package offers my personal collection of tensor routines
for continuum mechanics and other disciplines.
"""

# %% Import

import numpy as np

# %% Globals - initialize

# Voigt notation convention
VN_CONVENTION_ORIGINAL = np.array([
    [1, 1],  # diagonals
    [2, 2],
    [3, 3],
    [2, 3],  # off-diagonals
    [1, 3],
    [1, 2]
]) - 1
VN_CONVENTION_ABAQUS = np.array([
    [1, 1],  # diagonals
    [2, 2],
    [3, 3],
    [1, 2],  # off-diagonals
    [1, 3],
    [2, 3]
]) - 1
VN_CONVENTION = []

# %% Auxiliary routines


def set_vn_convention(conv="original"):
    global VN_CONVENTION
    if conv == "original":
        data = VN_CONVENTION_ORIGINAL
    elif conv == "abaqus":
        data = VN_CONVENTION_ABAQUS
    else:
        raise Exception(f"Convention {conv} not implemented")
    print(f"Global convention for Voigt notation: {conv}")
    VN_CONVENTION = [list(pair) for pair in data]
    print(np.array(VN_CONVENTION) + 1)


# %% Set globals

set_vn_convention("abaqus")
