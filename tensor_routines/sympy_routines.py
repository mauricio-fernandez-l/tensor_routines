"""Symbolic routines

Author: Dr.-Ing. Mauricio Fernández

This module offers a collection of tensor routines in sympy.
"""

# %% Import

import sympy as spy
import numpy as np
from itertools import permutations

import tensor_routines as tr

# %% Generate tensors


def t(n: int, s: str = "a", dims: list = None) -> spy.Array:
    """Generate tensor.

    Generate n-th-order tensor based on string `s` for the
    components and dimensions `dims`.

    Parameters
    ----------
    n : int
        Tensor order
    s : str, optional
        Base string for components, by default "a"
    dims : list, optional
        List of tensor dimensions, by default None.
        * For `dims == None`, `dims` is set to `[3]*n`

    Returns
    -------
    spy.Array
        Tensor
    """
    if dims is None:
        dims = [3] * n
    head = f"{s}:"
    dims_pattern = ":".join([str(d) for d in dims])
    symbol_pattern = head + dims_pattern
    symbols = spy.symbols(symbol_pattern, real=True)
    return spy.Array(symbols, shape=dims)


# %% Transform


def flatten(a: spy.Array) -> spy.Array:
    """Flatten tensor

    Parameters
    ----------
    a : spy.Array
        Tensor

    Returns
    -------
    spy.Array
        Flattened tensor (row vector)
    """
    return a.reshape(spy.prod(a.shape))


def vec(a: spy.Array) -> spy.Matrix:
    """Column vector representation of tensor

    Parameters
    ----------
    a : spy.Array
        Tensor

    Returns
    -------
    spy.Matrix
        Tensor as vector in R^{n x 1}
    """
    return a.reshape(spy.prod(a.shape), 1).tomatrix()


# %% Products


def sp(a: spy.Array, b: spy.Array):
    """Scalar product.

    Compute the scalar product (full contraction) of
    tensors with equal shape.

    Parameters
    ----------
    a : spy.Array
        Tensor
    b : spy.Array
        Tensor with b.shape == a.shape

    Returns
    -------
    scalar
        a_ijkl... b_ijkl...

    Examples
    --------
    >>> a = t(3, "a", [2, 3, 4])
    ... b = t(3, "b", [2, 3, 4])
    ... c = sp(a, b) # c = a_ijk b_ijk
    """
    return vec(a).dot(vec(b))


def nf(a: spy.Array):
    """Frobenius norm.

    Frobenius norm of a real-valued tensor (square
    root of the sum of all squared tensor components).

    Parameters
    ----------
    a : spy.Array
        Tensor

    Returns
    -------
    scalar
        Frobenius norm
    """
    return vec(a).norm()


def td(a: spy.Array, b: spy.Array, n: int) -> spy.Array:
    if a.shape == b.shape and n == a.rank():
        return sp(a, b)
    else:
        return spy.Array(np.tensordot(a, b, n))


def tp(*args):
    temp = args[0]
    for i in range(1, len(args)):
        temp = spy.tensorproduct(temp, args[i])
    return temp


def tc(a: spy.Array, i: list):
    return spy.tensorcontraction(a, i)


def lm(a: spy.Array, b: spy.Array):
    return td(a, b, b.rank())


def rp(q: spy.Array, a: spy.Array):
    ra = a.rank()
    if ra == 1:
        return spy.Array(q.tomatrix() * vec(a))
    else:
        con = (1, ra + 2 - 1)
        temp = tc(tp(q, a), con)
        for _ in range(ra - 1):
            temp = tc(tp(q, temp), con)
        return temp


# %% Linear algebra


def linsolve_t(eqs: spy.Array, vs=None) -> list:
    if vs is None:
        vs = eqs.free_symbols
    if eqs.is_scalar:
        eqs = [eqs]
    else:
        eqs = flatten(eqs)
    sol = list(spy.linsolve(eqs, *vs))[0]
    return [(v, s) for v, s in zip(vs, sol)]


def orthogonalize_basis(basis: spy.Array) -> spy.Array:
    n = basis.shape[0]
    if basis.rank() > 2:
        dims = basis.shape[1:]
    gs = spy.GramSchmidt([vec(b) for b in basis], orthonormal=True)
    gs = spy.Array([spy.Array(g) for g in gs])
    if basis.rank() == 2:
        return gs
    else:
        return gs.reshape(n, *dims)


# %% Transpositions


def tt(a: spy.Array, axes) -> spy.Array:
    return spy.permutedims(a, axes)


def tt_m(a: spy.Array) -> spy.Array:
    ra = a.rank()
    front = list(range(ra // 2))
    back = list(range(ra // 2, ra))
    return tt(a, back + front)


# %% Symmetrizations


def symmetrize(a: spy.Array) -> spy.Array:
    p = list(permutations(list(range(a.rank()))))
    for i in p:
        eqs = a - spy.permutedims(a, i)
        vs = eqs.free_symbols
        if len(vs) > 0:
            a = a.subs(linsolve_t(eqs))
    return a


def symmetrize_explicit(a: spy.Array, p: list) -> spy.Array:
    eqs = a - spy.permutedims(a, p)
    vs = eqs.free_symbols
    if len(vs) > 0:
        a = a.subs(linsolve_t(eqs))
    return a


def symmetrize_l(a: spy.Array) -> spy.Array:
    return symmetrize_explicit(a, (1, 0, 2, 3))


def symmetrize_r(a: spy.Array) -> spy.Array:
    return symmetrize_explicit(a, (0, 1, 3, 2))


def symmetrize_m(a: spy.Array) -> spy.Array:
    return symmetrize_explicit(a, (2, 3, 0, 1))


def symmetrize_lr(a: spy.Array) -> spy.Array:
    return symmetrize_r(symmetrize_l(a))


def symmetrize_lrm(a: spy.Array) -> spy.Array:
    return symmetrize_m(symmetrize_lr(a))


def sym(a: spy.Array) -> spy.Array:
    out = 0 * a
    p = list(permutations(list(range(a.rank()))))
    for i in p:
        out += spy.permutedims(a, i)
    return out / len(p)


def sym_r(a: spy.Array) -> spy.Array:
    return (a + tt(a, (0, 1, 3, 2))) / 2


def sym_l(a: spy.Array) -> spy.Array:
    return (a + tt(a, (1, 0, 2, 3))) / 2


def sym_lr(a: spy.Array) -> spy.Array:
    return sym_r(sym_l(a))


# %% Voigt notation (not normalized)


def vn(a: spy.Array) -> spy.Matrix:
    if a.shape == (3, 3):
        out = spy.Matrix(
            [
                a[tr.VN_CONVENTION[0][0], tr.VN_CONVENTION[0][1]],
                a[tr.VN_CONVENTION[1][0], tr.VN_CONVENTION[1][1]],
                a[tr.VN_CONVENTION[2][0], tr.VN_CONVENTION[2][1]],
                a[tr.VN_CONVENTION[3][0], tr.VN_CONVENTION[3][1]],
                a[tr.VN_CONVENTION[4][0], tr.VN_CONVENTION[4][1]],
                a[tr.VN_CONVENTION[5][0], tr.VN_CONVENTION[5][1]],
            ]
        )
    else:
        out = spy.eye(6)
        for i_1 in range(6):
            for i_2 in range(6):
                out[i_1, i_2] = a[
                    tr.VN_CONVENTION[i_1][0],
                    tr.VN_CONVENTION[i_1][1],
                    tr.VN_CONVENTION[i_2][0],
                    tr.VN_CONVENTION[i_2][1],
                ]
    return out


# %% Normalized Voigt notation (NOT Voigt notation)


def nvn(a: spy.Array) -> spy.Matrix:
    """Normalized Voigt notation.

    Return the symmetric second- or minor symmetric fourth-order tensor
    `a` in normalized Voigt notation according to the convention defined
    by the module constant NVN_CONVENTION.

    The norm of the returned objects corresponds to the actual Frobenius
    norm of the original tensors. For fourth-order tensors, the inverse
    of the returned matrix corresponds to the `nvn` of the inverse of `a`
    (with respect to the linear map on symmetric second-order tensors).

    Parameters
    ----------
    a : spy.Array
        * Second order: a.shape == (3, 3) and symmetric
        * Fourth order: a.shape == (3, 3, 3, 3) and minor symmetric,
        i.e., a == sym_lr(a)

    Returns
    -------
    spy.Array
        * Second order: 6D vector representation
        * Fourth order: 6x6 matrix representation
    """
    if a.shape == (3, 3):
        out = spy.Matrix(
            [
                a[tr.VN_CONVENTION[0][0], tr.VN_CONVENTION[0][1]],
                a[tr.VN_CONVENTION[1][0], tr.VN_CONVENTION[1][1]],
                a[tr.VN_CONVENTION[2][0], tr.VN_CONVENTION[2][1]],
                a[tr.VN_CONVENTION[3][0], tr.VN_CONVENTION[3][1]] * SR2,
                a[tr.VN_CONVENTION[4][0], tr.VN_CONVENTION[4][1]] * SR2,
                a[tr.VN_CONVENTION[5][0], tr.VN_CONVENTION[5][1]] * SR2,
            ]
        )
    else:
        out = spy.eye(6)
        for i_1 in range(6):
            for i_2 in range(6):
                out[i_1, i_2] = a[
                    tr.VN_CONVENTION[i_1][0],
                    tr.VN_CONVENTION[i_1][1],
                    tr.VN_CONVENTION[i_2][0],
                    tr.VN_CONVENTION[i_2][1],
                ]
                if i_1 > 2:
                    out[i_1, i_2] *= SR2
                if i_2 > 2:
                    out[i_1, i_2] *= SR2
    return out


def nvn_inv(a: spy.Array) -> spy.Array:
    """Inverse of normalized Voigt notation.

    Reconstruct symmetric second- or minor symmetric fourth-order
    tensor from `nvn` representation.

    Parameters
    ----------
    a : spy.Array
        * 6D vector
        * 6x6 second-order tensor

    Returns
    -------
    spy.Array
        * symmetric 3x3 second-order tensor
        * minor symmetric 3x3x3x3x fourth-order tensor
    """
    if a.shape == (6, 1):
        out = spy.MutableDenseNDimArray(np.zeros([3] * 2))
        for i in range(3):
            out[tr.VN_CONVENTION[i][0], tr.VN_CONVENTION[i][1]] = a[i]
        for i in range(3, 6):
            out[tr.VN_CONVENTION[i][0], tr.VN_CONVENTION[i][1]] = a[i] / SR2
            out[tr.VN_CONVENTION[i][1], tr.VN_CONVENTION[i][0]] = a[i] / SR2
    else:
        out = spy.MutableDenseNDimArray(np.zeros([3] * 4))
        for i_1 in range(6):
            for i_2 in range(6):
                temp = a[i_1, i_2]
                if i_1 > 2:
                    temp /= SR2
                if i_2 > 2:
                    temp /= SR2
                out[
                    tr.VN_CONVENTION[i_1][0],
                    tr.VN_CONVENTION[i_1][1],
                    tr.VN_CONVENTION[i_2][0],
                    tr.VN_CONVENTION[i_2][1],
                ] = temp
                out[
                    tr.VN_CONVENTION[i_1][1],
                    tr.VN_CONVENTION[i_1][0],
                    tr.VN_CONVENTION[i_2][0],
                    tr.VN_CONVENTION[i_2][1],
                ] = temp
                out[
                    tr.VN_CONVENTION[i_1][0],
                    tr.VN_CONVENTION[i_1][1],
                    tr.VN_CONVENTION[i_2][1],
                    tr.VN_CONVENTION[i_2][0],
                ] = temp
                out[
                    tr.VN_CONVENTION[i_1][1],
                    tr.VN_CONVENTION[i_1][0],
                    tr.VN_CONVENTION[i_2][1],
                    tr.VN_CONVENTION[i_2][0],
                ] = temp
    out = out.as_immutable()
    return out


def inv_nvn(a: spy.Array) -> spy.Array:
    """Compute inverse through nvn.

    Compute inverse of minor symmetric fourth-order tensor
    through `nvn`. The inverse applies only on symmetric
    second-order tensors.

    Parameters
    ----------
    a : spy.Array
        Fourth-order minor symmetrc tensor with a.shape == (3, 3, 3, 3)

    Returns
    -------
    spy.Array
        Inverse of `a` w.r.t. symmetric second-order tensors
    """
    a = nvn(a)
    if a == a.T:
        a = a.inv(method="CH")
    else:
        a = a.inv()
    return nvn_inv(a)


# %% Material tensors: stiffness


def stiffness_cub_l(l_1: float, l_2: float, l_3: float) -> spy.Array:
    return td(spy.Array([l_1, l_2, l_3]), P_CUB, 1)


def stiffness_cub_get_l(stiffness: spy.Array) -> spy.Array:
    return spy.Array([sp(stiffness, P) / sp(P, P) for P in P_CUB])


def stiffness_cub(c_1111: float, c_1122: float, c_2323: float) -> spy.Array:
    l_1 = c_1111 + 2 * c_1122
    l_2 = c_1111 - c_1122
    l_3 = 2 * c_2323
    return stiffness_cub_l(l_1, l_2, l_3)


def stiffness_iso_l(l_1: float, l_2: float) -> spy.Array:
    return l_1 * P_ISO_1 + l_2 * P_ISO_2


def stiffness_iso(E: float, nu: float) -> spy.Array:
    l_1 = E / (1 - 2 * nu)
    l_2 = E / (1 + nu)
    return stiffness_iso_l(l_1, l_2)


# %% Rotations


def rotation_matrix(n: spy.Array, phi: float) -> spy.Array:
    n = n / nf(n)
    out = spy.cos(phi) * ID_2 - spy.sin(phi) * lm(PT, n) + (1 - spy.cos(phi)) * tp(n, n)
    return out


# %% Harmonic tensors


def harmonic(n: int, s: str = "h") -> spy.Array:
    if n == 0:
        return spy.symbols(s)
    elif n == 1:
        return t(1, s)
    else:
        a = symmetrize(t(n, s))
        return a.subs(linsolve_t(lm(a, ID_2)))


def harmonic_basis(n: int) -> spy.Array:
    h = harmonic(n)
    vs = list(h.free_symbols)
    return spy.Array([spy.diff(h, v) for v in vs])


def harmonic_onb(n: int) -> list:
    return orthogonalize_basis(harmonic_basis(n))


# %% Projections


def project_on_basis(a: spy.Array, basis: spy.Array) -> spy.Array:
    cs = spy.Array(spy.symbols(f"c:{basis.shape[0]}"))
    lc = td(cs, basis, 1)
    return cs.subs(linsolve_t(a - lc, cs))


# %% Algebraic decompositions


def iso(a: spy.Array) -> spy.Array:
    if a.rank() == 2:
        return (a[0, 0] + a[1, 1] + a[2, 2]) / 3 * ID_2
    elif a.rank() == 4:
        ls = [sp(a, P) / sp(P, P) for P in P_ISO]
        return ls[0] * P_ISO_1 + ls[1] * P_ISO_2 + ls[2] * P_ISO_3
    else:
        raise Exception("Not implemented")


def aniso(a: spy.Array) -> spy.Array:
    return a - iso(a)


def skw(a: spy.Array) -> spy.Array:
    return a - sym(a)


# %% Differentiation


def grad(a: spy.Array, x: spy.Array) -> spy.Array:
    temp = spy.derive_by_array(a, x)
    ra = a.rank()
    rx = x.rank()
    axes_x = list(range(rx))
    axes_a = list(range(rx, rx + ra))
    return tt(temp, axes_a + axes_x)


def diff(a: spy.Array, *xs) -> spy.Array:
    if isinstance(xs[-1], int):
        xs = [xs[0]] * xs[-1]
    g = a
    for x in xs:
        g = grad(g, x)
    return g


# %% Module constants

# Scalars
SR2 = spy.sqrt(2)

# Identity on vectors
ID_2 = spy.Array(spy.eye(3))

# Permutation tensor
PT = spy.MutableDenseNDimArray(np.zeros([3, 3, 3], dtype=int))
PT[0, 1, 2] = 1
PT[1, 2, 0] = 1
PT[2, 0, 1] = 1
PT[0, 2, 1] = -1
PT[2, 1, 0] = -1
PT[1, 0, 2] = -1

# Fourth-order
ITI = tp(ID_2, ID_2)
ID_4 = tt(ITI, (0, 2, 1, 3))
ID_S = sym_r(ID_4)
ID_A = ID_4 - ID_S
P_ISO_1 = ITI / 3
P_ISO_2 = ID_S - P_ISO_1
P_ISO_3 = ID_A
P_ISO = [P_ISO_1, P_ISO_2, P_ISO_3]
P_CUB_1 = P_ISO_1
D_CUB = spy.MutableDenseNDimArray(np.zeros(shape=[3] * 4, dtype=int))
for ii in range(3):
    D_CUB[ii, ii, ii, ii] = 1
P_CUB_2 = D_CUB - P_CUB_1
P_CUB_3 = ID_S - (P_CUB_1 + P_CUB_2)
P_CUB = spy.Array([P_CUB_1, P_CUB_2, P_CUB_3])
