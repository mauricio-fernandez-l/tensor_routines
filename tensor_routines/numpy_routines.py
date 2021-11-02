"""Numeric routines

Author: Dr.-Ing. Mauricio Fernández

This module offers a collection of tensor routines using numpy.
"""

# %% Import

import numpy as np

import tensor_routines as tr

# %% Products


def td(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Tensor dot.

    Shortcut for `np.tensordot(a, b, n)`.

    Parameters
    ----------
    a : np.ndarray
        Tensor
    b : np.ndarray
        Tensor
    n : int
        Number of neighbour axes to be contracted

    Returns
    -------
    np.ndarray
        Result

    Examples
    --------
    Contract 2 neighbour axes

    >>> a = np.random.rand(2, 3, 4, 5)
    ... b = np.random.rand(4, 5, 6, 7)
    ... c = td(a, b, 2) # c_ijkl = a_ijmn b_mnkl
    ... print(c.shape)
    (2, 3, 6, 7)
    """
    return np.tensordot(a, b, n)


def sp(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar product.

    Compute the scalar product (full contraction) of
    tensors with equal shape.

    Parameters
    ----------
    a : np.ndarray
        Tensor
    b : np.ndarray
        Tensor with b.shape == a.shape

    Returns
    -------
    float
        a_ijkl... b_ijkl...

    Examples
    --------
    >>> a = np.random.rand(2, 3, 4)
    ... b = np.random.rand(2, 3, 4)
    ... c = sp(a, b) # c = a_ijk b_ijk
    """
    return td(a, b, a.ndim)


def nf(a: np.ndarray) -> float:
    """Frobenius norm.

    Frobenius norm of a real-valued tensor (square
    root of the sum of all squared tensor components)
    by calling `np.linalg.norm(a)`.

    Parameters
    ----------
    a : np.ndarray
        Tensor

    Returns
    -------
    float
        Frobenius norm

    Examples
    --------
    >>> a = np.random.rand(2, 3, 4)
    ... nf(a) - np.sqrt(np.sum(a**2))
    """
    return np.linalg.norm(a)


def tp(*args) -> np.ndarray:
    """Tensor product.

    Recursively calls `np.tensordot(a, b, 0)` for argument list
    `args = [a0, a1, a2, ...]`, yielding, e.g.,
        tp(a0, a1, a2) = tp(tp(a0, a1), a2)

    Parameters
    ----------
    args : sequence
        Sequence of tensors

    Returns
    -------
    np.ndarray
        Tensor product

    Examples
    --------

    >>> a = np.random.rand(2, 3, 4)
    ... b = np.random.rand(7, 8, 9)
    ... c = tp(a, b) # c_ijkmno = a_ijk b_mno
    ... c.shape == (2, 3, 4, 7, 8, 9)
    """
    temp = args[0]
    for i in range(1, len(args)):
        temp = np.tensordot(temp, args[i], 0)
    return temp


def lm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear map.

    Compute linear map by contracting in order
    all axes of b with the last axes of a, e.g.,
    a_ijklm and b_nop, c = lm(a, b) with
    c_ij = a_ijmno b_mno.

    Parameters
    ----------
    a : np.ndarray
        Tensor
    b : np.ndarray
        Tensor with a.ndim >= b.ndim and matching commong axes

    Returns
    -------
    np.ndarray
        Linear map

    Examples
    --------
    >>> a = np.random.rand(2, 3, 4, 5, 6)
    ... b = np.random.rand(4, 5, 6)
    ... c = lm(a, b)
    ... c.shape == (2, 3)
    True
    """
    return td(a, b, b.ndim)


def rp(q: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Rayleight product.

    Compute Rayleight product `b = rp(q, a)` defined as
        b_{i_1 i_2 ...} = q_{i_1 j_1} q_{i_2 j_2} ... a_{j_1 j_2 ...}
    where the second order tensor q in R^{n_{out} x n_{in}}
    is used to map
        a in R^{n_{in} x n_{in} x ...}
    to
        b in R^{n_{out} x n_{out} x ...}

    Parameters
    ----------
    q : np.ndarray
        Second-order tensor
    a : np.ndarray
        Tensor with constant axis dimension, e.g., a.shape = (4, 4, 4, 4)

    Returns
    -------
    np.ndarray
        Rayleigh product
    """
    n = a.ndim
    p = [n-1] + list(range(n-1))  # index permutation per Q
    for _ in range(n):
        a = np.transpose(td(a, np.transpose(q), 1), p)
    return a


# %% Transpositions

def tt(a: np.ndarray, t: list) -> np.ndarray:
    """Tensor dimension transposition.

    Shortcut for np.transpose.

    Parameters
    ----------
    a : np.ndarray
        Any numpy array
    t : list
        List of dimension transpositions with a.ndim == len(t)

    Returns
    -------
    np.ndarray
        Transposed array


    Examples
    --------
    >>> a = np.random.rand(2, 3, 4, 5)
    ... tt(a, (2, 0, 3, 1)).shape == (4, 2, 5, 3)
    True

    Personal note
    -------------
    np transposes dimensions, NOT indices. It can be
    considered as the index transpose of the result `at`
    (see examples below).

    Pattern:
        a in R(d_0, d_1, ...)
        t = (t_0, t_1, ...)
        d_t = (d_{t_0}, d_{t_1}, ...)
        i_t = (i_{t_0}, i_{t_1}, ...)
        at = tr(a, t) in R(*d_t)
        at_{*i_t} = a_{i_0 i_1 ...}

    Example 1:
        a in R(d_0, d_1, d_2, d_3)
        at = tr(a, (3, 1, 0, 2)) in R(d_3, d_1, d_0, d_2)
        at_{i_0 i_1 i_2 i_3} = a_{i_2 i_1 i_3 i_0}
        at_{i_3 i_1 i_0 i_2} = a_{i_0 i_1 i_2 i_3} <- rename to
            get dimension transposition

    Example 2:
        a in R(d_0, d_1, d_2, d_3)
        at = tr(a, (1, 3, 2, 0)) in R(d_1, d_3, d_2, d_0)
        at_{i_0 i_1 i_2 i_3} = a_{i_3 i_0 i_2 i_1}
        at_{i_1 i_3 i_2 i_0} = a_{i_0 i_1 i_2 i_3} <-

    With respect to Mathematica (index transposition):
    a in R(d_0, d_1, d_2, d_3)
    tr(a, (3, 1, 0, 2)) == TensorTranspose(a, {2, 1, 3, 0} + 1)
    tr(a, (1, 3, 2, 0)) == TensorTranspose(a, {3, 0, 2, 1} + 1)
    TensorTranspose(tr(a, t), t+1) == a
    """
    return np.transpose(a, t)


def tt_r(a: np.ndarray) -> np.ndarray:
    """Right index pair transposition

    Transpose right index pair of fourth-order tensor a.

    Parameters
    ----------
    a : np.ndarray
        Fourth-order tensor, a_ijkl

    Returns
    -------
    np.ndarray
        Fourth-order tensor, a_jikl
    """
    return tt(a, (0, 1, 3, 2))


def tt_l(a: np.ndarray) -> np.ndarray:
    """Left index pair transposition.

    Transpose left index pair of fourth-order tensor a.

    Parameters
    ----------
    a : np.ndarray
        Fourth-order tensor, a_ijkl

    Returns
    -------
    np.ndarray
        Fourth-order tensor, a_ijlk
    """
    return tt(a, (1, 0, 2, 3))


# %% Symmetrizations

def sym_2(a: np.ndarray) -> np.ndarray:
    """Symmetric part of second-order tensor.

    Parameters
    ----------
    a : np.ndarray
        Second-order tensor

    Returns
    -------
    np.ndarray
        Symmetric part
    """
    return (a + a.T)/2


def sym_r(a: np.ndarray) -> np.ndarray:
    """Symmetrize right index pair.

    Symmetrize a fourth-order tensor a_ijkl with respect to
    its right index pair.

    Parameters
    ----------
    a : np.ndarray
        Fourth-order tensor, a_ijkl

    Returns
    -------
    np.ndarray
        Fourth-order tensor, symmetric with respect to kl
    """
    return (a + tt_r(a))/2


def sym_l(a: np.ndarray) -> np.ndarray:
    """Symmetrize left index pair.

    Symmetrize a fourth-order tensor a_ijkl with respect to
    its left index pair.

    Parameters
    ----------
    a : np.ndarray
        Fourth-order tensor, a_ijkl

    Returns
    -------
    np.ndarray
        Fourth-order tensor, symmetric with respect to ij
    """
    return (a + tt_l(a))/2


def sym_lr(a: np.ndarray) -> np.ndarray:
    """Symmetrize left and right index pair.

    Symmetrize a fourth-order tensor a_ijkl with respect to
    its left and right index pair.

    Parameters
    ----------
    a : np.ndarray
        Fourth-order tensor, a_ijkl

    Returns
    -------
    np.ndarray
        Fourth-order tensor, symmetric with respect to ij and kl
    """
    return sym_r(sym_l(a))

# %% Isotropic


def iso_ev(a: np.ndarray) -> np.ndarray:
    """Project eigenvalues of isotropic tensor.

    Compute for a given second- or fourth-order tensor the
    eigenvalues of the isotropic part of the given tensor `a`
        * For second-order tensors for `a = lambda ID_2 + aniso_part`,
            `lambda` is computed.
        * For fourth-order tensors
            `a = sum_{i=1}^3 lambda_i P_ISOi + aniso_part`,
            the lambda_i are computed.

    Parameters
    ----------
    a : np.ndarray
        Isotropic tensor.
        * Second order: a.shape == (3, 3)
        * Fourth order: a.shape == (3, 3, 3, 3)

    Returns
    -------
    np.ndarray
        * Second order: lambda
        * Fourth order: 1D-array with [lambda_1,2,3]

    Examples
    --------
    >>> iso_ev(7*np.eye(3)) == 7
    True
    >>> (iso_ev(13*P_ISO_1 + 20*P_ISO_2) == np.array([13, 20])).all()
    True
    """
    if a.shape == (3, 3, 3, 3):
        # Fourth-order
        return np.array([sp(a, P)/sp(P, P) for P in P_ISO])
    else:
        # Second-order
        return sp(a, ID_2)/3


def iso_t(lam: np.ndarray) -> np.ndarray:
    """Isotropic tensor.

    Create an isotropic second- or fourth-order tensor from the
    given eigenvalue(s).

    Parameters
    ----------
    lam : float or np.ndarray
        * Second order: float corresponding to lambda for lambda ID_2
        * Fourth order: 1D-array = [lambda_1,2|3] for
            sum_{i=1}^3 lambda_i P_ISO_i
            * [lambda_1, lambda_2] for only two first isotropic projects
            * [lambda_1,2,3] for all isotropic projectors

    Returns
    -------
    np.ndarray
        Isotropic tensor of second or fourth order.

    Examples
    --------
    >>> (iso_t(10) == 10*np.eye(3)).all()
    True
    >>> (iso_t([3, 7]) == 3*P_ISO_1 + 7*P_ISO_2).all()
    True
    """
    if isinstance(lam, (list, np.ndarray)):
        # Fourth-order
        if len(lam) == 2:
            return lam[0]*P_ISO_1 + lam[1]*P_ISO_2
        else:
            return lam[0]*P_ISO_1 + lam[1]*P_ISO_2 + lam[2]*P_ISO_3
    else:
        # Second-order
        return lam*ID_2


def iso_proj(a: np.ndarray) -> np.ndarray:
    """Isotropic projection.

    Compute the isotropic projection/part of the second-
    or fourth-order tensor `a`.

    Parameters
    ----------
    a : np.ndarray
        Second- or fourth-order tensor

    Returns
    -------
    np.ndarray
        Isotropic projection/part of `a`
    """
    return iso_t(iso_ev(a))


def iso_inv(a: np.ndarray) -> np.ndarray:
    """Inverse of isotropic tensor.

    Compute the inverse or pseudo-inverse of the isotropic
    second- or fourth-order tensor `a`.

    Parameters
    ----------
    a : np.ndarray
        Isotropic second- or fourth-order tensor

    Returns
    -------
    np.ndarray
        Inverse or pseudo-inverse

    Examples
    --------
    >>> ls = np.array([11, 7, 0])
    ... a = tn.iso_t(ls)
    ... ai = tn.iso_inv(a)
    ... tol = 1e-13
    ... assert nf(td(a, ai, 2) - tn.ID_S)/nf(tn.ID_S) < tol
    True
    """
    if a.shape == (3, 3, 3, 3):
        # Fourth order
        return iso_t([1/la for la in iso_ev(a) if la != 0])
    else:
        # Second order
        return iso_t(1/iso_ev(a))


# %% Voigt notation (not normalized)

# def set_vn_convention(sel="original"):
#     global VN_CONVENTION
#     if sel == "original":
#         print("Convention for Voigt notation: original")
#         VN_CONVENTION = [list(pair) for pair in VN_CONVENTION_ORIGINAL]
#     elif sel == "abaqus":
#         print("Convention for Voigt notation: abaqus")
#         VN_CONVENTION = [list(pair) for pair in VN_CONVENTION_ABAQUS]
#     else:
#         raise Exception(f"Convention {sel} not implemented")


def vn(a: np.ndarray) -> np.ndarray:
    """Voigt notation (based on convention of VN_CONVENTION)

    Parameters
    ----------
    a : np.ndarray
        Second- or fourth-order tensor

    Returns
    -------
    np.ndarray
        Corresponding first- or second-order output in Voigt notation
    """
    if a.shape == (3, 3):
        out = np.array([
            a[tr.VN_CONVENTION[0][0], tr.VN_CONVENTION[0][1]],
            a[tr.VN_CONVENTION[1][0], tr.VN_CONVENTION[1][1]],
            a[tr.VN_CONVENTION[2][0], tr.VN_CONVENTION[2][1]],
            a[tr.VN_CONVENTION[3][0], tr.VN_CONVENTION[3][1]],
            a[tr.VN_CONVENTION[4][0], tr.VN_CONVENTION[4][1]],
            a[tr.VN_CONVENTION[5][0], tr.VN_CONVENTION[5][1]]
        ])
    else:
        out = np.eye(6)
        for i_1 in range(6):
            for i_2 in range(6):
                out[i_1, i_2] = a[
                    tr.VN_CONVENTION[i_1][0],
                    tr.VN_CONVENTION[i_1][1],
                    tr.VN_CONVENTION[i_2][0],
                    tr.VN_CONVENTION[i_2][1]
                ]
    return out


# %% Normalized Voigt notation (NOT Voigt notation)

def nvn(a: np.ndarray) -> np.ndarray:
    """Normalized Voigt notation.

    Return the symmetric second- or minor symmetric fourth-order tensor
    `a` in normalized Voigt notation according to the convention defined
    in the module by the constant NVN_CONVENTION.

    The norm of the returned objects corresponds to the actual Frobenius
    norm of the original tensors. For fourth-order tensors, the inverse
    of the returned matrix corresponds to the `nvn` of the inverse of `a`
    (with respect to the linear map on symmetric second-order tensors).

    Parameters
    ----------
    a : np.ndarray
        * Second order: a.shape == (3, 3) and symmetric
        * Fourth order: a.shape == (3, 3, 3, 3) and minor symmetric,
            i.e., a == sym_lr(a)

    Returns
    -------
    np.ndarray
        * Second order: 6D vector representation
        * Fourth order: 6x6 matrix representation
    """
    if a.shape == (3, 3):
        out = np.array([
            a[tr.VN_CONVENTION[0][0], tr.VN_CONVENTION[0][1]],
            a[tr.VN_CONVENTION[1][0], tr.VN_CONVENTION[1][1]],
            a[tr.VN_CONVENTION[2][0], tr.VN_CONVENTION[2][1]],
            a[tr.VN_CONVENTION[3][0], tr.VN_CONVENTION[3][1]]*SR2,
            a[tr.VN_CONVENTION[4][0], tr.VN_CONVENTION[4][1]]*SR2,
            a[tr.VN_CONVENTION[5][0], tr.VN_CONVENTION[5][1]]*SR2
        ])
    else:
        out = np.eye(6)
        for i_1 in range(6):
            for i_2 in range(6):
                out[i_1, i_2] = a[
                    tr.VN_CONVENTION[i_1][0],
                    tr.VN_CONVENTION[i_1][1],
                    tr.VN_CONVENTION[i_2][0],
                    tr.VN_CONVENTION[i_2][1]
                ]
                if i_1 > 2:
                    out[i_1, i_2] *= SR2
                if i_2 > 2:
                    out[i_1, i_2] *= SR2
    return out


def nvn_inv(a: np.ndarray) -> np.ndarray:
    """Inverse of normalized Voigt notation.

    Reconstruct symmetric second- or minor symmetric fourth-order
    tensor from `nvn` representation.

    Parameters
    ----------
    a : np.ndarray
        * 6D vector
        * 6x6 second-order tensor

    Returns
    -------
    np.ndarray
        * symmetric 3x3 second-order tensor
        * minor symmetric 3x3x3x3x fourth-order tensor
    """
    if a.shape == (6,):
        out = np.zeros(shape=[3, 3])
        for i in range(3):
            out[
                tr.VN_CONVENTION[i][0],
                tr.VN_CONVENTION[i][1]
            ] = a[i]
        for i in range(3, 6):
            out[
                tr.VN_CONVENTION[i][0],
                tr.VN_CONVENTION[i][1]
            ] = a[i]/SR2
            out[
                tr.VN_CONVENTION[i][1],
                tr.VN_CONVENTION[i][0]
            ] = a[i]/SR2
    else:
        out = np.zeros((3, 3, 3, 3))
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
    return out


def inv_nvn(a: np.ndarray) -> np.ndarray:
    """Compute inverse through nvn.

    Compute inverse of minor symmetric fourth-order tensor
    through `nvn`. The inverse applies only on symmetric
    second-order tensors.

    Parameters
    ----------
    a : np.ndarray
        Fourth-order minor symmetrc tensor with a.shape == (3, 3, 3, 3)

    Returns
    -------
    np.ndarray
        Inverse of `a` w.r.t. symmetric second-order tensors
    """
    return nvn_inv(np.linalg.inv(nvn(a)))

# %% T1


def nv_polar(phi: float) -> np.ndarray:
    """Normal vector based on polar angle

    Parameters
    ----------
    phi : float
        Polar angle

    Returns
    -------
    np.ndarray
        2D normal vector
    """
    x = np.cos(phi)
    y = np.sin(phi)
    return np.array([x, y])


def nv_spherical(theta: float, phi: float) -> np.ndarray:
    """Normal vector based on spherical angles

    Parameters
    ----------
    theta : float
        Angle in [0, pi], start from z axis [0, 0, 1]
    phi : float
        Angle in x-y-plance, in [0, 2*pi]

    Returns
    -------
    np.ndarray
        Normal vector in 3D
    """
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

# %% T4: stiffness/compliance tensors of important symmetry groups


def stiffness_component_dict(components=None) -> dict:
    """Generate stiffness component dictionary

    Generate dictionary based on the 21 values passed
    and based on the active Voigt notation (see
    VN_CONSTANT in module)

    Parameters
    ----------
    components : float, optional
        The 21 components, by default None

    Returns
    -------
    dict
        Dictionary with index and component as key-value pairs.
    """
    out = {}
    counter = 0
    for i_1 in range(6):
        for i_2 in range(i_1, 6):
            k = tr.VN_CONVENTION[i_1] + tr.VN_CONVENTION[i_2]
            k = np.array(k) + 1
            k = "".join(str(k_) for k_ in k)
            if not isinstance(components, type(None)):
                out[k] = components[counter]
                counter += 1
            else:
                out[k] = 0
    return out


def stiffness_tric(
    components: np.ndarray = None,
    components_d: dict = None
) -> np.ndarray:
    """Generate triclinic fourth-order stiffness tensor.

    Parameters
    ----------
    components : np.ndarray
        21 components of triclinic tensor, see
        stiffness_component_dict
    components_d : dictionary
        dictionary with 21 components
        of triclinic tensor, see
        stiffness_component_dict

    Returns
    -------
    np.ndarray
        Fourth-order triclinic tensor with minor
        and major symmetries
    """
    out = np.zeros(shape=[3, 3, 3, 3])
    if not isinstance(components, type(None)):
        components_d = stiffness_component_dict(components)
    for k, v in components_d.items():
        i = [int(s)-1 for s in k]
        out[i[0], i[1], i[2], i[3]] = v
        # tt_l
        out[i[1], i[0], i[2], i[3]] = v
        # tt_r
        out[i[0], i[1], i[3], i[2]] = v
        out[i[1], i[0], i[3], i[2]] = v  # + tt_l
        # tt_m
        out[i[2], i[3], i[0], i[1]] = v
        out[i[3], i[2], i[0], i[1]] = v  # + tt_l
        out[i[2], i[3], i[1], i[0]] = v  # + tt_r
        out[i[3], i[2], i[1], i[0]] = v  # + tt_l + tt_r
    return out


def stiffness_hex(
    t_1111: float,
    t_1122: float,
    t_1133: float,
    t_3333: float,
    t_2323: float
) -> np.ndarray:
    """Generate hexagonal fourht-order tensor

    Generate hexagonal/transversaly isotropic fourth order
    tensor with symmetry axis e_3.

    Parameters
    ----------
    t_1111 : float
    t_1122 : float
    t_1133 : float
    t_3333 : float
    t_2323 : float

    Returns
    -------
    np.ndarray
        Hexagonal fourth-order tensor with shape (3, 3, 3, 3)
    """
    # Isotropy axis is e_3
    components_d = {
        "1111": t_1111,
        "1122": t_1122,
        "1133": t_1133,
        "1123": 0,
        "1113": 0,
        "1112": 0,
        "2222": t_1111,
        "2233": t_1133,
        "2223": 0,
        "2213": 0,
        "2212": 0,
        "3333": t_3333,
        "3323": 0,
        "3313": 0,
        "3312": 0,
        "2323": t_2323,
        "2313": 0,
        "2312": 0,
        "1313": t_2323,
        "1312": 0,
        "1212": (t_1111 - t_1122)/2
    }
    out = stiffness_tric(components_d=components_d)
    return out


def stiffness_cub_l(l_1: float, l_2: float, l_3: float) -> np.ndarray:
    """Generate cubic stiffness based on its 3 eigenvalues

    Parameters
    ----------
    l_1 : float
        First eigenvalue (1x)
    l_2 : float
        Second eigenvalue (2x)
    l_3 : float
        Third eigenvalue (3x)

    Returns
    -------
    np.ndarray
        Cubic stiffness, fourth-order tensor
    """
    return td(np.array([l_1, l_2, l_3]), P_CUB, 1)


def stiffness_cub_get_l(stiffness: np.ndarray) -> np.ndarray:
    """Get eigenvalues of cubic stiffness

    Parameters
    ----------
    stiffness : np.ndarray
        Cubic stiffness, fourth-order tensor

    Returns
    -------
    np.ndarray
        The 3 cubic eigenvalues
    """
    return np.array([sp(stiffness, P)/sp(P, P) for P in P_CUB])


def stiffness_cub(
    c_1111: float,
    c_1122: float,
    c_2323: float
) -> np.ndarray:
    """Generate cubic stiffness based on free components.

    Parameters
    ----------
    c_1111 : float
        Free component
    c_1122 : float
        Free component
    c_2323 : float
        Free component

    Returns
    -------
    np.ndarray
        Stiffness, fourth-order tensor
    """
    l_1 = c_1111 + 2*c_1122
    l_2 = c_1111 - c_1122
    l_3 = 2*c_2323
    return stiffness_cub_l(l_1, l_2, l_3)


def stiffness_iso_l(l_1: float, l_2: float) -> np.ndarray:
    """Generate an isotropic stiffness based on eigenvalues.

    Parameters
    ----------
    l_1 : float
        First eigenvalue (=3K)
    l_2 : float
        Second eigenvalue (=2G)

    Returns
    -------
    np.ndarray
        Stiffness, fourth-order tensor
    """
    return iso_t([l_1, l_2])


def stiffness_iso(E: float, nu: float) -> np.ndarray:
    """Generate an isotropic stiffness based on Young's modulus E
    and Poisson's ration nu.

    Parameters
    ----------
    E : float
        Young's modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    np.ndarray
        Stiffness, fourth-order tensor
    """
    l_1 = E/(1-2*nu)
    l_2 = E/(1+nu)
    return iso_t([l_1, l_2])

# %% Rotations


def rotation_matrix(n: np.ndarray, phi: float) -> np.ndarray:
    """Rotation matrix.

    Generate rotation matrix with rotation axis `n` and
    rotation angle `phi`.

    Parameters
    ----------
    n : np.ndarray
        Direction of ration axis (does not have to be normalized)
    phi : float
        Rotation angle in [0, np.pi)

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    n = np.array(n)
    n = n/nf(n)
    n_0 = np.cos(phi)*ID_2
    n_1 = -np.sin(phi)*lm(PT, n)
    n_2 = (1 - np.cos(phi))*tp(n, n)
    return n_0 + n_1 + n_2

# %% Constants


# Scalars
SR2 = np.sqrt(2)

# Identity on vectors
ID_2 = np.eye(3)

# Permutation tensor
PT = np.zeros((3, 3, 3))
PT[0, 1, 2] = 1
PT[1, 2, 0] = 1
PT[2, 0, 1] = 1
PT[1, 0, 2] = -1
PT[2, 1, 0] = -1
PT[0, 2, 1] = -1

# Fourth-order
ITI = tp(ID_2, ID_2)
ID_4 = tt(ITI, (0, 2, 1, 3))
ID_S = sym_r(ID_4)
ID_A = ID_4 - ID_S
P_ISO_1 = ITI/3
P_ISO_2 = ID_S - P_ISO_1
P_ISO_3 = ID_A
P_ISO = np.array([P_ISO_1, P_ISO_2, P_ISO_3])
P_CUB_1 = P_ISO_1
D_CUB = np.zeros(shape=[3]*4)
for ii in range(3):
    D_CUB[ii, ii, ii, ii] = 1
P_CUB_2 = D_CUB - P_CUB_1
P_CUB_3 = ID_S - (P_CUB_1 + P_CUB_2)
P_CUB = np.array([P_CUB_1, P_CUB_2, P_CUB_3])

# nvn variants
ID_S_NVN = nvn(ID_S)
