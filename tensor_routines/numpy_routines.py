# %% Import

import numpy as np

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

    Recursively calls `np.tensordot(a, b, 0)` for argument list `args = [a0, a1, a2, ...]`,
    yielding, e.g.,
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
    p = [n-1] + list(range(n-1)) # index permutation per Q
    for i in range(n):
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
        at_{i_3 i_1 i_0 i_2} = a_{i_0 i_1 i_2 i_3} <- rename to get dimension transposition
    
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
        * For fourth-order tensors `a = sum_{i=1}^3 lambda_i P_ISOi + aniso_part`,
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

def iso_t(l: np.ndarray) -> np.ndarray:
    """Isotropic tensor.

    Create an isotropic second- or fourth-order tensor from the 
    given eigenvalue(s).

    Parameters
    ----------
    l : float or np.ndarray
        * Second order: float corresponding to lambda for lambda ID_2
        * Fourth order: 1D-array = [lambda_1,2|3] for sum_{i=1}^3 lambda_i P_ISO_i
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
    if isinstance(l, (list, np.ndarray)):
        # Fourth-order
        if len(l) == 2:
            return l[0]*P_ISO_1 + l[1]*P_ISO_2
        else:
            return l[0]*P_ISO_1 + l[1]*P_ISO_2 + l[2]*P_ISO_3
    else:
        # Second-order
        return l*ID_2

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
        return iso_t([1/l for l in iso_ev(a) if l != 0])
    else:
        # Second order
        return iso_t(1/iso_ev(a))


# %% Normalized Voigt notation (NOT Voigt notation)

def nvn(a: np.ndarray) -> np.ndarray:
    """Normalized Voigt notation.

    Return the symmetric second- or minor symmetric fourth-order tensor 
    `a` in normalized Voigt notation according to the convention 
    (Python indices):

        * [0, 0] 
        * [1, 1] 
        * [2, 2] 
        * sqrt(2)*[1, 2] 
        * sqrt(2)*[0, 2] 
        * sqrt(2)*[0, 1]

    The norm of the returned objects corresponds to the actual Frobenius
    norm of the original tensors. For fourth-order tensors, the inverse
    of the returned matrix corresponds to the `nvn` of the inverse of `a`
    (with respect to the linear map on symmetric second-order tensors).

    Parameters
    ----------
    a : np.ndarray
        * Second order: a.shape == (3, 3) and symmetric
        * Fourth order: a.shape == (3, 3, 3, 3) and minor symmetric, i.e., a == sym_lr(a)

    Returns
    -------
    np.ndarray
        * Second order: 6D vector representation
        * Fourth order: 6x6 matrix representation
    """
    convention = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]]
    if a.shape == (3, 3):
        out = np.array([
            a[0, 0],
            a[1, 1],
            a[2, 2],
            a[1, 2]*SR2,
            a[0, 2]*SR2,
            a[0, 1]*SR2
        ])
    else:
        out = np.eye(6)
        for i_1 in range(6):
            for i_2 in range(6):
                out[i_1, i_2] = a[
                    convention[i_1][0],
                    convention[i_1][1],
                    convention[i_2][0],
                    convention[i_2][1]
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
    convention = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]]
    if a.shape == (6,):
        out = np.array([
            a[0], a[5]/SR2, a[4]/SR2,
            a[5]/SR2, a[1], a[3]/SR2,
            a[4]/SR2, a[3]/SR2, a[2]
            ]).reshape(3, 3)
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
                    convention[i_1][0],
                    convention[i_1][1],
                    convention[i_2][0],
                    convention[i_2][1],
                ] = temp
                out[
                    convention[i_1][1],
                    convention[i_1][0],
                    convention[i_2][0],
                    convention[i_2][1],
                ] = temp
                out[
                    convention[i_1][0],
                    convention[i_1][1],
                    convention[i_2][1],
                    convention[i_2][0],
                ] = temp
                out[
                    convention[i_1][1],
                    convention[i_1][0],
                    convention[i_2][1],
                    convention[i_2][0],
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

# %% T4: stiffness/compliance tensors of important symmetry groups

def t4_tric_nvn(
    t_1111, t_1122, t_1133, t_1123, t_1113, t_1112,
    t_2222, t_2233, t_2223, t_2213, t_2212,
    t_3333, t_3323, t_3313, t_3312,
    t_2323, t_2313, t_2312,
    t_1313, t_1312,
    t_1212 
    ) -> np.ndarray:
    """Generate triclinic tensor in `nvn` with given components.

    Parameters
    ----------
    t_1111 : float
    t_1122 : float
    t_1133 : float
    t_1123 : float
    t_1113 : float
    t_1112 : float
    t_2222 : float
    t_2233 : float
    t_2223 : float
    t_2213 : float
    t_2212 : float
    t_3333 : float
    t_3323 : float
    t_3313 : float
    t_3312 : float
    t_2323 : float
    t_2313 : float
    t_2312 : float
    t_1313 : float
    t_1312 : float
    t_1212 : float

    Returns
    -------
    np.ndarray
        6x6 matrix corresponding to `nvn`
    """
    out = np.array([
        [t_1111, t_1122, t_1133, SR2*t_1123, SR2*t_1113, SR2*t_1112],
        [t_1122, t_2222, t_2233, SR2*t_2223, SR2*t_2213, SR2*t_2212],
        [t_1133, t_2233, t_3333, SR2*t_3323, SR2*t_3313, SR2*t_3312],
        [SR2*t_1123, SR2*t_2223, SR2*t_3323, 2*t_2323, 2*t_2313, 2*t_2312],
        [SR2*t_1113, SR2*t_2213, SR2*t_3313, 2*t_2313, 2*t_1313, 2*t_1312],
        [SR2*t_1112, SR2*t_2212, SR2*t_3312, 2*t_2312, 2*t_1312, 2*t_1212]
    ])
    return out

def t4_tric(*args) -> np.ndarray:
    """Generate triclinic fourth-order tensor.

    Generate from 21 components of `nvn` variant.

    Parameters
    ----------
    args : 21 components of triclinic tensor
        row-wise in `nvn`

    Returns
    -------
    np.ndarray
        Fourth-order triclinic tensor with minor symmetries
    """
    return nvn_inv(t4_tric_nvn(*args))

def t4_hex_nvn(
    t_1111: float, 
    t_1122: float, 
    t_1133: float, 
    t_3333: float, 
    t_2323: float
    ):
    """Generate hexagonal tensor in `nvn`.

    Generate hexagonal/transversaly isotropic fourth-order
    tensor with symmetry axis e_3 in `nvn`.

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
        6x6 `nvn` representation of fourth-order hexagonal tensor
    """
    # Isotropy axis is e_3
    return t4_tric_nvn(
        t_1111, t_1122, t_1133, 0, 0, 0,
        t_1111, t_1133, 0, 0, 0,
        t_3333, 0, 0, 0,
        t_2323, 0, 0,
        t_2323, 0,
        (t_1111 - t_1122)/2
        )

def t4_hex(
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
    return nvn_inv(t4_hex_nvn(t_1111, t_1122, t_1133, t_3333, t_2323))

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
    n0 = np.cos(phi)*ID_2
    n1 = -np.sin(phi)*lm(PT,n)
    n2 = (1 - np.cos(phi))*tp(n,n)
    return n0 + n1 + n2

# %% Constants

# Scalars
SR2 = np.sqrt(2)

# Identity on vectors
ID_2 = np.eye(3)

# Permutation tensor
PT = np.zeros((3,3,3))
PT[0,1,2] = 1
PT[1,2,0] = 1
PT[2,0,1] = 1
PT[1,0,2] = -1
PT[2,1,0] = -1
PT[0,2,1] = -1

# Fourth-order
ITI = tp(ID_2, ID_2)
ID_4 = tt(ITI, (0, 2, 1, 3))
ID_S = sym_r(ID_4)
ID_A = ID_4 - ID_S
P_ISO_1 = ITI/3
P_ISO_2 = ID_S - P_ISO_1
P_ISO_3 = ID_A
P_ISO = [P_ISO_1, P_ISO_2, P_ISO_3]

# nvn variants
ID_S_NVN = nvn(ID_S)